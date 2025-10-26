use crate::recent_files::{RecentFiles, RecentItemType};
use crate::views::{DetectionView, WelcomeAction, WelcomeView};
use std::path::PathBuf;

pub trait View {
    #[allow(dead_code)]
    fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui);
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

impl View for DetectionView {
    fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        DetectionView::show(self, ctx, ui);
    }

    fn name(&self) -> &str {
        "Detection"
    }
}

enum AppState {
    Welcome(WelcomeView),
    Detection(DetectionView),
}

pub struct BeakerApp {
    state: AppState,
    recent_files: RecentFiles,
    use_native_menu: bool,
    #[cfg(target_os = "macos")]
    menu: Option<muda::Menu>,
    #[cfg(target_os = "macos")]
    menu_rx: Option<std::sync::mpsc::Receiver<muda::MenuEvent>>,
}

impl BeakerApp {
    pub fn new(use_native_menu: bool, image_path: Option<String>) -> Self {
        let mut recent_files = RecentFiles::default();

        let state = if let Some(path) = image_path {
            match DetectionView::new(&path) {
                Ok(view) => {
                    // Add to recent files
                    let _ = recent_files.add(PathBuf::from(&path), RecentItemType::Image);
                    AppState::Detection(view)
                }
                Err(e) => {
                    eprintln!("Failed to load image: {}", e);
                    AppState::Welcome(WelcomeView::new())
                }
            }
        } else {
            AppState::Welcome(WelcomeView::new())
        };

        Self {
            state,
            recent_files,
            use_native_menu,
            #[cfg(target_os = "macos")]
            menu: None,
            #[cfg(target_os = "macos")]
            menu_rx: None,
        }
    }

    #[cfg(target_os = "macos")]
    pub fn set_menu(&mut self, menu: muda::Menu, rx: std::sync::mpsc::Receiver<muda::MenuEvent>) {
        self.menu = Some(menu);
        self.menu_rx = Some(rx);
    }

    #[cfg(target_os = "macos")]
    fn poll_menu_events(&mut self, _ctx: &egui::Context) {
        if let Some(rx) = &self.menu_rx {
            while let Ok(event) = rx.try_recv() {
                if event.id == muda::MenuId::new("quit") {
                    std::process::exit(0);
                }
            }
        }
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        if let Some(path) = Self::open_file_dialog() {
                            self.open_image(path);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Open Folder...").clicked() {
                        if let Some(path) = Self::open_folder_dialog() {
                            self.open_folder(path);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        std::process::exit(0);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Detection").clicked() {
                        ui.close_menu();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn open_file_dialog() -> Option<PathBuf> {
        // On macOS, use async dialogs which work properly with the event loop
        #[cfg(target_os = "macos")]
        {
            let future = rfd::AsyncFileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png"])
                .add_filter("Beaker metadata", &["toml"])
                .pick_file();
            pollster::block_on(future).map(|f| f.path().to_path_buf())
        }

        // On other platforms, use synchronous dialogs
        #[cfg(not(target_os = "macos"))]
        {
            rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png"])
                .add_filter("Beaker metadata", &["toml"])
                .pick_file()
        }
    }

    fn open_folder_dialog() -> Option<PathBuf> {
        // On macOS, use async dialogs which work properly with the event loop
        #[cfg(target_os = "macos")]
        {
            let future = rfd::AsyncFileDialog::new().pick_folder();
            pollster::block_on(future).map(|f| f.path().to_path_buf())
        }

        // On other platforms, use synchronous dialogs
        #[cfg(not(target_os = "macos"))]
        {
            rfd::FileDialog::new().pick_folder()
        }
    }

    fn open_image(&mut self, path: PathBuf) {
        match DetectionView::new(path.to_str().unwrap()) {
            Ok(view) => {
                let _ = self.recent_files.add(path.clone(), RecentItemType::Image);
                self.state = AppState::Detection(view);
            }
            Err(e) => {
                eprintln!("Failed to load image: {}", e);
            }
        }
    }

    fn open_folder(&mut self, path: PathBuf) {
        // TODO: Implement folder/bulk mode in future (Proposal A)
        let _ = self.recent_files.add(path.clone(), RecentItemType::Folder);
        eprintln!("Folder mode not yet implemented: {:?}", path);
    }
}

impl eframe::App for BeakerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle native menu events on macOS
        #[cfg(target_os = "macos")]
        self.poll_menu_events(ctx);

        // Show egui menu if not using native
        if !self.use_native_menu {
            self.show_menu_bar(ctx);
        }

        egui::CentralPanel::default().show(ctx, |ui| match &mut self.state {
            AppState::Welcome(welcome_view) => {
                let action = welcome_view.show(ctx, ui);
                match action {
                    WelcomeAction::OpenImage(path) => {
                        self.open_image(path);
                    }
                    WelcomeAction::OpenFolder(path) => {
                        self.open_folder(path);
                    }
                    WelcomeAction::None => {}
                }
            }
            AppState::Detection(detection_view) => {
                detection_view.show(ctx, ui);
            }
        });
    }
}
