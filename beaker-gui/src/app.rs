use crate::recent_files::{RecentFiles, RecentItemType};
use crate::views::{DetectionView, WelcomeAction, WelcomeView};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    /// Pending file dialog result from menu (None = no dialog, Some(None) = cancelled, Some(Some(path)) = selected)
    pending_menu_file_dialog: Arc<Mutex<Option<Option<PathBuf>>>>,
    /// Pending folder dialog result from menu
    pending_menu_folder_dialog: Arc<Mutex<Option<Option<PathBuf>>>>,
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
            pending_menu_file_dialog: Arc::new(Mutex::new(None)),
            pending_menu_folder_dialog: Arc::new(Mutex::new(None)),
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
                } else if event.id == muda::MenuId::new("open_image") {
                    eprintln!("[BeakerApp] Native menu: Open Image clicked");
                    self.spawn_menu_file_dialog();
                } else if event.id == muda::MenuId::new("open_folder") {
                    eprintln!("[BeakerApp] Native menu: Open Folder clicked");
                    self.spawn_menu_folder_dialog();
                }
            }
        }
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        eprintln!("[BeakerApp] Menu: Open Image... clicked");
                        self.spawn_menu_file_dialog();
                        ui.close_menu();
                    }
                    if ui.button("Open Folder...").clicked() {
                        eprintln!("[BeakerApp] Menu: Open Folder... clicked");
                        self.spawn_menu_folder_dialog();
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

    /// Spawn a file dialog from menu in a separate thread to avoid blocking the UI
    fn spawn_menu_file_dialog(&self) {
        let dialog_result = Arc::clone(&self.pending_menu_file_dialog);
        std::thread::spawn(move || {
            eprintln!("[BeakerApp] Menu file dialog thread started");
            let path = rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png"])
                .add_filter("Beaker metadata", &["toml"])
                .pick_file();
            eprintln!("[BeakerApp] Menu file dialog result: {:?}", path);
            *dialog_result.lock().unwrap() = Some(path);
        });
    }

    /// Spawn a folder dialog from menu in a separate thread to avoid blocking the UI
    fn spawn_menu_folder_dialog(&self) {
        let dialog_result = Arc::clone(&self.pending_menu_folder_dialog);
        std::thread::spawn(move || {
            eprintln!("[BeakerApp] Menu folder dialog thread started");
            let path = rfd::FileDialog::new().pick_folder();
            eprintln!("[BeakerApp] Menu folder dialog result: {:?}", path);
            *dialog_result.lock().unwrap() = Some(path);
        });
    }

    fn open_image(&mut self, path: PathBuf) {
        eprintln!("[BeakerApp] Opening image: {:?}", path);
        match DetectionView::new(path.to_str().unwrap()) {
            Ok(view) => {
                eprintln!("[BeakerApp] Image loaded successfully, switching to Detection view");
                let _ = self.recent_files.add(path.clone(), RecentItemType::Image);
                self.state = AppState::Detection(view);
            }
            Err(e) => {
                eprintln!("[BeakerApp] ERROR: Failed to load image: {}", e);
            }
        }
    }

    fn open_folder(&mut self, path: PathBuf) {
        eprintln!("[BeakerApp] Opening folder: {:?}", path);
        // TODO: Implement folder/bulk mode in future (Proposal A)
        let _ = self.recent_files.add(path.clone(), RecentItemType::Folder);
        eprintln!("[BeakerApp] WARNING: Folder mode not yet implemented");
    }
}

impl eframe::App for BeakerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for completed menu dialogs (non-blocking)
        // Extract paths first, then process after releasing the lock
        let file_path = if let Ok(mut result) = self.pending_menu_file_dialog.try_lock() {
            if let Some(path_option) = result.take() {
                if let Some(path) = path_option {
                    eprintln!(
                        "[BeakerApp] Menu file dialog completed with path: {:?}",
                        path
                    );
                    Some(path)
                } else {
                    eprintln!("[BeakerApp] Menu file dialog cancelled");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let folder_path = if let Ok(mut result) = self.pending_menu_folder_dialog.try_lock() {
            if let Some(path_option) = result.take() {
                if let Some(path) = path_option {
                    eprintln!(
                        "[BeakerApp] Menu folder dialog completed with path: {:?}",
                        path
                    );
                    Some(path)
                } else {
                    eprintln!("[BeakerApp] Menu folder dialog cancelled");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Process paths after releasing locks
        if let Some(path) = file_path {
            self.open_image(path);
        }

        if let Some(path) = folder_path {
            self.open_folder(path);
        }

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
                        eprintln!("[BeakerApp] Received action: OpenImage({:?})", path);
                        self.open_image(path);
                    }
                    WelcomeAction::OpenFolder(path) => {
                        eprintln!("[BeakerApp] Received action: OpenFolder({:?})", path);
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
