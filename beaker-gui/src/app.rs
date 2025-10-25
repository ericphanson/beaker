use crate::views::{DetectionView, WelcomeAction, WelcomeView};

pub trait View {
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
    use_native_menu: bool,
    #[cfg(target_os = "macos")]
    menu: Option<muda::Menu>,
    #[cfg(target_os = "macos")]
    menu_rx: Option<std::sync::mpsc::Receiver<muda::MenuEvent>>,
}

impl BeakerApp {
    pub fn new(use_native_menu: bool, image_path: Option<String>) -> Self {
        let state = if let Some(path) = image_path {
            match DetectionView::new(&path) {
                Ok(view) => AppState::Detection(view),
                Err(e) => {
                    eprintln!("Failed to load image: {}", e);
                    AppState::Welcome(WelcomeView::default())
                }
            }
        } else {
            AppState::Welcome(WelcomeView::default())
        };

        Self {
            state,
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
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Images", &["jpg", "jpeg", "png"])
                            .pick_file()
                        {
                            self.open_image(path);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Open Folder...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
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
                    if ui.button("Welcome").clicked() {
                        self.state = AppState::Welcome(WelcomeView::default());
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

    fn open_image(&mut self, path: std::path::PathBuf) {
        match DetectionView::new(&path.display().to_string()) {
            Ok(view) => {
                // Add to recent files if we have a welcome view
                if let AppState::Welcome(ref mut welcome) = self.state {
                    let _ = welcome.add_recent_file(path.clone(), false);
                }
                self.state = AppState::Detection(view);
            }
            Err(e) => {
                eprintln!("Failed to load image: {}", e);
                if let AppState::Welcome(ref mut welcome) = self.state {
                    welcome.set_error(format!("Failed to load image: {}", e));
                }
            }
        }
    }

    fn open_folder(&mut self, _path: std::path::PathBuf) {
        // TODO: Implement directory processing view (Proposal A)
        // For now, show error
        if let AppState::Welcome(ref mut welcome) = self.state {
            welcome.set_error("Folder processing not yet implemented".to_string());
        }
    }

    fn handle_welcome_action(&mut self, action: WelcomeAction) {
        match action {
            WelcomeAction::OpenImage(path) => self.open_image(path),
            WelcomeAction::OpenFolder(path) => self.open_folder(path),
            WelcomeAction::OpenPath(path) => {
                if path.is_dir() {
                    self.open_folder(path);
                } else {
                    self.open_image(path);
                }
            }
        }
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

        egui::CentralPanel::default().show(ctx, |ui| {
            match &mut self.state {
                AppState::Welcome(welcome) => {
                    if let Some(action) = welcome.show(ctx, ui) {
                        self.handle_welcome_action(action);
                    }
                }
                AppState::Detection(detection) => {
                    detection.show(ctx, ui);
                }
            }
        });
    }
}
