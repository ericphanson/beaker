use crate::views::DetectionView;

pub trait View {
    fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui);
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

pub struct BeakerApp {
    current_view: Option<Box<dyn View>>,
    use_native_menu: bool,
    #[cfg(target_os = "macos")]
    menu: Option<muda::Menu>,
    #[cfg(target_os = "macos")]
    menu_rx: Option<std::sync::mpsc::Receiver<muda::MenuEvent>>,
}

impl BeakerApp {
    pub fn new(use_native_menu: bool, image_path: Option<String>) -> Self {
        let current_view: Option<Box<dyn View>> = if let Some(path) = image_path {
            match DetectionView::new(&path) {
                Ok(view) => Some(Box::new(view)),
                Err(e) => {
                    eprintln!("Failed to load image: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            current_view,
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
            if let Some(view) = &mut self.current_view {
                view.show(ctx, ui);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("Beaker - Bird Image Analysis");
                    ui.add_space(20.0);
                    ui.label("Load an image using: beaker-gui --image path/to/image.jpg");
                });
            }
        });
    }
}
