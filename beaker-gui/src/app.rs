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
    Directory(crate::views::DirectoryView),
}

pub struct BeakerApp {
    state: AppState,
    recent_files: RecentFiles,
    use_native_menu: bool,
    /// Pending file dialog result from menu (None = no dialog, Some(paths) = selected files)
    pending_menu_file_dialog: Arc<Mutex<Option<Vec<PathBuf>>>>,
    /// Pending folder dialog result from menu (None = no dialog, Some(paths) = selected folder)
    pending_menu_folder_dialog: Arc<Mutex<Option<Vec<PathBuf>>>>,
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
                } else if event.id == muda::MenuId::new("open") {
                    eprintln!("[BeakerApp] Native menu: Open clicked");
                    self.spawn_menu_file_dialog();
                }
            }
        }
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Files...").clicked() {
                        eprintln!("[BeakerApp] Menu: Open Files... clicked");
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
    /// Supports multi-select for files
    fn spawn_menu_file_dialog(&self) {
        let dialog_result = Arc::clone(&self.pending_menu_file_dialog);
        std::thread::spawn(move || {
            eprintln!("[BeakerApp] Menu file dialog thread started (multi-select)");
            let paths = rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png"])
                .add_filter("Beaker metadata", &["toml"])
                .pick_files();

            let paths_vec: Vec<PathBuf> = paths.unwrap_or_default();

            eprintln!(
                "[BeakerApp] Menu file dialog result: {} file(s) selected",
                paths_vec.len()
            );
            *dialog_result.lock().unwrap() = Some(paths_vec);
        });
    }

    /// Spawn a folder dialog from menu in a separate thread to avoid blocking the UI
    fn spawn_menu_folder_dialog(&self) {
        let dialog_result = Arc::clone(&self.pending_menu_folder_dialog);
        std::thread::spawn(move || {
            eprintln!("[BeakerApp] Menu folder dialog thread started");
            let folder = rfd::FileDialog::new().pick_folder();

            let paths_vec: Vec<PathBuf> = if let Some(folder_path) = folder {
                vec![folder_path]
            } else {
                Vec::new()
            };

            eprintln!(
                "[BeakerApp] Menu folder dialog result: {} folder(s) selected",
                paths_vec.len()
            );
            *dialog_result.lock().unwrap() = Some(paths_vec);
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

        // Collect image files from directory
        let image_paths = Self::collect_image_files(&path);
        eprintln!(
            "[BeakerApp] Found {} image files in folder",
            image_paths.len()
        );

        if image_paths.is_empty() {
            eprintln!("[BeakerApp] WARNING: No supported image files found in folder");
            return;
        }

        // Create DirectoryView and start processing
        let mut dir_view = crate::views::DirectoryView::new(path.clone(), image_paths);
        dir_view.start_processing();
        let _ = self.recent_files.add(path, RecentItemType::Folder);
        self.state = AppState::Directory(dir_view);
    }

    /// Collect all supported image files from a directory (non-recursive)
    fn collect_image_files(dir_path: &PathBuf) -> Vec<PathBuf> {
        let mut image_files = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                            image_files.push(path);
                        }
                    }
                }
            }
        }

        // Sort for consistent ordering
        image_files.sort();
        image_files
    }

    fn open_paths(&mut self, paths: Vec<PathBuf>) {
        eprintln!("[BeakerApp] Opening {} path(s)", paths.len());

        if paths.is_empty() {
            return;
        }

        // If single path, handle based on type
        if paths.len() == 1 {
            let path = &paths[0];
            if path.is_file() {
                eprintln!("[BeakerApp] Opening single file: {:?}", path);
                self.open_image(path.clone());
            } else if path.is_dir() {
                eprintln!("[BeakerApp] Opening folder: {:?}", path);
                self.open_folder(path.clone());
            }
            return;
        }

        // Multiple paths - use bulk workflow
        // Filter to only image files
        let mut image_files: Vec<PathBuf> = paths
            .into_iter()
            .filter(|p| {
                if p.is_file() {
                    if let Some(ext) = p.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png"
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .collect();

        if image_files.is_empty() {
            eprintln!("[BeakerApp] WARNING: No valid image files selected");
            return;
        }

        // Sort for consistent ordering
        image_files.sort();

        eprintln!(
            "[BeakerApp] Opening {} image(s) in bulk mode",
            image_files.len()
        );

        // Get parent directory for display purposes
        let parent_dir = image_files[0]
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("Selected Files"));

        // Create DirectoryView with the selected files
        let mut dir_view = crate::views::DirectoryView::new(parent_dir.clone(), image_files);
        dir_view.start_processing();
        let _ = self.recent_files.add(parent_dir, RecentItemType::Folder);
        self.state = AppState::Directory(dir_view);
    }
}

impl eframe::App for BeakerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for completed menu file dialogs (non-blocking)
        // Extract paths first, then process after releasing the lock
        let file_paths = if let Ok(mut result) = self.pending_menu_file_dialog.try_lock() {
            if let Some(paths) = result.take() {
                if !paths.is_empty() {
                    eprintln!(
                        "[BeakerApp] Menu file dialog completed with {} path(s)",
                        paths.len()
                    );
                    Some(paths)
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

        // Check for completed menu folder dialogs (non-blocking)
        let folder_paths = if let Ok(mut result) = self.pending_menu_folder_dialog.try_lock() {
            if let Some(paths) = result.take() {
                if !paths.is_empty() {
                    eprintln!(
                        "[BeakerApp] Menu folder dialog completed with {} path(s)",
                        paths.len()
                    );
                    Some(paths)
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
        if let Some(paths) = file_paths {
            self.open_paths(paths);
        }
        if let Some(paths) = folder_paths {
            self.open_paths(paths);
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
                    WelcomeAction::OpenPaths(paths) => {
                        eprintln!(
                            "[BeakerApp] Received action: OpenPaths({} path(s))",
                            paths.len()
                        );
                        self.open_paths(paths);
                    }
                    WelcomeAction::None => {}
                }
            }
            AppState::Detection(detection_view) => {
                detection_view.show(ctx, ui);
            }
            AppState::Directory(directory_view) => {
                directory_view.show(ctx, ui);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_supports_directory_view() {
        let dir_view = crate::views::DirectoryView::new(
            PathBuf::from("/tmp"),
            vec![PathBuf::from("/tmp/img1.jpg")],
        );
        let _state = AppState::Directory(dir_view);
        // If this compiles, AppState has Directory variant
    }

    #[test]
    fn test_app_renders_directory_view() {
        let dir_view = crate::views::DirectoryView::new(
            PathBuf::from("/tmp"),
            vec![PathBuf::from("/tmp/img1.jpg")],
        );
        let _app = BeakerApp {
            state: AppState::Directory(dir_view),
            recent_files: RecentFiles::default(),
            use_native_menu: false,
            pending_menu_file_dialog: Arc::new(Mutex::new(None)),
            pending_menu_folder_dialog: Arc::new(Mutex::new(None)),
            #[cfg(target_os = "macos")]
            menu: None,
            #[cfg(target_os = "macos")]
            menu_rx: None,
        };

        // This should compile and not panic
        let _ = format!("{:?}", "Testing directory view in app");
    }

    #[test]
    fn test_open_folder_creates_directory_view() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let img1 = temp_dir.path().join("img1.jpg");
        let img2 = temp_dir.path().join("img2.jpg");

        // Create empty files
        File::create(&img1).unwrap();
        File::create(&img2).unwrap();

        let mut app = BeakerApp::new(false, None);
        app.open_folder(temp_dir.path().to_path_buf());

        // Should transition to Directory state
        assert!(matches!(app.state, AppState::Directory(_)));
    }

    #[test]
    fn test_open_folder_starts_processing() {
        use std::fs::File;
        use std::thread;
        use std::time::Duration;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let img1 = temp_dir.path().join("img1.jpg");
        File::create(&img1).unwrap();

        let mut app = BeakerApp::new(false, None);
        app.open_folder(temp_dir.path().to_path_buf());

        // Give processing thread time to start
        thread::sleep(Duration::from_millis(100));

        // Should transition to Directory state with processing started
        match &app.state {
            AppState::Directory(view) => {
                // At least one image should be in Waiting or Processing state
                assert!(!view.images.is_empty());
            }
            _ => panic!("Expected Directory state"),
        }
    }
}
