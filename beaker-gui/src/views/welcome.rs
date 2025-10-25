use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Recent file entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentFile {
    pub path: PathBuf,
    pub timestamp: String,
    pub is_directory: bool,
}

/// Welcome view shown when no image/folder is loaded
pub struct WelcomeView {
    recent_files: Vec<RecentFile>,
    recent_files_path: PathBuf,
    show_open_error: Option<String>,
}

impl WelcomeView {
    pub fn new() -> Result<Self> {
        // Get config directory
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("beaker-gui");

        // Create config directory if it doesn't exist
        std::fs::create_dir_all(&config_dir)?;

        let recent_files_path = config_dir.join("recent.json");

        // Load recent files
        let recent_files = Self::load_recent_files(&recent_files_path);

        Ok(Self {
            recent_files,
            recent_files_path,
            show_open_error: None,
        })
    }

    fn load_recent_files(path: &PathBuf) -> Vec<RecentFile> {
        if let Ok(data) = std::fs::read_to_string(path) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    fn save_recent_files(&self) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.recent_files)?;
        std::fs::write(&self.recent_files_path, data)?;
        Ok(())
    }

    pub fn add_recent_file(&mut self, path: PathBuf, is_directory: bool) -> Result<()> {
        // Remove if already exists
        self.recent_files.retain(|f| f.path != path);

        // Add at front
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.recent_files.insert(
            0,
            RecentFile {
                path,
                timestamp,
                is_directory,
            },
        );

        // Keep only last 10
        self.recent_files.truncate(10);

        self.save_recent_files()
    }

    pub fn clear_recent_files(&mut self) -> Result<()> {
        self.recent_files.clear();
        self.save_recent_files()
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> Option<WelcomeAction> {
        let mut action = None;

        // Handle dropped files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(dropped) = i.raw.dropped_files.first() {
                    if let Some(path) = &dropped.path {
                        action = Some(WelcomeAction::OpenPath(path.clone()));
                    }
                }
            }
        });

        ui.vertical_centered(|ui| {
            ui.add_space(60.0);

            ui.heading(egui::RichText::new("Beaker").size(36.0));
            ui.label(egui::RichText::new("Bird Image Analysis").size(18.0).color(egui::Color32::GRAY));

            ui.add_space(40.0);

            // Drop zone
            let drop_zone_size = egui::vec2(400.0, 150.0);
            let (rect, response) = ui.allocate_exact_size(
                drop_zone_size,
                egui::Sense::click(),
            );

            let is_being_dragged = ctx.input(|i| !i.raw.dropped_files.is_empty());
            let bg_color = if is_being_dragged {
                egui::Color32::from_rgb(230, 240, 255)
            } else if response.hovered() {
                egui::Color32::from_rgb(245, 245, 245)
            } else {
                egui::Color32::from_rgb(250, 250, 250)
            };

            ui.painter().rect(
                rect,
                8.0,
                bg_color,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(200, 200, 200)),
            );

            let mut child_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(rect)
                    .layout(egui::Layout::centered_and_justified(egui::Direction::TopDown)),
            );
            child_ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("📁 🖼️").size(32.0));
                ui.label("Drop image or folder here");
            });

            ui.add_space(30.0);

            // Action buttons
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 300.0) / 2.0);

                if ui.button(egui::RichText::new("Open Image").size(16.0))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["jpg", "jpeg", "png"])
                        .pick_file()
                    {
                        action = Some(WelcomeAction::OpenImage(path));
                    }
                }

                ui.add_space(20.0);

                if ui.button(egui::RichText::new("Open Folder").size(16.0))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        action = Some(WelcomeAction::OpenFolder(path));
                    }
                }
            });

            ui.add_space(40.0);

            // Recent files
            if !self.recent_files.is_empty() {
                ui.separator();
                ui.add_space(20.0);

                ui.heading("Recent");

                ui.add_space(10.0);

                for recent in &self.recent_files {
                    ui.horizontal(|ui| {
                        let icon = if recent.is_directory { "📁" } else { "🖼️" };
                        let display_path = recent.path.display().to_string();

                        if ui.button(format!("{} {}", icon, display_path)).clicked() {
                            action = Some(WelcomeAction::OpenPath(recent.path.clone()));
                        }
                    });
                }

                ui.add_space(10.0);

                if ui.button("Clear recent files").clicked() {
                    let _ = self.clear_recent_files();
                }
            }

            // Show error if any
            if let Some(ref error) = self.show_open_error {
                ui.add_space(20.0);
                ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
            }

            ui.add_space(40.0);

            // Tips
            ui.label(egui::RichText::new("💡 Tip:").color(egui::Color32::GRAY));
            ui.label(egui::RichText::new(
                "Process a folder to analyze quality across multiple images"
            ).color(egui::Color32::GRAY).size(12.0));
        });

        action
    }

    pub fn set_error(&mut self, error: String) {
        self.show_open_error = Some(error);
    }

    pub fn clear_error(&mut self) {
        self.show_open_error = None;
    }
}

impl Default for WelcomeView {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            recent_files: Vec::new(),
            recent_files_path: PathBuf::from(".beaker-gui-recent.json"),
            show_open_error: None,
        })
    }
}

/// Actions that can be triggered from the welcome view
#[derive(Debug, Clone)]
pub enum WelcomeAction {
    OpenImage(PathBuf),
    OpenFolder(PathBuf),
    OpenPath(PathBuf), // Generic - will determine if it's a file or folder
}
