use crate::recent_files::{RecentFiles, RecentItemType};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Action to be taken based on user interaction
#[derive(Debug, Clone)]
pub enum WelcomeAction {
    None,
    OpenPaths(Vec<PathBuf>),
}

pub struct WelcomeView {
    recent_files: RecentFiles,
    drag_hover: bool,
    /// Pending file dialog result (None = no dialog, Some(paths) = selected files)
    pending_file_dialog: Arc<Mutex<Option<Vec<PathBuf>>>>,
}

impl WelcomeView {
    pub fn new() -> Self {
        Self {
            recent_files: RecentFiles::default(),
            drag_hover: false,
            pending_file_dialog: Arc::new(Mutex::new(None)),
        }
    }

    /// Show the welcome view and return an action if user clicked something
    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> WelcomeAction {
        let mut action = WelcomeAction::None;

        // Check for completed file dialogs (non-blocking)
        if let Ok(mut result) = self.pending_file_dialog.try_lock() {
            if let Some(paths) = result.take() {
                if !paths.is_empty() {
                    eprintln!(
                        "[WelcomeView] File dialog completed with {} path(s)",
                        paths.len()
                    );
                    action = WelcomeAction::OpenPaths(paths);
                } else {
                    eprintln!("[WelcomeView] File dialog cancelled");
                }
            }
        }

        // Check for dropped files (support multiple)
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                let mut paths = Vec::new();
                for dropped_file in &i.raw.dropped_files {
                    if let Some(path) = &dropped_file.path {
                        eprintln!("[WelcomeView] File dropped: {:?}", path);
                        paths.push(path.clone());
                    }
                }
                if !paths.is_empty() {
                    eprintln!("[WelcomeView] Opening {} dropped item(s)", paths.len());
                    action = WelcomeAction::OpenPaths(paths);
                }
            }

            // Check if files are being hovered
            let was_hovering = self.drag_hover;
            self.drag_hover = !i.raw.hovered_files.is_empty();
            if self.drag_hover && !was_hovering {
                eprintln!("[WelcomeView] Files hovering over drop zone");
            }
        });

        ui.vertical_centered(|ui| {
            ui.add_space(60.0);

            // Title
            ui.heading(egui::RichText::new("Beaker - Bird Image Analysis").size(32.0));
            ui.add_space(40.0);

            // Drag & drop zone (also clickable)
            let drop_zone_height = 200.0;
            let available_width = ui.available_width().min(600.0);

            let drop_zone_rect = egui::Rect::from_min_size(
                ui.cursor().min,
                egui::vec2(available_width, drop_zone_height),
            );

            // Make the drop zone clickable
            let drop_zone_response = ui.allocate_rect(drop_zone_rect, egui::Sense::click());

            // Draw drop zone
            let fill_color = if self.drag_hover {
                egui::Color32::from_rgb(230, 240, 255)
            } else if drop_zone_response.hovered() {
                egui::Color32::from_rgb(245, 245, 250)
            } else {
                egui::Color32::from_rgb(250, 250, 250)
            };

            ui.painter().rect(
                drop_zone_rect,
                6.0,
                fill_color,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(200, 200, 200)),
            );

            // Drop zone text
            let text_pos = drop_zone_rect.center();
            ui.painter().text(
                text_pos - egui::vec2(0.0, 30.0),
                egui::Align2::CENTER_CENTER,
                "Drop image or folder here",
                egui::FontId::proportional(20.0),
                egui::Color32::from_rgb(100, 100, 100),
            );

            // Icon hint
            ui.painter().text(
                text_pos + egui::vec2(0.0, 10.0),
                egui::Align2::CENTER_CENTER,
                "📁 or 🖼️",
                egui::FontId::proportional(32.0),
                egui::Color32::from_rgb(100, 100, 100),
            );

            // Click hint
            ui.painter().text(
                text_pos + egui::vec2(0.0, 50.0),
                egui::Align2::CENTER_CENTER,
                "(or click to browse)",
                egui::FontId::proportional(14.0),
                egui::Color32::from_rgb(120, 120, 120),
            );

            // Show highlight on hover or drag
            if drop_zone_response.hovered() || self.drag_hover {
                ui.painter().rect(
                    drop_zone_rect,
                    6.0,
                    egui::Color32::TRANSPARENT,
                    egui::Stroke::new(3.0, egui::Color32::from_rgb(100, 149, 237)),
                );
            }

            // Handle click on drop zone
            if drop_zone_response.clicked() {
                eprintln!("[WelcomeView] Drop zone clicked, opening file dialog...");
                self.spawn_file_dialog();
            }

            ui.add_space(drop_zone_height + 20.0);

            // Open button
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 200.0) / 2.0);

                if ui
                    .add_sized(
                        [200.0, 50.0],
                        egui::Button::new(egui::RichText::new("Open...").size(18.0)),
                    )
                    .clicked()
                {
                    eprintln!("[WelcomeView] 'Open...' button clicked");
                    self.spawn_file_dialog();
                }
            });

            ui.add_space(40.0);

            // Recent files section
            self.show_recent_files(ui, &mut action);

            ui.add_space(30.0);

            // Tip
            ui.label(
                egui::RichText::new(
                    "💡 Tip: Process a folder to triage quality across multiple images,\n\
                     or open a single image to inspect detections",
                )
                .size(14.0)
                .color(egui::Color32::from_rgb(100, 100, 100)),
            );

            ui.add_space(20.0);
        });

        action
    }

    fn show_recent_files(&mut self, ui: &mut egui::Ui, action: &mut WelcomeAction) {
        let items = self.recent_files.items();

        if items.is_empty() {
            return;
        }

        ui.separator();
        ui.add_space(10.0);
        ui.label(egui::RichText::new("Recent Files").size(20.0));
        ui.add_space(10.0);

        for item in items.iter().take(10) {
            let icon = match item.item_type {
                RecentItemType::Image => "🖼️",
                RecentItemType::Folder => "📁",
            };

            // Format timestamp nicely
            let time_ago = Self::format_time_ago(&item.timestamp);

            ui.horizontal(|ui| {
                ui.add_space(50.0);

                let path_str = item.path.to_str().unwrap_or("Unknown path").to_string();

                let button_text = format!("{} {}", icon, path_str);

                if ui
                    .add_sized(
                        [500.0, 30.0],
                        egui::Button::new(egui::RichText::new(&button_text).size(14.0)),
                    )
                    .clicked()
                {
                    eprintln!("[WelcomeView] Recent item clicked: {:?}", item.path);
                    *action = WelcomeAction::OpenPaths(vec![item.path.clone()]);
                }

                ui.label(
                    egui::RichText::new(format!("({})", time_ago))
                        .size(12.0)
                        .color(egui::Color32::from_rgb(150, 150, 150)),
                );
            });

            ui.add_space(5.0);
        }
    }

    fn format_time_ago(timestamp: &str) -> String {
        use chrono::{DateTime, Utc};

        let parsed = DateTime::parse_from_rfc3339(timestamp);
        if let Ok(dt) = parsed {
            let now = Utc::now();
            let dt_utc = dt.with_timezone(&Utc);
            let duration = now.signed_duration_since(dt_utc);

            if duration.num_seconds() < 60 {
                "just now".to_string()
            } else if duration.num_minutes() < 60 {
                let mins = duration.num_minutes();
                format!("{} minute{} ago", mins, if mins == 1 { "" } else { "s" })
            } else if duration.num_hours() < 24 {
                let hours = duration.num_hours();
                format!("{} hour{} ago", hours, if hours == 1 { "" } else { "s" })
            } else if duration.num_days() < 7 {
                let days = duration.num_days();
                format!("{} day{} ago", days, if days == 1 { "" } else { "s" })
            } else if duration.num_weeks() < 4 {
                let weeks = duration.num_weeks();
                format!("{} week{} ago", weeks, if weeks == 1 { "" } else { "s" })
            } else {
                "a while ago".to_string()
            }
        } else {
            "unknown".to_string()
        }
    }

    /// Spawn a file dialog in a separate thread to avoid blocking the UI
    /// Supports multi-select for both files and folders
    fn spawn_file_dialog(&self) {
        let dialog_result = Arc::clone(&self.pending_file_dialog);
        std::thread::spawn(move || {
            eprintln!("[WelcomeView] File dialog thread started (multi-select)");
            let paths = rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png"])
                .add_filter("Beaker metadata", &["toml"])
                .pick_files();

            let paths_vec: Vec<PathBuf> = paths.unwrap_or_default();

            eprintln!(
                "[WelcomeView] File dialog result: {} file(s) selected",
                paths_vec.len()
            );
            *dialog_result.lock().unwrap() = Some(paths_vec);
        });
    }

    /// Add a file to recent files list
    #[allow(dead_code)]
    pub fn add_recent_file(&mut self, path: PathBuf, item_type: RecentItemType) {
        let _ = self.recent_files.add(path, item_type);
    }
}

impl Default for WelcomeView {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welcome_view_creation() {
        let view = WelcomeView::new();
        assert!(!view.drag_hover);
    }

    #[test]
    fn test_format_time_ago() {
        use chrono::Utc;

        // Test "just now"
        let now = Utc::now().to_rfc3339();
        assert_eq!(WelcomeView::format_time_ago(&now), "just now");

        // Test "minutes ago"
        let mins_ago = (Utc::now() - chrono::Duration::minutes(5)).to_rfc3339();
        assert_eq!(WelcomeView::format_time_ago(&mins_ago), "5 minutes ago");

        // Test "hours ago"
        let hours_ago = (Utc::now() - chrono::Duration::hours(3)).to_rfc3339();
        assert_eq!(WelcomeView::format_time_ago(&hours_ago), "3 hours ago");

        // Test "days ago"
        let days_ago = (Utc::now() - chrono::Duration::days(2)).to_rfc3339();
        assert_eq!(WelcomeView::format_time_ago(&days_ago), "2 days ago");
    }
}
