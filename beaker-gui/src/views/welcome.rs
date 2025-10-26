use crate::recent_files::{RecentFiles, RecentItemType};
use std::path::PathBuf;

/// Action to be taken based on user interaction
#[derive(Debug, Clone)]
pub enum WelcomeAction {
    None,
    OpenImage(PathBuf),
    OpenFolder(PathBuf),
}

pub struct WelcomeView {
    recent_files: RecentFiles,
    drag_hover: bool,
}

impl WelcomeView {
    pub fn new() -> Self {
        Self {
            recent_files: RecentFiles::default(),
            drag_hover: false,
        }
    }

    /// Show the welcome view and return an action if user clicked something
    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> WelcomeAction {
        let mut action = WelcomeAction::None;

        // Check for dropped files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(dropped_file) = i.raw.dropped_files.first() {
                    if let Some(path) = &dropped_file.path {
                        if path.is_file() {
                            action = WelcomeAction::OpenImage(path.clone());
                        } else if path.is_dir() {
                            action = WelcomeAction::OpenFolder(path.clone());
                        }
                    }
                }
            }

            // Check if files are being hovered
            self.drag_hover = !i.raw.hovered_files.is_empty();
        });

        ui.vertical_centered(|ui| {
            ui.add_space(60.0);

            // Title
            ui.heading(egui::RichText::new("Beaker - Bird Image Analysis").size(32.0));
            ui.add_space(40.0);

            // Drag & drop zone
            let drop_zone_height = 200.0;
            let available_width = ui.available_width().min(600.0);

            let drop_zone_rect = egui::Rect::from_min_size(
                ui.cursor().min,
                egui::vec2(available_width, drop_zone_height),
            );

            let drop_zone_response = ui.allocate_rect(drop_zone_rect, egui::Sense::hover());

            // Draw drop zone
            let fill_color = if self.drag_hover {
                egui::Color32::from_rgb(230, 240, 255)
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
                text_pos - egui::vec2(0.0, 20.0),
                egui::Align2::CENTER_CENTER,
                "Drop image or folder here",
                egui::FontId::proportional(20.0),
                egui::Color32::from_rgb(100, 100, 100),
            );

            // Icon hint
            ui.painter().text(
                text_pos + egui::vec2(0.0, 20.0),
                egui::Align2::CENTER_CENTER,
                "📁 or 🖼️",
                egui::FontId::proportional(32.0),
                egui::Color32::from_rgb(100, 100, 100),
            );

            // Show hint on hover
            if drop_zone_response.hovered() {
                ui.painter().rect(
                    drop_zone_rect,
                    6.0,
                    egui::Color32::TRANSPARENT,
                    egui::Stroke::new(3.0, egui::Color32::from_rgb(100, 149, 237)),
                );
            }

            ui.add_space(drop_zone_height + 20.0);

            // Buttons
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 400.0) / 2.0);

                if ui
                    .add_sized(
                        [190.0, 50.0],
                        egui::Button::new(egui::RichText::new("Open Image").size(18.0)),
                    )
                    .clicked()
                {
                    if let Some(path) = Self::open_file_dialog() {
                        action = WelcomeAction::OpenImage(path);
                    }
                }

                ui.add_space(20.0);

                if ui
                    .add_sized(
                        [190.0, 50.0],
                        egui::Button::new(egui::RichText::new("Open Folder").size(18.0)),
                    )
                    .clicked()
                {
                    if let Some(path) = Self::open_folder_dialog() {
                        action = WelcomeAction::OpenFolder(path);
                    }
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
                    match item.item_type {
                        RecentItemType::Image => {
                            *action = WelcomeAction::OpenImage(item.path.clone());
                        }
                        RecentItemType::Folder => {
                            *action = WelcomeAction::OpenFolder(item.path.clone());
                        }
                    }
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

    fn open_file_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter("Images", &["jpg", "jpeg", "png"])
            .add_filter("Beaker metadata", &["toml"])
            .pick_file()
    }

    fn open_folder_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new().pick_folder()
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
