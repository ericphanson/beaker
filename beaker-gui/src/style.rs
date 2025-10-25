// Style and theme configuration for Beaker GUI

/// Set up custom styling for high-DPI displays with modern aesthetics
pub fn setup_custom_style(ctx: &egui::Context) {
    // High DPI for retina displays
    ctx.set_pixels_per_point(2.0);

    let mut style = (*ctx.style()).clone();

    // Professional spacing
    style.spacing.item_spacing = egui::vec2(16.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(20.0);

    // Rounded corners
    let rounding = egui::Rounding::same(8.0);
    style.visuals.window_rounding = rounding;
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    // Clean color scheme (light mode, scientific tool aesthetic)
    style.visuals.window_fill = egui::Color32::from_gray(248);
    style.visuals.panel_fill = egui::Color32::from_gray(250);
    style.visuals.extreme_bg_color = egui::Color32::WHITE;

    // Accent colors (blue for science/analysis)
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(70, 130, 200);
    style.visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 220));

    // Subtle shadows
    style.visuals.window_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 4.0),
        blur: 12.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(30),
    };

    ctx.set_style(style);
}

// Design constants
pub const MIN_WINDOW_WIDTH: f32 = 900.0;
pub const MIN_WINDOW_HEIGHT: f32 = 600.0;
pub const DETECTION_PANEL_WIDTH: f32 = 250.0;
