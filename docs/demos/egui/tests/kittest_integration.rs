use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use eframe::egui;

// Re-create the app structure for testing
#[derive(Default)]
struct HelloWorldApp {
    name: String,
}

impl eframe::App for HelloWorldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.heading(egui::RichText::new("Hello, World!")
                    .size(32.0)
                    .strong());
            });
            ui.add_space(24.0);

            egui::Frame::none()
                .inner_margin(egui::Margin::symmetric(20.0, 12.0))
                .rounding(6.0)
                .fill(egui::Color32::WHITE)
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(200)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Your name:")
                            .size(16.0));
                        ui.add_space(8.0);
                        ui.add(egui::TextEdit::singleline(&mut self.name)
                            .hint_text("Enter your name...")
                            .desired_width(200.0));
                    });
                });

            ui.add_space(16.0);

            if !self.name.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new(format!("Hello, {}!", self.name))
                        .size(24.0)
                        .color(egui::Color32::from_rgb(60, 120, 180)));
                });
            }

            ui.add_space(20.0);
        });
    }
}

fn apply_custom_style(ctx: &egui::Context) {
    // Note: Not setting pixels_per_point in tests to avoid harness issues

    let mut style = (*ctx.style()).clone();

    // Generous spacing
    style.spacing.item_spacing = egui::vec2(16.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(20.0);

    // Beautiful rounded corners
    style.visuals.window_rounding = egui::Rounding::same(12.0);
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    // Crisp borders
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(180));
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.5, egui::Color32::from_gray(160));
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 220));
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(80, 130, 200));

    // Refined backgrounds
    style.visuals.window_fill = egui::Color32::from_gray(248);
    style.visuals.panel_fill = egui::Color32::from_gray(248);

    // Better button colors
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_gray(70);
    style.visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_gray(60);
    style.visuals.widgets.active.weak_bg_fill = egui::Color32::from_gray(50);

    ctx.set_style(style);
}

#[test]
fn test_styled_hello_world() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        ui.add_space(32.0);
        ui.vertical_centered(|ui| {
            ui.heading(egui::RichText::new("Hello, World!")
                .size(40.0)
                .strong()
                .color(egui::Color32::from_gray(50)));
        });
        ui.add_space(40.0);

        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(24.0, 16.0))
            .rounding(10.0)
            .fill(egui::Color32::WHITE)
            .stroke(egui::Stroke::new(1.5, egui::Color32::from_gray(220)))
            .shadow(egui::epaint::Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur: 8.0,
                spread: 0.0,
                color: egui::Color32::from_black_alpha(15),
            })
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Your name:")
                        .size(18.0)
                        .color(egui::Color32::from_gray(80)));
                    ui.add_space(12.0);
                    ui.add(egui::TextEdit::singleline(&mut String::new())
                        .hint_text("Enter your name...")
                        .desired_width(220.0));
                });
            });
    });

    harness.run();
    harness.wgpu_snapshot("styled_hello_world");
}

#[test]
fn test_styled_button() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        ui.add_space(20.0);
        ui.vertical_centered(|ui| {
            let _ = ui.button(egui::RichText::new("Click Me").size(18.0));
        });
    });

    harness.run();
    harness.wgpu_snapshot("styled_button");
}

#[test]
fn test_styled_card_layout() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        ui.add_space(16.0);

        // Card-style frame
        egui::Frame::none()
            .inner_margin(egui::Margin::same(20.0))
            .rounding(8.0)
            .fill(egui::Color32::WHITE)
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(200)))
            .shadow(egui::epaint::Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur: 4.0,
                spread: 0.0,
                color: egui::Color32::from_black_alpha(20),
            })
            .show(ui, |ui| {
                ui.heading(egui::RichText::new("Card Title").size(20.0));
                ui.add_space(8.0);
                ui.label("This is a nicely styled card with shadows and rounded corners.");
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    let _ = ui.button("Action 1");
                    let _ = ui.button("Action 2");
                });
            });
    });

    harness.run();
    harness.wgpu_snapshot("styled_card");
}

#[test]
fn test_styled_colors() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        ui.add_space(16.0);
        ui.vertical_centered(|ui| {
            ui.heading(egui::RichText::new("Color Showcase").size(24.0));
            ui.add_space(12.0);

            ui.label(egui::RichText::new("Primary Blue")
                .size(18.0)
                .color(egui::Color32::from_rgb(60, 120, 180)));
            ui.label(egui::RichText::new("Success Green")
                .size(18.0)
                .color(egui::Color32::from_rgb(80, 180, 100)));
            ui.label(egui::RichText::new("Warning Orange")
                .size(18.0)
                .color(egui::Color32::from_rgb(255, 160, 50)));
            ui.label(egui::RichText::new("Error Red")
                .size(18.0)
                .color(egui::Color32::from_rgb(220, 80, 80)));
        });
    });

    harness.run();
    harness.wgpu_snapshot("styled_colors");
}

#[test]
fn test_app_creation() {
    let app = HelloWorldApp::default();
    assert_eq!(app.name, "");
}
