use eframe::egui;
use egui_kittest::Harness;

#[derive(Default)]
struct HelloWorldApp {
    name: String,
}

impl eframe::App for HelloWorldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Clean macOS-style menu bar
        egui::TopBottomPanel::top("menu_bar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(12.0, 4.0)),
            )
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        let new_shortcut =
                            egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::N);
                        if ui
                            .add(
                                egui::Button::new("New")
                                    .shortcut_text(ctx.format_shortcut(&new_shortcut)),
                            )
                            .clicked()
                        {}

                        let open_shortcut =
                            egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
                        if ui
                            .add(
                                egui::Button::new("Open...")
                                    .shortcut_text(ctx.format_shortcut(&open_shortcut)),
                            )
                            .clicked()
                        {}

                        ui.separator();

                        let quit_shortcut =
                            egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Q);
                        if ui
                            .add(
                                egui::Button::new("Quit")
                                    .shortcut_text(ctx.format_shortcut(&quit_shortcut)),
                            )
                            .clicked()
                        {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        let clear_shortcut =
                            egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
                        if ui
                            .add(
                                egui::Button::new("Clear")
                                    .shortcut_text(ctx.format_shortcut(&clear_shortcut)),
                            )
                            .clicked()
                        {
                            self.name.clear();
                        }
                    });

                    ui.menu_button("Help", |ui| {
                        ui.label("egui Demo Application");
                        ui.separator();
                        ui.label("Built with egui 0.30");
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(32.0);
            ui.vertical_centered(|ui| {
                ui.heading(
                    egui::RichText::new("Hello, World!")
                        .size(40.0)
                        .strong()
                        .color(egui::Color32::from_gray(50)),
                );
            });
            ui.add_space(40.0);

            ui.horizontal(|ui| {
                let card_width = 400.0;
                if ui.available_width() > card_width {
                    ui.add_space((ui.available_width() - card_width) / 2.0);
                }

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
                        ui.set_max_width(card_width);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("Your name:")
                                    .size(18.0)
                                    .color(egui::Color32::from_gray(80)),
                            );
                            ui.add_space(12.0);
                            ui.add(
                                egui::TextEdit::singleline(&mut self.name)
                                    .hint_text("Enter your name...")
                                    .desired_width(220.0),
                            );
                        });
                    });
            });

            ui.add_space(28.0);

            if !self.name.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Hello, {}!", self.name))
                            .size(28.0)
                            .strong()
                            .color(egui::Color32::from_rgb(70, 130, 200)),
                    );
                });
            }
        });
    }
}

fn apply_custom_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(16.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(20.0);

    style.visuals.window_rounding = egui::Rounding::same(12.0);
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    style.visuals.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_gray(180));
    style.visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.5, egui::Color32::from_gray(160));
    style.visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 220));
    style.visuals.widgets.active.bg_stroke =
        egui::Stroke::new(2.5, egui::Color32::from_rgb(80, 130, 200));

    style.visuals.window_fill = egui::Color32::from_gray(248);
    style.visuals.panel_fill = egui::Color32::from_gray(248);

    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_gray(70);
    style.visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_gray(60);
    style.visuals.widgets.active.weak_bg_fill = egui::Color32::from_gray(50);

    ctx.set_style(style);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn test_menu_bar() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        // Clean macOS-style menu bar
        egui::Frame::none()
            .fill(egui::Color32::WHITE)
            .inner_margin(egui::Margin::symmetric(12.0, 4.0))
            .show(ui, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |_ui| {});
                    ui.menu_button("Edit", |_ui| {});
                    ui.menu_button("Help", |_ui| {});
                });
            });
    });

    harness.run();
    harness.wgpu_snapshot("menu_bar");
}

#[test]
#[cfg(not(target_os = "macos"))]
fn test_styled_card() {
    let mut harness = Harness::new_ui(|ui| {
        apply_custom_style(ui.ctx());

        ui.add_space(16.0);

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
                ui.heading(egui::RichText::new("Card Component").size(20.0));
                ui.add_space(8.0);
                ui.label("Beautifully styled with shadows and rounded corners.");
            });
    });

    harness.run();
    harness.wgpu_snapshot("styled_card");
}

#[test]
fn test_app_creation() {
    let app = HelloWorldApp::default();
    assert_eq!(app.name, "");
}
