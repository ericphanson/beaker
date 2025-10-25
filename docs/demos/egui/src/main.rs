use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 500.0])
            .with_min_inner_size([550.0, 450.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Hello World - egui Demo",
        options,
        Box::new(|cc| {
            setup_custom_style(&cc.egui_ctx);
            Ok(Box::new(HelloWorldApp::default()))
        }),
    )
}

fn setup_custom_style(ctx: &egui::Context) {
    // High DPI rendering
    ctx.set_pixels_per_point(2.0);

    let mut style = (*ctx.style()).clone();

    // Generous spacing for comfortable UI
    style.spacing.item_spacing = egui::vec2(16.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(20.0);

    // Beautiful rounded corners throughout
    style.visuals.window_rounding = egui::Rounding::same(12.0);
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    // Crisp, subtle borders
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(180));
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.5, egui::Color32::from_gray(160));
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 220));
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(80, 130, 200));

    // Refined background colors
    style.visuals.window_fill = egui::Color32::from_gray(248);
    style.visuals.panel_fill = egui::Color32::from_gray(248);

    // Better button colors
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_gray(70);
    style.visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_gray(60);
    style.visuals.widgets.active.weak_bg_fill = egui::Color32::from_gray(50);

    // Softer, more sophisticated shadows
    style.visuals.window_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 4.0),
        blur: 16.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(40),
    };

    ctx.set_style(style);
}

#[derive(Default)]
struct HelloWorldApp {
    name: String,
}

impl eframe::App for HelloWorldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Clean macOS-style menu bar
        egui::TopBottomPanel::top("menu_bar")
            .frame(egui::Frame::none()
                .fill(egui::Color32::WHITE)
                .inner_margin(egui::Margin::symmetric(12.0, 4.0)))
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        let new_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::N);
                        if ui.add(egui::Button::new("New").shortcut_text(ctx.format_shortcut(&new_shortcut))).clicked() {
                            // Action placeholder
                        }
                        if ctx.input_mut(|i| i.consume_shortcut(&new_shortcut)) {
                            // Handle Cmd+N
                        }

                        let open_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
                        if ui.add(egui::Button::new("Open...").shortcut_text(ctx.format_shortcut(&open_shortcut))).clicked() {
                            // Action placeholder
                        }
                        if ctx.input_mut(|i| i.consume_shortcut(&open_shortcut)) {
                            // Handle Cmd+O
                        }

                        ui.separator();

                        let quit_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Q);
                        if ui.add(egui::Button::new("Quit").shortcut_text(ctx.format_shortcut(&quit_shortcut))).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ctx.input_mut(|i| i.consume_shortcut(&quit_shortcut)) {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        let clear_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
                        if ui.add(egui::Button::new("Clear").shortcut_text(ctx.format_shortcut(&clear_shortcut))).clicked() {
                            self.name.clear();
                        }
                        if ctx.input_mut(|i| i.consume_shortcut(&clear_shortcut)) {
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

            // Large, prominent heading
            ui.vertical_centered(|ui| {
                ui.heading(egui::RichText::new("Hello, World!")
                    .size(40.0)
                    .strong()
                    .color(egui::Color32::from_gray(50)));
            });

            ui.add_space(40.0);

            // Beautifully styled input card
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
                            ui.label(egui::RichText::new("Your name:")
                                .size(18.0)
                                .color(egui::Color32::from_gray(80)));
                            ui.add_space(12.0);
                            ui.add(egui::TextEdit::singleline(&mut self.name)
                                .hint_text("Enter your name...")
                                .desired_width(220.0)
                                .font(egui::TextStyle::Body));
                        });
                    });
            });

            ui.add_space(28.0);

            // Greeting with beautiful color
            if !self.name.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new(format!("Hello, {}!", self.name))
                        .size(28.0)
                        .strong()
                        .color(egui::Color32::from_rgb(70, 130, 200)));
                });
            }

            ui.add_space(24.0);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_app_can_be_created() {
        let _app = HelloWorldApp::default();
    }
}
