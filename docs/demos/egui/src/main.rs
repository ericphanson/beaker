use eframe::egui;

#[cfg(target_os = "macos")]
use muda::{Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};

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

            #[cfg(target_os = "macos")]
            {
                let mut app = Box::new(HelloWorldApp::default());
                let (menu, rx) = create_native_menu();
                menu.init_for_nsapp();
                app.menu = Some(menu);
                app.menu_rx = Some(rx);
                Ok(app)
            }

            #[cfg(not(target_os = "macos"))]
            {
                Ok(Box::new(HelloWorldApp::default()))
            }
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

struct HelloWorldApp {
    name: String,
    #[cfg(target_os = "macos")]
    menu: Option<Menu>,
    #[cfg(target_os = "macos")]
    menu_rx: Option<std::sync::mpsc::Receiver<MenuEvent>>,
}

impl Default for HelloWorldApp {
    fn default() -> Self {
        Self {
            name: String::new(),
            #[cfg(target_os = "macos")]
            menu: None,
            #[cfg(target_os = "macos")]
            menu_rx: None,
        }
    }
}

impl eframe::App for HelloWorldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll menu events on macOS
        #[cfg(target_os = "macos")]
        if let Some(rx) = self.menu_rx.take() {
            while let Ok(event) = rx.try_recv() {
                self.handle_menu_event(event, ctx);
            }
            self.menu_rx = Some(rx);
        }

        // On non-macOS platforms, show egui-rendered menu bar
        #[cfg(not(target_os = "macos"))]
        {
            egui::TopBottomPanel::top("menu_bar")
                .frame(egui::Frame::none()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(12.0, 4.0)))
                .show(ctx, |ui| {
                    egui::menu::bar(ui, |ui| {
                        ui.menu_button("File", |ui| {
                            let new_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::N);
                            if ui.add(egui::Button::new("New").shortcut_text(ctx.format_shortcut(&new_shortcut))).clicked() {}

                            let open_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
                            if ui.add(egui::Button::new("Open...").shortcut_text(ctx.format_shortcut(&open_shortcut))).clicked() {}

                            ui.separator();

                            let quit_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Q);
                            if ui.add(egui::Button::new("Quit").shortcut_text(ctx.format_shortcut(&quit_shortcut))).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });

                        ui.menu_button("Edit", |ui| {
                            let clear_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
                            if ui.add(egui::Button::new("Clear").shortcut_text(ctx.format_shortcut(&clear_shortcut))).clicked() {
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
        }

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

#[cfg(target_os = "macos")]
impl HelloWorldApp {
    fn handle_menu_event(&mut self, event: MenuEvent, ctx: &egui::Context) {
        let id = event.id;
        println!("Menu event: {id:?}");

        // Handle menu events based on ID
        // Since we're using PredefinedMenuItems, they handle their actions automatically
        // We can add custom handling here if needed

        // For custom items, we'd check the ID and handle accordingly
        // For now, the predefined items (Quit, etc.) work automatically
    }
}

#[cfg(target_os = "macos")]
fn create_native_menu() -> (Menu, std::sync::mpsc::Receiver<MenuEvent>) {
    let menu = Menu::new();

    // Set up menu event channel
    let (tx, rx) = std::sync::mpsc::channel();
    MenuEvent::set_event_handler(Some(move |event| {
        let _ = tx.send(event);
    }));

    // App menu (first menu with app name)
    let app_menu = Submenu::new("Hello World", true);
    app_menu.append(&PredefinedMenuItem::about(None, None)).unwrap();
    app_menu.append(&PredefinedMenuItem::separator()).unwrap();
    app_menu.append(&PredefinedMenuItem::services(None)).unwrap();
    app_menu.append(&PredefinedMenuItem::separator()).unwrap();
    app_menu.append(&PredefinedMenuItem::hide(None)).unwrap();
    app_menu.append(&PredefinedMenuItem::hide_others(None)).unwrap();
    app_menu.append(&PredefinedMenuItem::show_all(None)).unwrap();
    app_menu.append(&PredefinedMenuItem::separator()).unwrap();
    app_menu.append(&PredefinedMenuItem::quit(None)).unwrap();
    menu.append(&app_menu).unwrap();

    // File menu
    let file_menu = Submenu::new("File", true);
    let new_item = MenuItem::new("New", true, Some(muda::accelerator::Accelerator::new(
        Some(muda::accelerator::Modifiers::SUPER),
        muda::accelerator::Code::KeyN,
    )));
    let open_item = MenuItem::new("Open...", true, Some(muda::accelerator::Accelerator::new(
        Some(muda::accelerator::Modifiers::SUPER),
        muda::accelerator::Code::KeyO,
    )));
    file_menu.append(&new_item).unwrap();
    file_menu.append(&open_item).unwrap();
    file_menu.append(&PredefinedMenuItem::separator()).unwrap();
    file_menu.append(&PredefinedMenuItem::close_window(None)).unwrap();
    menu.append(&file_menu).unwrap();

    // Edit menu
    let edit_menu = Submenu::new("Edit", true);
    edit_menu.append(&PredefinedMenuItem::undo(None)).unwrap();
    edit_menu.append(&PredefinedMenuItem::redo(None)).unwrap();
    edit_menu.append(&PredefinedMenuItem::separator()).unwrap();
    edit_menu.append(&PredefinedMenuItem::cut(None)).unwrap();
    edit_menu.append(&PredefinedMenuItem::copy(None)).unwrap();
    edit_menu.append(&PredefinedMenuItem::paste(None)).unwrap();
    edit_menu.append(&PredefinedMenuItem::select_all(None)).unwrap();
    menu.append(&edit_menu).unwrap();

    // Window menu
    let window_menu = Submenu::new("Window", true);
    window_menu.append(&PredefinedMenuItem::minimize(None)).unwrap();
    window_menu.append(&PredefinedMenuItem::maximize(None)).unwrap();
    window_menu.append(&PredefinedMenuItem::separator()).unwrap();
    window_menu.append(&PredefinedMenuItem::fullscreen(None)).unwrap();
    menu.append(&window_menu).unwrap();

    // Help menu
    let help_menu = Submenu::new("Help", true);
    let about_item = MenuItem::new("About Hello World", true, None);
    help_menu.append(&about_item).unwrap();
    menu.append(&help_menu).unwrap();

    (menu, rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_app_can_be_created() {
        let _app = HelloWorldApp::default();
    }
}
