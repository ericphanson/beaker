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
            ui.heading("Hello, World!");
            ui.horizontal(|ui| {
                ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name);
            });
            if !self.name.is_empty() {
                ui.label(format!("Hello, {}!", self.name));
            }
        });
    }
}

#[test]
fn test_hello_world_initial_state() {
    let mut harness = Harness::new_ui(|ui| {
        ui.heading("Hello, World!");
        ui.horizontal(|ui| {
            ui.label("Your name: ");
            ui.text_edit_singleline(&mut String::new());
        });
    });

    harness.run();

    // Create snapshot of initial state
    harness.wgpu_snapshot("hello_world_initial");
}

#[test]
fn test_simple_heading() {
    let mut harness = Harness::new_ui(|ui| {
        ui.heading("Hello, World!");
    });

    harness.run();

    // Verify the heading exists
    harness.get_by_label("Hello, World!");

    // Create snapshot
    harness.wgpu_snapshot("simple_heading");
}

#[test]
fn test_text_input_widget() {
    let mut harness = Harness::new_ui(|ui| {
        ui.label("Your name: ");
        ui.text_edit_singleline(&mut String::new());
    });

    harness.run();

    // Verify the label exists
    harness.get_by_label("Your name: ");

    // Create snapshot
    harness.wgpu_snapshot("text_input_widget");
}

#[test]
fn test_app_creation() {
    let app = HelloWorldApp::default();
    assert_eq!(app.name, "");
}

#[test]
fn test_button_widget() {
    let mut harness = Harness::new_ui(|ui| {
        let _ = ui.button("Click me");
    });

    harness.run();

    // Find the button
    harness.get_by_label("Click me");

    // Create snapshot before click
    harness.wgpu_snapshot("button_before_click");

    // Click the button
    harness.get_by_label("Click me").click();

    // Create snapshot after click (button state changes)
    harness.wgpu_snapshot("button_after_click");
}

#[test]
fn test_multiple_widgets() {
    let mut harness = Harness::new_ui(|ui| {
        ui.heading("Demo Application");
        ui.separator();
        ui.label("This is a label");
        ui.horizontal(|ui| {
            let _ = ui.button("Button 1");
            let _ = ui.button("Button 2");
        });
        ui.checkbox(&mut false, "Checkbox");
    });

    harness.run();

    // Verify all widgets exist
    harness.get_by_label("Demo Application");
    harness.get_by_label("This is a label");
    harness.get_by_label("Button 1");
    harness.get_by_label("Button 2");

    // Create snapshot
    harness.wgpu_snapshot("multiple_widgets");
}

#[test]
fn test_colored_text() {
    let mut harness = Harness::new_ui(|ui| {
        ui.heading("Styled Text Demo");
        ui.colored_label(egui::Color32::RED, "Red text");
        ui.colored_label(egui::Color32::GREEN, "Green text");
        ui.colored_label(egui::Color32::BLUE, "Blue text");
    });

    harness.run();

    // Create snapshot showing colored text
    harness.wgpu_snapshot("colored_text");
}
