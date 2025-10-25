use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Hello World",
        options,
        Box::new(|_cc| Ok(Box::new(HelloWorldApp::default()))),
    )
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_app_can_be_created() {
        let _app = HelloWorldApp::default();
    }
}
