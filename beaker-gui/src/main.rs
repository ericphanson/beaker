mod app;
mod style;
mod views;

use app::BeakerApp;
use clap::Parser;

#[derive(Parser)]
#[command(name = "beaker-gui")]
#[command(about = "Beaker GUI - Bird Image Analysis Tool", long_about = None)]
struct Args {
    /// Path to image file
    #[arg(long)]
    image: Option<String>,

    /// View to open (detection, etc.)
    #[arg(long, default_value = "detection")]
    view: String,
}

#[cfg(target_os = "macos")]
fn create_native_menu() -> (muda::Menu, std::sync::mpsc::Receiver<muda::MenuEvent>) {
    let menu = muda::Menu::new();

    // File menu
    let file_menu = muda::Submenu::new("File", true);
    let quit_item = muda::MenuItem::new(
        "Quit",
        true,
        Some(muda::Accelerator::new(
            Some(muda::Modifiers::SUPER),
            muda::Code::KeyQ,
        )),
    );
    file_menu.append(&quit_item).unwrap();
    menu.append(&file_menu).unwrap();

    // View menu
    let view_menu = muda::Submenu::new("View", true);
    let detection_item = muda::MenuItem::new("Detection", true, None);
    view_menu.append(&detection_item).unwrap();
    menu.append(&view_menu).unwrap();

    // Help menu
    let help_menu = muda::Submenu::new("Help", true);
    let about_item = muda::MenuItem::new("About", true, None);
    help_menu.append(&about_item).unwrap();
    menu.append(&help_menu).unwrap();

    let (tx, rx) = std::sync::mpsc::channel();
    muda::MenuEvent::set_event_handler(Some(move |event: muda::MenuEvent| {
        let _ = tx.send(event);
    }));

    (menu, rx)
}

fn main() -> eframe::Result {
    let args = Args::parse();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([
                crate::style::MIN_WINDOW_WIDTH,
                crate::style::MIN_WINDOW_HEIGHT,
            ]),
        ..Default::default()
    };

    eframe::run_native(
        "Beaker - Bird Image Analysis",
        options,
        Box::new(move |cc| {
            style::setup_custom_style(&cc.egui_ctx);

            let use_native_menu =
                cfg!(target_os = "macos") && std::env::var("USE_EGUI_MENU").is_err();

            #[cfg(target_os = "macos")]
            {
                let mut app = BeakerApp::new(use_native_menu, args.image);
                if use_native_menu {
                    let (menu, rx) = create_native_menu();
                    menu.init_for_nsapp();
                    app.set_menu(menu, rx);
                }
                Ok(Box::new(app))
            }

            #[cfg(not(target_os = "macos"))]
            {
                let app = BeakerApp::new(use_native_menu, args.image);
                Ok(Box::new(app))
            }
        }),
    )
}
