use beaker_gui::{style, BeakerApp};
use egui_kittest::Harness;

#[test]
fn test_app_creation_no_image() {
    // Test that app can be created without an image
    let app = BeakerApp::new(false, None);

    // App should be created successfully
    // This validates basic app structure without requiring model downloads
    drop(app);
}

#[test]
fn test_style_setup() {
    // Test that style setup doesn't panic and applies correctly
    let mut harness = Harness::new_ui(|ui| {
        let ctx = ui.ctx();
        style::setup_custom_style(ctx);

        // Verify style was applied by checking spacing
        let style = ctx.style();
        assert!(style.spacing.item_spacing.x > 0.0);

        ui.label("Test");
    });
    harness.run();
}

#[test]
fn test_constants() {
    // Test that style constants are reasonable
    assert!(crate::style::MIN_WINDOW_WIDTH > 0.0);
    assert!(crate::style::MIN_WINDOW_HEIGHT > 0.0);
    assert!(crate::style::DETECTION_PANEL_WIDTH > 0.0);
}

// Note: Tests that require actual detection are skipped in CI
// as they would require model downloads. These tests validate:
// - App structure creation
// - Style application
// - Constants validity
//
// Actual detection with bounding boxes can be tested manually:
// cargo run --release -- --image example.jpg
