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

#[test]
#[ignore = "Requires model download - run with: cargo test -- --ignored"]
fn test_detection_with_real_image() {
    // Test that detection works with a real image and produces at least 1 detection
    // This test exercises the full detection pipeline including:
    // - Image loading
    // - Running beaker detection
    // - TOML parsing
    // - Bounding box image generation
    //
    // Run with: cargo test -- --ignored test_detection_with_real_image

    use beaker_gui::views::detection::DetectionView;

    let image_path = "../example.jpg";

    // Create detection view - this runs the full detection pipeline
    let view =
        DetectionView::new(image_path).expect("Failed to create detection view with example.jpg");

    // Verify we got at least 1 detection
    assert!(
        view.detections().len() >= 1,
        "Expected at least 1 detection in example.jpg, got {}",
        view.detections().len()
    );

    // Verify detections have valid data
    for (idx, detection) in view.detections().iter().enumerate() {
        assert!(
            !detection.class_name.is_empty(),
            "Detection {} has empty class name",
            idx
        );
        assert!(
            detection.confidence >= 0.0 && detection.confidence <= 1.0,
            "Detection {} has invalid confidence: {}",
            idx,
            detection.confidence
        );
    }

    eprintln!(
        "✓ Successfully detected {} object(s)",
        view.detections().len()
    );
}

// Note: Tests that require actual detection are marked with #[ignore]
// Run them with: cargo test -- --ignored
//
// Quick tests (no model download):
// - App structure creation
// - Style application
// - Constants validity
//
// Full integration test with model download:
// - test_detection_with_real_image (run with --ignored flag)
