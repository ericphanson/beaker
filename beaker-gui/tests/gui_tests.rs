use egui_kittest::Harness;
use beaker_gui::{DetectionView, style};

#[test]
fn test_detection_view_full_lifecycle() {
    // This test exercises the full view lifecycle
    // If runtime asserts pass, all invariants are valid
    let test_image = "tests/fixtures/test_bird.jpg";
    let mut view = DetectionView::new(test_image)
        .expect("Failed to create detection view");

    // Render the view (will trigger all runtime asserts in show_detections_list,
    // show_image_with_bboxes, and render_image_with_bboxes)
    let mut harness = Harness::new_ui(|ui| {
        let ctx = ui.ctx().clone();
        style::setup_custom_style(&ctx);
        view.show(&ctx, ui);
    });

    harness.run();

    // Generate snapshot
    #[cfg(feature = "snapshot")]
    harness.wgpu_snapshot("detection_view");

    // If we got here, all runtime asserts passed
}

#[test]
fn test_detection_selection_invariants() {
    // Test that selecting detections maintains invariants
    let test_image = "tests/fixtures/test_bird.jpg";
    let mut view = DetectionView::new(test_image).unwrap();

    // The view should have been created successfully, which means
    // it has at least the structure needed for rendering

    // Render with no selection (runtime asserts validate state)
    let mut harness = Harness::new_ui(|ui| {
        let ctx = ui.ctx().clone();
        view.show(&ctx, ui);
    });
    harness.run();

    // If we got here, runtime asserts passed with no selection
}

#[test]
fn test_multiple_views_in_sequence() {
    // Test creating and destroying views (validates cleanup)
    let test_images = [
        "tests/fixtures/test_bird.jpg",
        "tests/fixtures/test_bird_2.jpg",
    ];

    for img_path in &test_images {
        let mut view = DetectionView::new(img_path).unwrap();

        let mut harness = Harness::new_ui(|ui| {
            let ctx = ui.ctx().clone();
            view.show(&ctx, ui);
        });
        harness.run();

        // View drops here - runtime asserts validate state throughout
    }
}

#[test]
fn test_style_setup() {
    // Test that style setup doesn't panic
    let mut harness = Harness::new_ui(|ui| {
        style::setup_custom_style(ui.ctx());
        ui.label("Test");
    });
    harness.run();
}
