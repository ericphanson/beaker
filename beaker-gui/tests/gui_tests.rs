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

// =============================================================================
// SNAPSHOT TESTS
// =============================================================================
// Visual regression tests that capture UI snapshots
//
// To generate/update snapshots:
//   UPDATE_SNAPSHOTS=1 cargo test
//
// After generating, compress with pngquant:
//   pngquant --force --ext .png tests/snapshots/*.png

#[test]
#[cfg(not(target_os = "macos"))] // Skip on macOS due to menu differences
fn snapshot_welcome_view() {
    use beaker_gui::views::WelcomeView;

    let mut harness = Harness::new_ui(|ui| {
        style::setup_custom_style(ui.ctx());

        let mut welcome = WelcomeView::new();
        let ctx_clone = ui.ctx().clone();
        let _action = welcome.show(&ctx_clone, ui);
    });

    harness.run();
    harness.wgpu_snapshot("welcome_view");
}

#[test]
#[cfg(not(target_os = "macos"))]
fn snapshot_directory_processing() {
    use beaker_gui::views::{DirectoryView, ProcessingStatus};
    use std::path::PathBuf;

    let mut harness = Harness::new_ui(|ui| {
        style::setup_custom_style(ui.ctx());

        // Create a DirectoryView with mock data (processing state)
        let images = vec![
            PathBuf::from("/tmp/bird1.jpg"),
            PathBuf::from("/tmp/bird2.jpg"),
            PathBuf::from("/tmp/bird3.jpg"),
            PathBuf::from("/tmp/bird4.jpg"),
            PathBuf::from("/tmp/bird5.jpg"),
        ];

        let mut view = DirectoryView::new(PathBuf::from("/tmp"), images);

        // Simulate processing state
        view.images[0].status = ProcessingStatus::Success {
            detections_count: 2,
            good_count: 2,
            unknown_count: 0,
            bad_count: 0,
            processing_time_ms: 150.0,
        };
        view.images[1].status = ProcessingStatus::Success {
            detections_count: 1,
            good_count: 0,
            unknown_count: 1,
            bad_count: 0,
            processing_time_ms: 120.0,
        };
        view.images[2].status = ProcessingStatus::Processing;
        view.images[3].status = ProcessingStatus::Waiting;
        view.images[4].status = ProcessingStatus::Waiting;

        // Set stage tracking to show progress
        view.current_stage = Some(beaker::ProcessingStage::Detection);
        view.quality_completed = 5;
        view.detection_completed = 2;

        let ctx_clone = ui.ctx().clone();
        view.show(&ctx_clone, ui);
    });

    harness.run();
    harness.wgpu_snapshot("directory_processing");
}

#[test]
#[cfg(not(target_os = "macos"))]
fn snapshot_directory_gallery() {
    use beaker_gui::views::detection::Detection;
    use beaker_gui::views::{DirectoryView, ProcessingStatus};
    use std::path::PathBuf;

    let mut harness = Harness::new_ui(|ui| {
        style::setup_custom_style(ui.ctx());

        // Create a DirectoryView with completed processing
        let images = vec![
            PathBuf::from("/tmp/bird1.jpg"),
            PathBuf::from("/tmp/bird2.jpg"),
            PathBuf::from("/tmp/bird3.jpg"),
        ];

        let mut view = DirectoryView::new(PathBuf::from("/tmp"), images);

        // Set all images to completed with detections
        view.images[0].status = ProcessingStatus::Success {
            detections_count: 2,
            good_count: 2,
            unknown_count: 0,
            bad_count: 0,
            processing_time_ms: 150.0,
        };
        view.images[0].detections = vec![
            Detection {
                class_name: "head".to_string(),
                confidence: 0.95,
                x1: 100.0,
                y1: 100.0,
                x2: 200.0,
                y2: 200.0,
                blur_score: Some(0.12),
            },
            Detection {
                class_name: "head".to_string(),
                confidence: 0.88,
                x1: 300.0,
                y1: 150.0,
                x2: 400.0,
                y2: 250.0,
                blur_score: Some(0.08),
            },
        ];

        view.images[1].status = ProcessingStatus::Success {
            detections_count: 1,
            good_count: 0,
            unknown_count: 1,
            bad_count: 0,
            processing_time_ms: 120.0,
        };
        view.images[1].detections = vec![Detection {
            class_name: "head".to_string(),
            confidence: 0.72,
            x1: 150.0,
            y1: 120.0,
            x2: 250.0,
            y2: 220.0,
            blur_score: Some(0.45),
        }];

        view.images[2].status = ProcessingStatus::Success {
            detections_count: 0,
            good_count: 0,
            unknown_count: 0,
            bad_count: 0,
            processing_time_ms: 100.0,
        };

        // Build aggregate detection list
        view.build_aggregate_detection_list();

        // Mark as processing complete (no receiver = gallery mode)
        view.progress_receiver = None;

        let ctx_clone = ui.ctx().clone();
        view.show(&ctx_clone, ui);
    });

    harness.run();
    harness.wgpu_snapshot("directory_gallery");
}
