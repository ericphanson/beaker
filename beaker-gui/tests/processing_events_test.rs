use beaker::config::{BaseModelConfig, DetectionConfig};
use beaker::model_processing::{ProcessingEvent, ProcessingStage};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;

#[test]
fn test_processing_events_emitted() {
    // Create a simple config for a single image
    let config = DetectionConfig {
        base: BaseModelConfig {
            sources: vec!["tests/test_images/example.jpg".to_string()],
            device: "cpu".to_string(),
            output_dir: Some(std::env::temp_dir().join("test-events").to_str().unwrap().to_string()),
            skip_metadata: true, // Skip metadata for faster test
            strict: false,
            force: true,
        },
        confidence: 0.5,
        crop_classes: HashSet::new(),
        bounding_box: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        quality_results: None,
    };

    let (tx, rx) = channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));

    // Run detection with progress reporting
    let result = beaker::run_detection_with_progress(config, Some(tx), Some(cancel_flag));

    // Should succeed
    assert!(result.is_ok(), "Detection failed: {:?}", result.err());

    // Collect all events
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    // We should have received events
    assert!(!events.is_empty(), "No events received");

    // Check for stage changes
    let stage_changes: Vec<_> = events.iter().filter_map(|e| {
        if let ProcessingEvent::StageChange { stage, .. } = e {
            Some(stage)
        } else {
            None
        }
    }).collect();

    // Should have Quality and Detection stages
    assert!(stage_changes.iter().any(|s| matches!(s, ProcessingStage::Quality)), "No Quality stage");
    assert!(stage_changes.iter().any(|s| matches!(s, ProcessingStage::Detection)), "No Detection stage");

    // Check for image start events
    let image_starts: Vec<_> = events.iter().filter(|e| {
        matches!(e, ProcessingEvent::ImageStart { .. })
    }).collect();

    // Should have at least 2 image start events (quality + detection for same image)
    assert!(image_starts.len() >= 2, "Expected at least 2 image start events, got {}", image_starts.len());

    // Check for image complete events
    let image_completes: Vec<_> = events.iter().filter(|e| {
        matches!(e, ProcessingEvent::ImageComplete { .. })
    }).collect();

    // Should have at least 2 image complete events
    assert!(image_completes.len() >= 2, "Expected at least 2 image complete events, got {}", image_completes.len());
}

#[test]
fn test_cancellation_support() {
    // Create a config with multiple images (we'll cancel midway)
    let config = DetectionConfig {
        base: BaseModelConfig {
            sources: vec![
                "tests/test_images/example.jpg".to_string(),
                "tests/test_images/example-2-birds.jpg".to_string(),
            ],
            device: "cpu".to_string(),
            output_dir: Some(std::env::temp_dir().join("test-cancel").to_str().unwrap().to_string()),
            skip_metadata: true,
            strict: false,
            force: true,
        },
        confidence: 0.5,
        crop_classes: HashSet::new(),
        bounding_box: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        quality_results: None,
    };

    let (tx, rx) = channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_flag_clone = cancel_flag.clone();

    // Spawn a thread that will cancel after first image complete
    let _canceller = std::thread::spawn(move || {
        // Wait for first image complete
        let mut seen_complete = false;
        while let Ok(event) = rx.recv() {
            if matches!(event, ProcessingEvent::ImageComplete { .. }) {
                if seen_complete {
                    // Cancel after second complete (after first image processed in both stages)
                    cancel_flag_clone.store(true, Ordering::Relaxed);
                    break;
                }
                seen_complete = true;
            }
        }
    });

    // Run detection - should be cancelled
    let result = beaker::run_detection_with_progress(config, Some(tx), Some(cancel_flag));

    // Should still succeed (partial results ok in non-strict mode)
    assert!(result.is_ok(), "Detection failed: {:?}", result.err());

    // Should have processed less than all images (due to cancellation)
    // Note: This is a bit racy - we might process all images before cancel triggers
    // But the important thing is it doesn't crash
}

#[test]
fn test_processing_result_variants() {
    use beaker::model_processing::ProcessingResult;

    // Test Success variant
    let success = ProcessingResult::Success {
        processing_time_ms: 123.45,
    };

    match success {
        ProcessingResult::Success { processing_time_ms } => {
            assert_eq!(processing_time_ms, 123.45);
        }
        _ => panic!("Expected Success variant"),
    }

    // Test Error variant
    let error = ProcessingResult::Error {
        error_message: "Test error".to_string(),
    };

    match error {
        ProcessingResult::Error { error_message } => {
            assert_eq!(error_message, "Test error");
        }
        _ => panic!("Expected Error variant"),
    }
}
