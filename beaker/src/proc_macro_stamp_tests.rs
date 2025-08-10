// Test the new proc macro stamp generation
use crate::config::{BaseModelConfig, CutoutConfig, DetectionClass, DetectionConfig};
use beaker_stamp::Stamp;
use std::collections::HashSet;

#[test]
fn test_detection_config_stamp_generation() {
    let config1 = DetectionConfig {
        base: BaseModelConfig {
            sources: vec!["test.jpg".to_string()],
            device: "cpu".to_string(),
            output_dir: None,
            depfile: None,
            skip_metadata: false,
            strict: true,
        },
        confidence: 0.25,
        iou_threshold: 0.45,
        crop_classes: HashSet::new(),
        bounding_box: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        output_dir: None,
    };

    let mut config2 = config1.clone();
    config2.confidence = 0.5; // Different confidence

    let stamp1 = config1.stamp_hash();
    let stamp2 = config2.stamp_hash();

    assert_ne!(
        stamp1, stamp2,
        "Different configs should have different stamps"
    );
    assert!(stamp1.starts_with("sha256:"));
    assert!(stamp2.starts_with("sha256:"));
}

#[test]
fn test_cutout_config_stamp_generation() {
    let config1 = CutoutConfig {
        base: BaseModelConfig {
            sources: vec!["test.jpg".to_string()],
            device: "cpu".to_string(),
            output_dir: None,
            depfile: None,
            skip_metadata: false,
            strict: true,
        },
        post_process_mask: false,
        alpha_matting: false,
        alpha_matting_foreground_threshold: 240,
        alpha_matting_background_threshold: 10,
        alpha_matting_erode_size: 10,
        background_color: None,
        save_mask: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        output_dir: None,
    };

    let mut config2 = config1.clone();
    config2.save_mask = true; // Different save_mask

    let stamp1 = config1.stamp_hash();
    let stamp2 = config2.stamp_hash();

    assert_ne!(
        stamp1, stamp2,
        "Different configs should have different stamps"
    );
    assert!(stamp1.starts_with("sha256:"));
    assert!(stamp2.starts_with("sha256:"));
}

#[test]
fn test_config_stamp_ignores_base_fields() {
    // Test that changing base fields doesn't affect the stamp (since base is not stamped)
    let config1 = DetectionConfig {
        base: BaseModelConfig {
            sources: vec!["test.jpg".to_string()],
            device: "cpu".to_string(),
            output_dir: None,
            depfile: None,
            skip_metadata: false,
            strict: true,
        },
        confidence: 0.25,
        iou_threshold: 0.45,
        crop_classes: HashSet::new(),
        bounding_box: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        output_dir: None,
    };

    let mut config2 = config1.clone();
    config2.base.device = "coreml".to_string(); // Change base device (not stamped)

    let stamp1 = config1.stamp_hash();
    let stamp2 = config2.stamp_hash();

    assert_eq!(stamp1, stamp2, "Base fields should not affect stamps");
}

#[test]
fn test_config_stamp_includes_output_dir() {
    // Test that changing output_dir (which is stamped) affects the stamp
    let config1 = DetectionConfig {
        base: BaseModelConfig {
            sources: vec!["test.jpg".to_string()],
            device: "cpu".to_string(),
            output_dir: None,
            depfile: None,
            skip_metadata: false,
            strict: true,
        },
        confidence: 0.25,
        iou_threshold: 0.45,
        crop_classes: HashSet::new(),
        bounding_box: false,
        model_path: None,
        model_url: None,
        model_checksum: None,
        output_dir: None,
    };

    let mut config2 = config1.clone();
    config2.output_dir = Some("/tmp".to_string()); // Change output_dir (is stamped)

    let stamp1 = config1.stamp_hash();
    let stamp2 = config2.stamp_hash();

    assert_ne!(stamp1, stamp2, "Output dir changes should affect stamps");
}
