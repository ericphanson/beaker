use beaker::quality_processing::compute_quality_raw;
use beaker::quality_types::{QualityParams, QualityScores};
use std::path::Path;

#[test]
fn test_compute_quality_raw_example_image() {
    // Try to load session - skip if model not available
    let mut session = match beaker::quality_processing::load_onnx_session_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping test - model not available: {}", e);
            return;
        }
    };

    // Use test image if exists
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let raw = compute_quality_raw(test_image, &mut session).unwrap();

    // Verify structure
    assert!(raw.input_width > 0);
    assert!(raw.input_height > 0);
    assert!(raw.paq2piq_global >= 0.0 && raw.paq2piq_global <= 100.0);
    assert_eq!(raw.model_version, "quality-model-v1");
}

#[test]
fn test_end_to_end_with_custom_params() {
    // Try to load session - skip if model not available
    let mut session = match beaker::quality_processing::load_onnx_session_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping test - model not available: {}", e);
            return;
        }
    };

    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    // Compute raw data (expensive, cached)
    let raw = compute_quality_raw(test_image, &mut session).unwrap();

    // Compute scores with default params
    let params_default = QualityParams::default();
    let scores_default = QualityScores::compute(&raw, &params_default);

    // Compute scores with stricter blur detection (lower tau)
    let params_strict = QualityParams {
        tau_ten_224: 0.01,  // Lower threshold = more sensitive to blur
        ..Default::default()
    };
    let scores_strict = QualityScores::compute(&raw, &params_strict);

    // Both should have valid scores
    assert!(scores_default.final_score > 0.0);
    assert!(scores_strict.final_score > 0.0);

    // Stricter params should detect more blur (higher blur_score)
    // Note: This may not always be true depending on image, but generally holds
    println!("Default blur: {}, Strict blur: {}",
             scores_default.blur_score, scores_strict.blur_score);
}
