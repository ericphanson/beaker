use beaker::quality_processing::{compute_quality_raw, load_onnx_session_default};
use beaker::quality_types::{QualityParams, QualityScores};
use std::path::Path;

#[test]
fn test_quality_command_workflow() {
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let mut session = load_onnx_session_default().unwrap();
    let params = QualityParams::default();

    // Step 1: Compute raw data (cached)
    let raw = compute_quality_raw(test_image, &mut session).unwrap();

    // Step 2: Compute scores with params
    let scores = QualityScores::compute(&raw, &params);

    // Verify complete workflow
    assert!(scores.final_score > 0.0);
    assert!(scores.final_score <= 100.0);
    println!("Quality score: {:.1}", scores.final_score);
}

#[test]
fn test_quality_command_custom_params() {
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let mut session = load_onnx_session_default().unwrap();

    // Test with stricter blur detection
    let strict_params = QualityParams {
        tau_ten_224: 0.01,  // More sensitive to blur
        ..Default::default()
    };

    let raw = compute_quality_raw(test_image, &mut session).unwrap();
    let scores = QualityScores::compute(&raw, &strict_params);

    assert!(scores.final_score > 0.0);
    println!("Quality score (strict): {:.1}", scores.final_score);
}
