use beaker::quality_types::{QualityParams, QualityRawData, QualityScores};

#[test]
fn test_quality_params_default_values() {
    let params = QualityParams::default();

    assert_eq!(params.beta, 1.2);
    assert_eq!(params.tau_ten_224, 0.02);
    assert_eq!(params.p_floor, 0.0);
    assert_eq!(params.alpha, 0.7);
    assert_eq!(params.min_weight, 0.2);
    assert_eq!(params.s_ref, 96.0);
    assert_eq!(params.cov_ref, 4.0);
    assert_eq!(params.core_ratio, 0.60);
}

#[test]
fn test_quality_params_custom_values() {
    let params = QualityParams {
        beta: 1.5,
        alpha: 0.8,
        ..Default::default()
    };

    assert_eq!(params.beta, 1.5);
    assert_eq!(params.alpha, 0.8);
    assert_eq!(params.tau_ten_224, 0.02); // Still default
}

#[test]
fn test_quality_raw_data_creation() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 75.5,
        paq2piq_local: [[50u8; 20]; 20],
        tenengrad_224: [[0.1f32; 20]; 20],
        tenengrad_112: [[0.05f32; 20]; 20],
        median_tenengrad_224: 0.08,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    assert_eq!(raw.input_width, 640);
    assert_eq!(raw.input_height, 480);
    assert_eq!(raw.paq2piq_global, 75.5);
    assert_eq!(raw.model_version, "quality-model-v1");
}


#[test]
fn test_quality_scores_compute_from_raw() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 80.0,
        paq2piq_local: [[60u8; 20]; 20],
        tenengrad_224: [[0.05f32; 20]; 20],
        tenengrad_112: [[0.025f32; 20]; 20],
        median_tenengrad_224: 0.04,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    // Verify final score is computed
    assert!(scores.final_score > 0.0);
    assert!(scores.final_score <= 100.0);

    // Verify component scores
    assert_eq!(scores.paq2piq_score, 80.0);
    assert!(scores.blur_score >= 0.0 && scores.blur_score <= 1.0);

    // Verify grids have correct shape
    assert_eq!(scores.blur_probability.len(), 20);
    assert_eq!(scores.blur_weights.len(), 20);
}

#[test]
fn test_quality_scores_compute_no_blur() {
    // High Tenengrad = sharp = low blur probability
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 90.0,
        paq2piq_local: [[80u8; 20]; 20],
        tenengrad_224: [[1.0f32; 20]; 20], // Very high gradient
        tenengrad_112: [[0.5f32; 20]; 20],
        median_tenengrad_224: 0.8,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    // Low blur probability should mean blur_score is low
    assert!(
        scores.blur_score < 0.5,
        "Sharp image should have low blur score"
    );

    // Low blur means high weight, so final_score should be close to paq2piq
    assert!(
        scores.final_score > 85.0,
        "Sharp image should preserve quality score"
    );
}
