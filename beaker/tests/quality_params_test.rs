use beaker::quality_types::{QualityParams, QualityRawData, QualityScores};
use std::time::SystemTime;

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
        computed_at: SystemTime::now(),
    };

    assert_eq!(raw.input_width, 640);
    assert_eq!(raw.input_height, 480);
    assert_eq!(raw.paq2piq_global, 75.5);
    assert_eq!(raw.model_version, "quality-model-v1");
}

#[test]
fn test_quality_scores_creation() {
    let params = QualityParams::default();
    let scores = QualityScores {
        final_score: 65.0,
        paq2piq_score: 75.0,
        blur_score: 0.3,
        blur_probability: [[0.3f32; 20]; 20],
        blur_weights: [[0.79f32; 20]; 20],
        params: params.clone(),
    };

    assert_eq!(scores.final_score, 65.0);
    assert_eq!(scores.paq2piq_score, 75.0);
    assert_eq!(scores.blur_score, 0.3);
    assert_eq!(scores.params.alpha, 0.7);
}
