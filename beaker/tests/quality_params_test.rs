use beaker::quality_types::{QualityParams, QualityRawData, QualityScores, TriageParams};

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
        relative_blur: 0.3,
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
        relative_blur: 0.3,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    // Verify final score is computed
    assert!(scores.final_score > 0.0);
    assert!(scores.final_score <= 100.0);

    // Verify component scores
    assert_eq!(scores.paq2piq_score, 80.0);
    assert!(scores.blur_score >= 0.0 && scores.blur_score <= 100.0);

    // Verify grids have correct shape
    assert_eq!(scores.blur_probability.len(), 20);
    assert_eq!(scores.blur_weights.len(), 20);
}

#[test]
fn test_quality_scores_blur_score_from_relative_blur() {
    let raw_with = |relative_blur: f32, tenengrad: f32| QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 90.0,
        paq2piq_local: [[80u8; 20]; 20],
        tenengrad_224: [[tenengrad; 20]; 20],
        tenengrad_112: [[tenengrad / 2.0; 20]; 20],
        median_tenengrad_224: tenengrad,
        scale_ratio: 0.5,
        relative_blur,
        model_version: "quality-model-v1".to_string(),
    };
    let params = QualityParams::default();

    // At or below the sharp anchor: no blur penalty, final score equals PaQ-2-PiQ
    let sharp = QualityScores::compute(&raw_with(0.15, 1.0), &params);
    assert_eq!(sharp.blur_score, 0.0);
    assert_eq!(sharp.final_score, 90.0);

    // Nothing lost on re-blurring: full blur score, weight 1 - alpha
    let blurry = QualityScores::compute(&raw_with(1.0, 1.0), &params);
    assert_eq!(blurry.blur_score, 100.0);
    assert!((blurry.final_score - 90.0 * (1.0 - params.alpha)).abs() < 1e-3);

    // The blur weight never drops below min_weight
    let strong = QualityParams {
        alpha: 1.0,
        ..Default::default()
    };
    let floored = QualityScores::compute(&raw_with(1.0, 1.0), &strong);
    assert!((floored.final_score - 90.0 * strong.min_weight).abs() < 1e-3);

    // The Tenengrad grids change the local maps but not the blur score
    let high_energy = QualityScores::compute(&raw_with(0.575, 1.0), &params);
    let low_energy = QualityScores::compute(&raw_with(0.575, 0.001), &params);
    assert!((high_energy.blur_score - 50.0).abs() < 1e-3);
    assert_eq!(high_energy.blur_score, low_energy.blur_score);
    assert_eq!(high_energy.final_score, low_energy.final_score);
    assert!(low_energy.blur_probability[0][0] > high_energy.blur_probability[0][0] + 0.5);
}

#[test]
fn test_triage_params_default_values() {
    let params = TriageParams::default();

    assert_eq!(params.core_ring_sharpness_ratio_bad, 1.19);
    assert_eq!(params.grid_cells_covered_bad, 6.15);
    assert_eq!(params.delta_bad_core_ring_sharpness_ratio, 0.4);
    assert_eq!(params.delta_bad_grid_cells_covered, 1.5);
    assert_eq!(params.delta_good_core_ring_sharpness_ratio, 0.0);
    assert_eq!(params.delta_good_grid_cells_covered, 0.0);
}

#[test]
fn test_triage_params_custom_values() {
    let params = TriageParams {
        core_ring_sharpness_ratio_bad: 1.5,
        delta_bad_core_ring_sharpness_ratio: 0.3,
        ..Default::default()
    };

    assert_eq!(params.core_ring_sharpness_ratio_bad, 1.5);
    assert_eq!(params.delta_bad_core_ring_sharpness_ratio, 0.3);
    assert_eq!(params.grid_cells_covered_bad, 6.15); // Still default
}

#[test]
fn test_triage_decision_with_custom_params() {
    use beaker::blur_detection::triage_decision;

    let params = TriageParams {
        core_ring_sharpness_ratio_bad: 1.0,
        grid_cells_covered_bad: 5.0,
        delta_bad_core_ring_sharpness_ratio: 0.2,
        delta_bad_grid_cells_covered: 1.0,
        delta_good_core_ring_sharpness_ratio: 0.1,
        delta_good_grid_cells_covered: 0.5,
    };

    // Test case 1: Should be BAD (low core/ring ratio)
    let (decision, _) = triage_decision(0.7, 8.0, &params);
    assert_eq!(decision, "bad", "Low core/ring ratio should be bad");

    // Test case 2: Should be BAD (low coverage)
    let (decision, _) = triage_decision(1.5, 3.0, &params);
    assert_eq!(decision, "bad", "Low coverage should be bad");

    // Test case 3: Should be GOOD (high values on both)
    let (decision, _) = triage_decision(1.5, 8.0, &params);
    assert_eq!(decision, "good", "High values on both should be good");

    // Test case 4: Should be UNKNOWN (in buffer zone)
    let (decision, _) = triage_decision(0.9, 8.0, &params);
    assert_eq!(decision, "unknown", "Buffer zone should be unknown");
}
