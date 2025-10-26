use beaker::quality_types::QualityParams;

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
