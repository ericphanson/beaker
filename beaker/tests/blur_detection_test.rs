use beaker::blur_detection::{
    apply_tenengrad_params, compute_raw_tenengrad, compute_weights, fuse_probabilities,
};
use beaker::quality_types::QualityParams;
use ndarray::{Array2, Array4};

#[test]
fn test_compute_raw_tenengrad() {
    // Create a simple 224x224 test image in NCHW format
    let img_nchw = Array4::<f32>::from_elem((1, 3, 224, 224), 0.5);

    let result = compute_raw_tenengrad(&img_nchw).unwrap();

    // Verify structure
    assert_eq!(result.t224.shape(), &[20, 20]);
    assert_eq!(result.t112.shape(), &[20, 20]);
    assert!(result.median_224 >= 0.0);
    assert!(result.scale_ratio >= 0.05 && result.scale_ratio <= 0.80);
}

#[test]
fn test_compute_raw_tenengrad_high_contrast() {
    // High contrast image should have higher Tenengrad
    let mut img_nchw = Array4::<f32>::zeros((1, 3, 224, 224));

    // Create checkerboard pattern (high gradient)
    for i in 0..224 {
        for j in 0..224 {
            let val = if (i / 10 + j / 10) % 2 == 0 { 1.0 } else { 0.0 };
            img_nchw[[0, 0, i, j]] = val;
            img_nchw[[0, 1, i, j]] = val;
            img_nchw[[0, 2, i, j]] = val;
        }
    }

    let result = compute_raw_tenengrad(&img_nchw).unwrap();

    // High contrast should produce non-zero Tenengrad
    assert!(result.median_224 > 0.0);
}

#[test]
fn test_apply_tenengrad_params() {
    let t224 = Array2::<f32>::from_elem((20, 20), 0.05);
    let t112 = Array2::<f32>::from_elem((20, 20), 0.025);
    let median_224 = 0.04;
    let scale_ratio = 0.5;
    let params = QualityParams::default();

    let (p224, p112) = apply_tenengrad_params(&t224, &t112, median_224, scale_ratio, &params);

    assert_eq!(p224.shape(), &[20, 20]);
    assert_eq!(p112.shape(), &[20, 20]);
    // All probabilities should be in [0, 1]
    assert!(p224.iter().all(|&p| (0.0..=1.0).contains(&p)));
    assert!(p112.iter().all(|&p| (0.0..=1.0).contains(&p)));
}

#[test]
fn test_fuse_probabilities() {
    let p224 = Array2::<f32>::from_elem((20, 20), 0.3);
    let p112 = Array2::<f32>::from_elem((20, 20), 0.2);

    let fused = fuse_probabilities(&p224, &p112);

    assert_eq!(fused.shape(), &[20, 20]);
    // Fused probability should be >= max of inputs (probabilistic OR)
    assert!(fused[[0, 0]] >= 0.3); // >= 0.3 implies >= 0.2
    assert!(fused[[0, 0]] <= 1.0);
}

#[test]
fn test_compute_weights() {
    let blur_prob = Array2::<f32>::from_elem((20, 20), 0.5);
    let params = QualityParams::default();

    let weights = compute_weights(&blur_prob, &params);

    assert_eq!(weights.shape(), &[20, 20]);
    // weight = (1 - alpha * p).clamp(min_weight, 1.0)
    // With alpha=0.7, p=0.5: weight = 1 - 0.7*0.5 = 0.65
    let expected = (1.0 - params.alpha * 0.5).clamp(params.min_weight, 1.0);
    assert!((weights[[0, 0]] - expected).abs() < 1e-5);
}
