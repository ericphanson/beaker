use beaker::blur_detection::blur_weights_from_nchw;
use ndarray::Array4;

#[test]
fn test_blur_weights_from_nchw_still_works() {
    // Create a simple test image
    let img_nchw = Array4::<f32>::from_elem((1, 3, 224, 224), 0.5);

    // Call existing function (should still work)
    let (w20, p20, t20, global_blur) = blur_weights_from_nchw(&img_nchw, None);

    // Verify structure (same as before)
    assert_eq!(w20.shape(), &[20, 20]);
    assert_eq!(p20.shape(), &[20, 20]);
    assert_eq!(t20.shape(), &[20, 20]);
    assert!((0.0..=1.0).contains(&global_blur));
}

#[test]
fn test_blur_weights_from_nchw_produces_same_results() {
    // This test ensures backward compatibility
    // Create a high-contrast image
    let mut img_nchw = Array4::<f32>::zeros((1, 3, 224, 224));
    for i in 0..224 {
        for j in 0..224 {
            let val = if (i / 10 + j / 10) % 2 == 0 { 1.0 } else { 0.0 };
            img_nchw[[0, 0, i, j]] = val;
            img_nchw[[0, 1, i, j]] = val;
            img_nchw[[0, 2, i, j]] = val;
        }
    }

    let (w20, _p20, _t20, global_blur) = blur_weights_from_nchw(&img_nchw, None);

    // High contrast = low blur probability
    assert!(
        global_blur < 0.5,
        "High contrast should have low blur probability"
    );

    // Weights should be close to 1.0 (low blur means high weight)
    let mean_weight: f32 = w20.iter().sum::<f32>() / 400.0;
    assert!(mean_weight > 0.7, "Low blur should produce high weights");
}
