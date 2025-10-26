use beaker::blur_detection::compute_raw_tenengrad;
use ndarray::Array4;

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
