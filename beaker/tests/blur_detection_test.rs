use beaker::blur_detection::{
    apply_tenengrad_params, blur_score_from_ratio, compute_raw_tenengrad, compute_weights,
    fuse_probabilities, relative_blur_ratio,
};
use beaker::quality_types::QualityParams;
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
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

/// 192x128 image of 16 px blocks with pseudo-random grey levels
fn blocky_image() -> RgbImage {
    RgbImage::from_fn(192, 128, |x, y| {
        let v = ((x / 16) * 37 + (y / 16) * 101) % 7 * 40;
        image::Rgb([v as u8, v as u8, v as u8])
    })
}

#[test]
fn test_relative_blur_ratio_rises_with_blur() {
    let sharp = blocky_image();
    let ratios: Vec<f32> = [0.0, 1.0, 2.0, 4.0]
        .iter()
        .map(|&sigma| {
            let img = if sigma > 0.0 {
                image::imageops::blur(&sharp, sigma)
            } else {
                sharp.clone()
            };
            relative_blur_ratio(&DynamicImage::ImageRgb8(img))
        })
        .collect();

    assert!(
        ratios.windows(2).all(|w| w[0] < w[1]),
        "ratio should rise with every blur step: {ratios:?}"
    );
    assert!(ratios.iter().all(|r| (0.0..=1.0).contains(r)));
    assert!(ratios[0] < 0.5 && ratios[3] > 0.8, "{ratios:?}");
}

#[test]
fn test_relative_blur_ratio_featureless_image_reads_fully_blurred() {
    let flat = RgbImage::from_pixel(64, 48, image::Rgb([120, 120, 120]));
    let ratio = relative_blur_ratio(&DynamicImage::ImageRgb8(flat));
    assert_eq!(ratio, 1.0);
    assert_eq!(blur_score_from_ratio(ratio), 100.0);
}

#[test]
fn test_relative_blur_ratio_small_and_non_rgb_images() {
    // Narrower than the blur kernels in one or both directions
    for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 3), (5, 5), (13, 4)] {
        let img = RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 50 + y * 30) as u8; 3]));
        let ratio = relative_blur_ratio(&DynamicImage::ImageRgb8(img));
        assert!((0.0..=1.0).contains(&ratio), "{w}x{h}: {ratio}");
    }

    // A grey image gives the same answer as its RGB equivalent
    let rgb = blocky_image();
    let grey = GrayImage::from_fn(192, 128, |x, y| image::Luma([rgb.get_pixel(x, y)[0]]));
    let from_rgb = relative_blur_ratio(&DynamicImage::ImageRgb8(rgb));
    let from_grey = relative_blur_ratio(&DynamicImage::ImageLuma8(grey));
    assert!((from_rgb - from_grey).abs() < 1e-5);
}

/// Exact values from docs/blur-score/reference.py for the same pixel pattern.
/// These pin the kernel radius, border handling, Sobel weights and luma weights.
#[test]
fn test_relative_blur_ratio_matches_python_reference() {
    let pattern = |w: u32, h: u32| {
        RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                ((x * 97 + y * 31) % 256) as u8,
                ((x * 13 + y * 57) % 256) as u8,
                ((x * 5 + y * 201) % 256) as u8,
            ])
        })
    };

    for (w, h, expected) in [(7, 7, 0.152481), (13, 13, 0.101855), (40, 30, 0.034853)] {
        let rgb = pattern(w, h);
        let ratio = relative_blur_ratio(&DynamicImage::ImageRgb8(rgb.clone()));
        assert!(
            (ratio - expected).abs() < 2e-5,
            "{w}x{h}: got {ratio}, reference {expected}"
        );

        // Detection crops arrive as RGBA; opaque alpha must not change the result
        let rgba = RgbaImage::from_fn(w, h, |x, y| {
            let p = rgb.get_pixel(x, y);
            image::Rgba([p[0], p[1], p[2], 255])
        });
        assert_eq!(relative_blur_ratio(&DynamicImage::ImageRgba8(rgba)), ratio);
    }
}

#[test]
fn test_blur_score_from_ratio_scale() {
    assert_eq!(blur_score_from_ratio(0.0), 0.0);
    assert_eq!(blur_score_from_ratio(0.15), 0.0);
    assert_eq!(blur_score_from_ratio(1.0), 100.0);
    assert_eq!(blur_score_from_ratio(1.5), 100.0);
    assert!((blur_score_from_ratio(0.575) - 50.0).abs() < 1e-4);
}
