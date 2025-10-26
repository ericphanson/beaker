use beaker::quality_types::{
    ColorMap, HeatmapStyle, QualityParams, QualityRawData, QualityScores, QualityVisualization,
};
use beaker::quality_visualization::{
    apply_colormap, bilinear_sample, composite_with_alpha, render_heatmap_to_buffer, render_overlay,
};
use image::{DynamicImage, Rgba, RgbaImage};
use ndarray::Array2;

#[test]
fn test_bilinear_sample_corners() {
    let data = Array2::from_shape_fn((20, 20), |(i, j)| (i * 20 + j) as f32);

    // Test corners (should return exact values)
    assert_eq!(bilinear_sample(&data, 0.0, 0.0), 0.0);
    assert_eq!(bilinear_sample(&data, 19.0, 0.0), 19.0);
    assert_eq!(bilinear_sample(&data, 0.0, 19.0), 380.0);
    assert_eq!(bilinear_sample(&data, 19.0, 19.0), 399.0);
}

#[test]
fn test_bilinear_sample_interpolation() {
    // Simple 2x2 grid
    let mut data = Array2::zeros((2, 2));
    data[[0, 0]] = 0.0;
    data[[0, 1]] = 1.0;
    data[[1, 0]] = 0.0;
    data[[1, 1]] = 1.0;

    // Center should be average
    let center = bilinear_sample(&data, 0.5, 0.5);
    assert!((center - 0.5).abs() < 1e-5);
}

#[test]
fn test_bilinear_sample_out_of_bounds() {
    let data = Array2::from_shape_fn((20, 20), |(i, j)| (i + j) as f32);

    // Should clamp to edges
    let val = bilinear_sample(&data, -1.0, -1.0);
    assert_eq!(val, 0.0);

    let val = bilinear_sample(&data, 25.0, 25.0);
    assert_eq!(val, 38.0); // data[[19, 19]]
}

#[test]
fn test_apply_colormap_grayscale() {
    let black = apply_colormap(0.0, ColorMap::Grayscale);
    assert_eq!(black, image::Rgba([0, 0, 0, 255]));

    let white = apply_colormap(1.0, ColorMap::Grayscale);
    assert_eq!(white, image::Rgba([255, 255, 255, 255]));

    let gray = apply_colormap(0.5, ColorMap::Grayscale);
    assert_eq!(gray, image::Rgba([127, 127, 127, 255]));
}

#[test]
fn test_apply_colormap_all_variants() {
    let value = 0.5;

    // Test all colormaps produce valid colors
    for colormap in [
        ColorMap::Viridis,
        ColorMap::Plasma,
        ColorMap::Inferno,
        ColorMap::Turbo,
        ColorMap::Grayscale,
    ] {
        let color = apply_colormap(value, colormap);

        // Alpha should always be 255
        assert_eq!(color[3], 255);
        // RGB values are u8, always valid (0-255)
    }
}

#[test]
fn test_apply_colormap_clamping() {
    // Values outside [0, 1] should be clamped
    let below = apply_colormap(-0.5, ColorMap::Grayscale);
    let at_zero = apply_colormap(0.0, ColorMap::Grayscale);
    assert_eq!(below, at_zero);

    let above = apply_colormap(1.5, ColorMap::Grayscale);
    let at_one = apply_colormap(1.0, ColorMap::Grayscale);
    assert_eq!(above, at_one);
}

#[test]
fn test_render_heatmap_to_buffer_size() {
    let data = [[0.5f32; 20]; 20];

    let style = HeatmapStyle {
        colormap: ColorMap::Grayscale,
        alpha: 0.7,
        size: (100, 100),
    };

    let img = render_heatmap_to_buffer(&data, &style).unwrap();

    assert_eq!(img.dimensions(), (100, 100));
}

#[test]
fn test_render_heatmap_gradient() {
    // Create gradient from 0 to 1
    let mut data = [[0.0f32; 20]; 20];
    for (i, row) in data.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (i * 20 + j) as f32 / 399.0;
        }
    }

    let style = HeatmapStyle {
        colormap: ColorMap::Grayscale,
        alpha: 0.7,
        size: (224, 224),
    };

    let img = render_heatmap_to_buffer(&data, &style).unwrap();

    // Top-left should be dark
    let top_left = img.get_pixel(0, 0);
    assert!(top_left[0] < 50);

    // Bottom-right should be bright
    let bottom_right = img.get_pixel(223, 223);
    assert!(bottom_right[0] > 200);
}

#[test]
fn test_render_heatmap_different_sizes() {
    let data = [[0.5f32; 20]; 20];

    for size in [(50, 50), (100, 100), (224, 224), (300, 200)] {
        let style = HeatmapStyle {
            colormap: ColorMap::Viridis,
            alpha: 0.7,
            size,
        };

        let img = render_heatmap_to_buffer(&data, &style).unwrap();
        assert_eq!(img.dimensions(), size);
    }
}

#[test]
fn test_composite_with_alpha_zero() {
    let mut base = RgbaImage::new(10, 10);
    base.fill(255); // White

    let mut overlay = RgbaImage::new(10, 10);
    overlay.fill(0); // Black

    // Alpha = 0 should show only base
    let result = composite_with_alpha(&base, &overlay, 0.0).unwrap();

    let pixel = result.get_pixel(5, 5);
    assert_eq!(pixel, &Rgba([255, 255, 255, 255]));
}

#[test]
fn test_composite_with_alpha_one() {
    let mut base = RgbaImage::new(10, 10);
    base.fill(255); // White

    let mut overlay = RgbaImage::new(10, 10);
    overlay.fill(0); // Black

    // Alpha = 1 should show only overlay
    let result = composite_with_alpha(&base, &overlay, 1.0).unwrap();

    let pixel = result.get_pixel(5, 5);
    assert_eq!(pixel, &Rgba([0, 0, 0, 255]));
}

#[test]
fn test_composite_with_alpha_half() {
    let mut base = RgbaImage::new(10, 10);
    for pixel in base.pixels_mut() {
        *pixel = Rgba([100, 100, 100, 255]);
    }

    let mut overlay = RgbaImage::new(10, 10);
    for pixel in overlay.pixels_mut() {
        *pixel = Rgba([200, 200, 200, 255]);
    }

    // Alpha = 0.5 should blend equally
    let result = composite_with_alpha(&base, &overlay, 0.5).unwrap();

    let pixel = result.get_pixel(5, 5);
    // (100 * 0.5 + 200 * 0.5) = 150
    assert_eq!(pixel[0], 150);
    assert_eq!(pixel[1], 150);
    assert_eq!(pixel[2], 150);
}

#[test]
fn test_composite_size_mismatch() {
    let base = RgbaImage::new(10, 10);
    let overlay = RgbaImage::new(20, 20);

    let result = composite_with_alpha(&base, &overlay, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_render_overlay_size() {
    // Create a simple test image
    let test_img = DynamicImage::new_rgb8(640, 480);

    let data = [[0.5f32; 20]; 20];
    let style = HeatmapStyle {
        colormap: ColorMap::Viridis,
        alpha: 0.5,
        size: (224, 224), // Heatmap size (will be resized to match image)
    };

    let overlay = render_overlay(&test_img, &data, &style).unwrap();

    // Result should match original image size
    assert_eq!(overlay.dimensions(), (640, 480));
}

#[test]
fn test_render_overlay_alpha_effect() {
    // Create simple black image
    let test_img = DynamicImage::new_rgb8(100, 100);

    // All white heatmap
    let data = [[1.0f32; 20]; 20];

    // Low alpha - should be mostly black
    let style_low = HeatmapStyle {
        colormap: ColorMap::Grayscale,
        alpha: 0.1,
        size: (100, 100),
    };

    let overlay_low = render_overlay(&test_img, &data, &style_low).unwrap();
    let pixel_low = overlay_low.get_pixel(50, 50);

    // High alpha - should be mostly white
    let style_high = HeatmapStyle {
        colormap: ColorMap::Grayscale,
        alpha: 0.9,
        size: (100, 100),
    };

    let overlay_high = render_overlay(&test_img, &data, &style_high).unwrap();
    let pixel_high = overlay_high.get_pixel(50, 50);

    // High alpha should produce brighter result
    assert!(pixel_high[0] > pixel_low[0]);
}

#[test]
fn test_quality_visualization_render() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 75.0,
        paq2piq_local: [[60u8; 20]; 20],
        tenengrad_224: [[0.05f32; 20]; 20],
        tenengrad_112: [[0.025f32; 20]; 20],
        median_tenengrad_224: 0.04,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    let style = HeatmapStyle {
        colormap: ColorMap::Viridis,
        alpha: 0.7,
        size: (224, 224),
    };

    let viz = QualityVisualization::render(&raw, &scores, &style).unwrap();

    // All heatmaps should be rendered
    assert!(viz.blur_probability_heatmap.is_some());
    assert!(viz.blur_weights_heatmap.is_some());
    assert!(viz.tenengrad_heatmap.is_some());

    // Verify dimensions
    let heatmap = viz.blur_probability_heatmap.unwrap();
    assert_eq!(heatmap.dimensions(), (224, 224));
}

#[test]
fn test_quality_visualization_render_blur_only() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 75.0,
        paq2piq_local: [[60u8; 20]; 20],
        tenengrad_224: [[0.05f32; 20]; 20],
        tenengrad_112: [[0.025f32; 20]; 20],
        median_tenengrad_224: 0.04,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    let style = HeatmapStyle::default();

    let heatmap = QualityVisualization::render_blur_only(&scores, &style).unwrap();

    assert_eq!(heatmap.dimensions(), (224, 224));
}

#[test]
fn test_quality_visualization_render_overlay() {
    let test_img = image::DynamicImage::new_rgb8(640, 480);

    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 75.0,
        paq2piq_local: [[60u8; 20]; 20],
        tenengrad_224: [[0.05f32; 20]; 20],
        tenengrad_112: [[0.025f32; 20]; 20],
        median_tenengrad_224: 0.04,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    let style = HeatmapStyle::default();

    let overlay =
        QualityVisualization::render_overlay_on_image(&test_img, &scores, &style).unwrap();

    // Overlay should match original image size
    assert_eq!(overlay.dimensions(), (640, 480));
}
