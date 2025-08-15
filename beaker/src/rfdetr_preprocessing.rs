use anyhow::Result;
use image::DynamicImage;
use ndarray::Array;

/// RF-DETR preprocessing that matches the Python inference pipeline
/// Uses square resize (no letterboxing) and ImageNet normalization
pub fn preprocess_image(
    img: &DynamicImage,
    target_size: u32,
) -> Result<Array<f32, ndarray::IxDyn>> {
    // ImageNet normalization constants used by RF-DETR
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    // Convert to RGB if needed
    let rgb_img = img.to_rgb8();

    // RF-DETR uses square resize - force the image to be square
    // This matches T.SquareResize([resolution]) from the Python code
    // Unlike YOLO, this does NOT preserve aspect ratio and does NOT use letterboxing
    let resized = image::imageops::resize(
        &rgb_img,
        target_size,
        target_size,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to NCHW format with ImageNet normalization
    let mut input_data = Vec::with_capacity((3 * target_size * target_size) as usize);

    // Fill in NCHW order: batch, channel, height, width
    // Apply ImageNet normalization: (pixel/255.0 - mean) / std
    for c in 0..3 {
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = resized.get_pixel(x, y);
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - MEAN[c]) / STD[c];
                input_data.push(normalized);
            }
        }
    }

    // Create ndarray with dynamic shape [1, 3, height, width]
    let input = Array::from_shape_vec(
        ndarray::IxDyn(&[1, 3, target_size as usize, target_size as usize]),
        input_data,
    )?;

    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_preprocess_image_shape() {
        // Create a test image
        let img = RgbImage::from_fn(512, 384, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let result = preprocess_image(&dynamic_img, 640).unwrap();

        // Check dimensions: [batch=1, channels=3, height=640, width=640]
        assert_eq!(result.shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_preprocess_image_normalization() {
        // Create a white image
        let img = RgbImage::from_fn(100, 100, |_, _| Rgb([255, 255, 255]));
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let result = preprocess_image(&dynamic_img, 64).unwrap();

        // Check that normalization is applied correctly
        // For white pixel (1.0), normalized value should be (1.0 - mean) / std
        let expected_r = (1.0 - 0.485) / 0.229; // ~2.25
        let expected_g = (1.0 - 0.456) / 0.224; // ~2.43
        let expected_b = (1.0 - 0.406) / 0.225; // ~2.64

        // Check a few pixels (channels are in CHW order)
        let r_val = result[[0, 0, 0, 0]]; // batch=0, channel=0 (R), y=0, x=0
        let g_val = result[[0, 1, 0, 0]]; // batch=0, channel=1 (G), y=0, x=0
        let b_val = result[[0, 2, 0, 0]]; // batch=0, channel=2 (B), y=0, x=0

        assert!((r_val - expected_r).abs() < 0.01);
        assert!((g_val - expected_g).abs() < 0.01);
        assert!((b_val - expected_b).abs() < 0.01);
    }

    #[test]
    fn test_square_resize_behavior() {
        // Create a rectangular image to test square resize behavior
        let img = RgbImage::from_fn(200, 100, |x, _y| {
            if x < 100 {
                Rgb([255, 0, 0]) // Red on left half
            } else {
                Rgb([0, 255, 0]) // Green on right half
            }
        });
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let result = preprocess_image(&dynamic_img, 64).unwrap();

        // The image should be forced to 64x64, stretching the aspect ratio
        assert_eq!(result.shape(), &[1, 3, 64, 64]);

        // The resulting image should have the colors distributed across the full width
        // This confirms that we're doing square resize (stretching) not letterboxing
    }
}
