use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use ndarray::Array2;

/// Post-process the raw model output to create a clean mask
pub fn postprocess_mask(
    raw_output: &ndarray::ArrayView2<f32>,
    original_size: (u32, u32),
    post_process: bool,
) -> Result<GrayImage> {
    let (width, height) = original_size;

    // Find min and max values for normalization
    let min_val = raw_output.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max_val = raw_output
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    // Normalize to 0-1 range
    let normalized: Array2<f32> = if (max_val - min_val).abs() < f32::EPSILON {
        // Handle case where all values are the same
        Array2::zeros(raw_output.raw_dim())
    } else {
        raw_output.mapv(|x| (x - min_val) / (max_val - min_val))
    };

    // Convert to 0-255 range and create mask image
    let mask_1024 = GrayImage::from_fn(1024, 1024, |x, y| {
        let normalized_val = normalized[[y as usize, x as usize]];
        let pixel_val = (normalized_val * 255.0).round().clamp(0.0, 255.0) as u8;
        Luma([pixel_val])
    });

    // Resize back to original dimensions
    let mask = image::imageops::resize(
        &mask_1024,
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    );

    // Apply post-processing if requested
    if post_process {
        Ok(apply_mask_postprocessing(mask))
    } else {
        Ok(mask)
    }
}

/// Apply morphological operations and Gaussian blur to smooth the mask
/// Based on: https://www.sciencedirect.com/science/article/pii/S2352914821000757
fn apply_mask_postprocessing(mask: GrayImage) -> GrayImage {
    // For now, implement a simple version
    // In a full implementation, you would use opencv-rust or imageproc for morphological operations

    // Simple Gaussian-like blur approximation
    let (width, height) = mask.dimensions();
    let mut blurred = mask.clone();

    // Apply a simple 3x3 averaging kernel (poor man's blur)
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    if nx < width && ny < height {
                        sum += mask.get_pixel(nx, ny)[0] as u32;
                        count += 1;
                    }
                }
            }

            let avg = (sum / count) as u8;
            // Apply threshold
            let final_val = if avg < 127 { 0 } else { 255 };
            blurred.put_pixel(x, y, Luma([final_val]));
        }
    }

    blurred
}

/// Create a cutout image by applying the mask with transparency
pub fn create_cutout(img: &DynamicImage, mask: &GrayImage) -> Result<RgbaImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();

    // Ensure mask matches image dimensions
    let resized_mask = if mask.dimensions() != (width, height) {
        image::imageops::resize(mask, width, height, image::imageops::FilterType::Lanczos3)
    } else {
        mask.clone()
    };

    // Create cutout by setting alpha channel based on mask
    let cutout = RgbaImage::from_fn(width, height, |x, y| {
        let original_pixel = rgba_img.get_pixel(x, y);
        let mask_value = resized_mask.get_pixel(x, y)[0];

        Rgba([
            original_pixel[0], // R
            original_pixel[1], // G
            original_pixel[2], // B
            mask_value,        // A (use mask as alpha)
        ])
    });

    Ok(cutout)
}

/// Create a cutout with a solid background color
pub fn create_cutout_with_background(
    img: &DynamicImage,
    mask: &GrayImage,
    background_color: [u8; 4], // RGBA
) -> Result<RgbaImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();

    // Ensure mask matches image dimensions
    let resized_mask = if mask.dimensions() != (width, height) {
        image::imageops::resize(mask, width, height, image::imageops::FilterType::Lanczos3)
    } else {
        mask.clone()
    };

    // Create cutout by blending with background based on mask
    let cutout = RgbaImage::from_fn(width, height, |x, y| {
        let original_pixel = rgba_img.get_pixel(x, y);
        let mask_value = resized_mask.get_pixel(x, y)[0] as f32 / 255.0;

        // Blend original pixel with background color based on mask
        let r = (original_pixel[0] as f32 * mask_value
            + background_color[0] as f32 * (1.0 - mask_value)) as u8;
        let g = (original_pixel[1] as f32 * mask_value
            + background_color[1] as f32 * (1.0 - mask_value)) as u8;
        let b = (original_pixel[2] as f32 * mask_value
            + background_color[2] as f32 * (1.0 - mask_value)) as u8;
        let a = (original_pixel[3] as f32 * mask_value
            + background_color[3] as f32 * (1.0 - mask_value)) as u8;

        Rgba([r, g, b, a])
    });

    Ok(cutout)
}

/// Simple alpha matting implementation (basic version)
pub fn apply_alpha_matting(
    img: &DynamicImage,
    mask: &GrayImage,
    foreground_threshold: u8,
    background_threshold: u8,
    _erode_size: u32, // For future implementation
) -> Result<RgbaImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();

    // Ensure mask matches image dimensions
    let resized_mask = if mask.dimensions() != (width, height) {
        image::imageops::resize(mask, width, height, image::imageops::FilterType::Lanczos3)
    } else {
        mask.clone()
    };

    // Create trimap: 0 = background, 128 = unknown, 255 = foreground
    let trimap = GrayImage::from_fn(width, height, |x, y| {
        let mask_val = resized_mask.get_pixel(x, y)[0];
        let trimap_val = if mask_val > foreground_threshold {
            255 // Foreground
        } else if mask_val < background_threshold {
            0 // Background
        } else {
            128 // Unknown - needs alpha matting
        };
        Luma([trimap_val])
    });

    // For simplicity, we'll just use the trimap as alpha directly
    // A full alpha matting implementation would solve the matting equation
    let cutout = RgbaImage::from_fn(width, height, |x, y| {
        let original_pixel = rgba_img.get_pixel(x, y);
        let trimap_val = trimap.get_pixel(x, y)[0];

        let alpha = trimap_val;

        Rgba([
            original_pixel[0],
            original_pixel[1],
            original_pixel[2],
            alpha,
        ])
    });

    Ok(cutout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_postprocess_mask() {
        // Create a test output array
        let mut raw_output = Array2::<f32>::zeros((1024, 1024));
        // Fill with some test pattern
        for i in 0..512 {
            for j in 0..512 {
                raw_output[[i, j]] = 1.0;
            }
        }

        let mask = postprocess_mask(&raw_output.view(), (512, 512), false).unwrap();
        assert_eq!(mask.dimensions(), (512, 512));
    }

    #[test]
    fn test_create_cutout() {
        use image::{Rgb, RgbImage};

        // Create test image and mask
        let img = RgbImage::from_pixel(100, 100, Rgb([255, 0, 0]));
        let mask = GrayImage::from_pixel(100, 100, Luma([255]));
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let cutout = create_cutout(&dynamic_img, &mask).unwrap();
        assert_eq!(cutout.dimensions(), (100, 100));

        // Check that a pixel has the expected values
        let pixel = cutout.get_pixel(50, 50);
        assert_eq!(pixel[0], 255); // R
        assert_eq!(pixel[1], 0); // G
        assert_eq!(pixel[2], 0); // B
        assert_eq!(pixel[3], 255); // A (from mask)
    }
}
