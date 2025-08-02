use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Array4, Axis};

/// Normalize image for ISNet model input
/// Expects: mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0), size=1024x1024
pub fn preprocess_image_for_isnet(img: &DynamicImage) -> Result<Array4<f32>> {
    const MODEL_SIZE: u32 = 1024;
    const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    const STD: [f32; 3] = [1.0, 1.0, 1.0];

    // Convert to RGB and resize
    let rgb_img = img.to_rgb8();
    let resized = image::imageops::resize(
        &rgb_img,
        MODEL_SIZE,
        MODEL_SIZE,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to ndarray and normalize
    let (width, height) = resized.dimensions();
    let mut input_array = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x, y);

            // Normalize each channel: (pixel/255.0 - mean) / std
            for c in 0..3 {
                let normalized = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                input_array[[0, c, y as usize, x as usize]] = normalized;
            }
        }
    }

    Ok(input_array)
}

/// Alternative preprocessing that matches the Python rembg normalization more closely
pub fn preprocess_image_for_isnet_v2(img: &DynamicImage) -> Result<Array4<f32>> {
    const MODEL_SIZE: u32 = 1024;
    const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    const STD: [f32; 3] = [1.0, 1.0, 1.0];

    // Convert to RGB
    let rgb_img = img.to_rgb8();

    // Resize using Lanczos resampling (matches PIL's LANCZOS)
    let resized = image::imageops::resize(
        &rgb_img,
        MODEL_SIZE,
        MODEL_SIZE,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to array similar to Python rembg approach
    let (width, height) = resized.dimensions();
    let mut img_array = Array::zeros((height as usize, width as usize, 3));

    // Fill the array with pixel values
    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x, y);
            for c in 0..3 {
                img_array[[y as usize, x as usize, c]] = pixel[c] as f32;
            }
        }
    }

    // Normalize by max value (similar to Python: im_ary / max(np.max(im_ary), 1e-6))
    let max_val = img_array.iter().fold(1e-6f32, |acc, &x| acc.max(x));
    img_array.mapv_inplace(|x| x / max_val);

    // Apply mean and std normalization
    let mut tmp_img = Array::zeros((height as usize, width as usize, 3));
    for c in 0..3 {
        let channel_slice = img_array.slice(s![.., .., c]);
        let normalized = channel_slice.mapv(|x| (x - MEAN[c]) / STD[c]);
        tmp_img.slice_mut(s![.., .., c]).assign(&normalized);
    }

    // Transpose to (C, H, W) and add batch dimension
    let transposed = tmp_img.permuted_axes([2, 0, 1]);
    let with_batch = transposed.insert_axis(Axis(0));

    Ok(with_batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_preprocess_image_shape() {
        // Create a test image
        let img = RgbImage::from_fn(512, 512, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let result = preprocess_image_for_isnet(&dynamic_img).unwrap();

        // Check dimensions: [batch=1, channels=3, height=1024, width=1024]
        assert_eq!(result.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_preprocess_normalization() {
        // Create a white image (should normalize to specific values)
        let img = RgbImage::from_pixel(100, 100, Rgb([255, 255, 255]));
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let result = preprocess_image_for_isnet(&dynamic_img).unwrap();

        // For white pixels (255), with mean=0.5, std=1.0: (1.0 - 0.5) / 1.0 = 0.5
        let pixel_value = result[[0, 0, 0, 0]];
        assert!((pixel_value - 0.5).abs() < 1e-6);
    }
}
