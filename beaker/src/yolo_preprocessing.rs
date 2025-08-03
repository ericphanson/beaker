use anyhow::Result;
use image::DynamicImage;
use ndarray::Array;

pub fn preprocess_image(
    img: &DynamicImage,
    target_size: u32,
) -> Result<Array<f32, ndarray::IxDyn>> {
    // Convert to RGB if needed
    let rgb_img = img.to_rgb8();
    let (orig_width, orig_height) = rgb_img.dimensions();

    // Calculate letterbox resize dimensions
    let max_dim = if orig_width > orig_height {
        orig_width
    } else {
        orig_height
    };
    let scale = (target_size as f32) / (max_dim as f32);
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    // Resize image
    let resized = image::imageops::resize(
        &rgb_img,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    );

    // Create letterboxed image with gray padding (114, 114, 114)
    let mut letterboxed = image::RgbImage::new(target_size, target_size);
    for pixel in letterboxed.pixels_mut() {
        *pixel = image::Rgb([114, 114, 114]);
    }

    // Calculate offsets to center the resized image
    let x_offset = (target_size - new_width) / 2;
    let y_offset = (target_size - new_height) / 2;

    // Copy resized image to center of letterboxed image
    for y in 0..new_height {
        for x in 0..new_width {
            let src_pixel = resized.get_pixel(x, y);
            letterboxed.put_pixel(x + x_offset, y + y_offset, *src_pixel);
        }
    }

    // Convert to NCHW format and normalize
    let mut input_data = Vec::with_capacity((3 * target_size * target_size) as usize);

    // Fill in NCHW order: batch, channel, height, width
    for c in 0..3 {
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = letterboxed.get_pixel(x, y);
                let value = pixel[c] as f32 / 255.0;
                input_data.push(value);
            }
        }
    }

    // Create ndarray with dynamic shape
    let input = Array::from_shape_vec(
        ndarray::IxDyn(&[1, 3, target_size as usize, target_size as usize]),
        input_data,
    )?;

    Ok(input)
}
