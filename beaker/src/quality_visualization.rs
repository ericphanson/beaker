//! Visualization layer for quality assessment
//! Renders heatmaps and overlays from quality data

use crate::quality_types::{
    ColorMap, HeatmapStyle, QualityRawData, QualityScores, QualityVisualization,
};
use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use ndarray::Array2;

/// Bilinear interpolation for smooth upscaling
/// Samples a 2D array at fractional coordinates (u, v)
pub fn bilinear_sample(data: &Array2<f32>, u: f32, v: f32) -> f32 {
    let (rows, cols) = data.dim();

    // Clamp coordinates to valid range
    let u = u.clamp(0.0, (cols - 1) as f32);
    let v = v.clamp(0.0, (rows - 1) as f32);

    // Get integer and fractional parts
    let u0 = u.floor() as usize;
    let v0 = v.floor() as usize;
    let u1 = (u0 + 1).min(cols - 1);
    let v1 = (v0 + 1).min(rows - 1);

    let fu = u - u0 as f32;
    let fv = v - v0 as f32;

    // Bilinear interpolation
    let val00 = data[[v0, u0]];
    let val10 = data[[v0, u1]];
    let val01 = data[[v1, u0]];
    let val11 = data[[v1, u1]];

    let val0 = val00 * (1.0 - fu) + val10 * fu;
    let val1 = val01 * (1.0 - fu) + val11 * fu;

    val0 * (1.0 - fv) + val1 * fv
}

/// Apply colormap to a normalized value [0, 1]
pub fn apply_colormap(value: f32, colormap: ColorMap) -> Rgba<u8> {
    let v = value.clamp(0.0, 1.0);

    match colormap {
        ColorMap::Viridis => viridis_colormap(v),
        ColorMap::Plasma => plasma_colormap(v),
        ColorMap::Inferno => inferno_colormap(v),
        ColorMap::Turbo => turbo_colormap(v),
        ColorMap::Grayscale => {
            let intensity = (v * 255.0) as u8;
            Rgba([intensity, intensity, intensity, 255])
        }
    }
}

// Viridis colormap (approximation)
fn viridis_colormap(t: f32) -> Rgba<u8> {
    // Simplified Viridis: purple -> blue -> green -> yellow
    let r = ((-4.5 * t + 11.0) * t - 4.5).clamp(0.0, 1.0);
    let g = ((5.0 * t - 9.5) * t + 4.5).clamp(0.0, 1.0);
    let b = ((-1.5 * t + 1.0) * t + 0.5).clamp(0.0, 1.0);

    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

// Plasma colormap (approximation)
fn plasma_colormap(t: f32) -> Rgba<u8> {
    // Simplified Plasma: dark blue -> purple -> red -> yellow
    let r = ((4.0 * t - 1.5) * t + 0.1).clamp(0.0, 1.0);
    let g = ((-4.0 * t + 4.0) * t).clamp(0.0, 1.0);
    let b = ((-1.5 * t + 0.5) * t + 0.9).clamp(0.0, 1.0);

    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

// Inferno colormap (approximation)
fn inferno_colormap(t: f32) -> Rgba<u8> {
    // Simplified Inferno: black -> dark red -> orange -> yellow -> white
    let r = ((3.5 * t - 1.0) * t + 0.05).clamp(0.0, 1.0);
    let g = ((4.0 * t - 3.5) * t + 0.5) * t;
    let g = g.clamp(0.0, 1.0);
    let b = ((10.0 * t - 7.0) * t + 0.1).clamp(0.0, 1.0);

    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

// Turbo colormap (approximation)
fn turbo_colormap(t: f32) -> Rgba<u8> {
    // Simplified Turbo: blue -> cyan -> green -> yellow -> red
    let r = ((6.0 * t - 3.0) * t * t).clamp(0.0, 1.0);
    let g = (-4.0 * (t - 0.5).powi(2) + 1.0).clamp(0.0, 1.0);
    let b = ((-6.0 * t + 3.0) * (1.0 - t)).clamp(0.0, 1.0);

    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

/// Render a 20x20 data grid to an image buffer with colormap
pub fn render_heatmap_to_buffer(data: &[[f32; 20]; 20], style: &HeatmapStyle) -> Result<RgbaImage> {
    let (width, height) = style.size;
    let mut img = ImageBuffer::new(width, height);

    // Convert fixed array to ndarray for interpolation
    let data_array = Array2::from_shape_fn((20, 20), |(i, j)| data[i][j]);

    // Render each pixel with bilinear interpolation
    for y in 0..height {
        for x in 0..width {
            // Map pixel to data coordinates
            let u = (x as f32 / width as f32) * 19.0;
            let v = (y as f32 / height as f32) * 19.0;

            // Sample with interpolation
            let value = bilinear_sample(&data_array, u, v);

            // Apply colormap
            let color = apply_colormap(value, style.colormap);

            img.put_pixel(x, y, color);
        }
    }

    Ok(img)
}

/// Composite two RGBA images with alpha blending
pub fn composite_with_alpha(
    base: &RgbaImage,
    overlay: &RgbaImage,
    overlay_alpha: f32,
) -> Result<RgbaImage> {
    let (width, height) = base.dimensions();

    if overlay.dimensions() != (width, height) {
        anyhow::bail!("Images must have same dimensions for compositing");
    }

    let mut result = base.clone();
    let alpha = overlay_alpha.clamp(0.0, 1.0);

    for y in 0..height {
        for x in 0..width {
            let base_pixel = base.get_pixel(x, y);
            let overlay_pixel = overlay.get_pixel(x, y);

            // Alpha blending
            let r = ((1.0 - alpha) * base_pixel[0] as f32 + alpha * overlay_pixel[0] as f32) as u8;
            let g = ((1.0 - alpha) * base_pixel[1] as f32 + alpha * overlay_pixel[1] as f32) as u8;
            let b = ((1.0 - alpha) * base_pixel[2] as f32 + alpha * overlay_pixel[2] as f32) as u8;
            let a = 255u8; // Result is always opaque

            result.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    Ok(result)
}

/// Render overlay of heatmap on original image
pub fn render_overlay(
    original: &DynamicImage,
    heatmap_data: &[[f32; 20]; 20],
    style: &HeatmapStyle,
) -> Result<RgbaImage> {
    // First render heatmap at its native size
    let heatmap = render_heatmap_to_buffer(heatmap_data, style)?;

    // Resize heatmap to match original image size
    let (orig_width, orig_height) = (original.width(), original.height());
    let heatmap_resized = image::imageops::resize(
        &heatmap,
        orig_width,
        orig_height,
        image::imageops::FilterType::Lanczos3,
    );

    // Composite with original
    let original_rgba = original.to_rgba8();
    composite_with_alpha(&original_rgba, &heatmap_resized, style.alpha)
}

#[allow(dead_code)]
impl QualityVisualization {
    /// Render all heatmaps from raw data and scores
    pub fn render(
        raw: &QualityRawData,
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<Self> {
        Ok(Self {
            blur_probability_heatmap: Some(render_heatmap_to_buffer(
                &scores.blur_probability,
                style,
            )?),
            blur_weights_heatmap: Some(render_heatmap_to_buffer(&scores.blur_weights, style)?),
            tenengrad_heatmap: Some(render_heatmap_to_buffer(&raw.tenengrad_224, style)?),
            blur_overlay: None, // Can render lazily if needed
        })
    }

    /// Render only blur probability heatmap (fast: ~3-4ms)
    pub fn render_blur_only(scores: &QualityScores, style: &HeatmapStyle) -> Result<RgbaImage> {
        render_heatmap_to_buffer(&scores.blur_probability, style)
    }

    /// Render overlay of blur probability on original image
    pub fn render_overlay_on_image(
        original: &DynamicImage,
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<RgbaImage> {
        render_overlay(original, &scores.blur_probability, style)
    }
}
