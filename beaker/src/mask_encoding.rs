//! Mask encoding functionality for storing binary masks in TOML metadata.
//!
//! This module provides RLE (Run Length Encoding) + gzip + base64 encoding
//! for binary masks to efficiently store them in TOML metadata files.

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use flate2::{write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use std::io::Write;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MaskEntry {
    pub width: u32,
    pub height: u32,
    pub format: String,  // e.g., "rle-binary-v1 | gzip | base64"
    pub start_value: u8, // 0 or 1
    pub order: String,   // "row-major"
    pub data: String,    // base64(gzip(rle_csv))

    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<AsciiPreview>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AsciiPreview {
    pub format: String, // "ascii"
    pub width: u32,     // preview cols
    pub height: u32,    // preview rows
    pub rows: Vec<String>,
}

/// Calculate preview dimensions based on original image aspect ratio.
/// The longer dimension is fixed to 40, and the shorter dimension is scaled proportionally.
fn calculate_preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    const MAX_DIM: u32 = 40;

    if width >= height {
        // Width is longer or equal, fix width to 40
        let aspect_ratio = height as f64 / width as f64;
        let preview_height = (MAX_DIM as f64 * aspect_ratio).round() as u32;
        (MAX_DIM, preview_height.max(1))
    } else {
        // Height is longer, fix height to 40
        let aspect_ratio = width as f64 / height as f64;
        let preview_width = (MAX_DIM as f64 * aspect_ratio).round() as u32;
        (preview_width.max(1), MAX_DIM)
    }
}

/// Encode a binary mask (0/1 values) into the specified TOML-friendly format.
/// `mask` must be length == width*height, row-major (top-left to bottom-right).
pub fn encode_mask_to_entry(
    mask: &[u8],
    width: u32,
    height: u32,
    start_value: u8, // typically 0; if the first pixel is 1, we'll emit a leading 0-run
) -> Result<MaskEntry, String> {
    let preview_dims = calculate_preview_dimensions(width, height);
    encode_mask_to_entry_with_preview(mask, width, height, start_value, Some(preview_dims))
}

/// Encode a binary mask (row-major, values 0/1) to the TOML-friendly entry,
/// with an optional ASCII preview of (pw x ph).
pub fn encode_mask_to_entry_with_preview(
    mask: &[u8],
    width: u32,
    height: u32,
    start_value: u8,
    preview_dims: Option<(u32, u32)>, // e.g., Some((80, 60)) or None
) -> Result<MaskEntry, String> {
    // --- validate input ---
    let expected_len = (width as usize) * (height as usize);
    if mask.len() != expected_len {
        return Err(format!("mask length {} != {}", mask.len(), expected_len));
    }
    if start_value > 1 {
        return Err("start_value must be 0 or 1".into());
    }
    if let Some((i, bad)) = mask
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| *v != 0 && *v != 1)
    {
        return Err(format!("mask contains non-binary value {bad} at index {i}"));
    }

    // --- RLE (binary, alternating runs starting at start_value) ---
    let mut runs: Vec<usize> = Vec::with_capacity(mask.len() / 4);
    let mut current = start_value;
    let mut run_len: usize = 0;
    for &px in mask {
        if px == current {
            run_len += 1;
        } else {
            runs.push(run_len);
            current ^= 1;
            run_len = 1;
        }
    }
    runs.push(run_len);

    // --- CSV string of runs ---
    let mut rle = String::with_capacity(runs.len() * 3);
    for (i, r) in runs.iter().enumerate() {
        if i > 0 {
            rle.push(',');
        }
        use std::fmt::Write as _;
        write!(&mut rle, "{r}").unwrap();
    }

    // --- gzip and base64 ---
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(rle.as_bytes())
        .map_err(|e| e.to_string())?;
    let compressed = encoder.finish().map_err(|e| e.to_string())?;
    let b64 = B64.encode(&compressed);

    // --- optional ASCII preview ---
    let preview = match preview_dims {
        None => None,
        Some((pw, ph)) => {
            let rows = downsample_ascii(mask, width, height, pw, ph);
            Some(AsciiPreview {
                format: "ascii".to_string(),
                width: pw,
                height: ph,
                rows,
            })
        }
    };

    Ok(MaskEntry {
        width,
        height,
        format: "rle-binary-v1 | gzip | base64".to_string(),
        start_value,
        order: "row-major".to_string(),
        data: b64,
        preview,
    })
}

/// Downsample by block averaging to a fixed (pw x ph) ASCII preview.
/// Threshold at 0.5: ≥0.5 -> '#', else '.'.
fn downsample_ascii(mask: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<String> {
    let (w, h, pw, ph) = (w as usize, h as usize, pw as usize, ph as usize);
    let sx = (w as f64) / (pw as f64);
    let sy = (h as f64) / (ph as f64);

    let mut rows = Vec::with_capacity(ph);
    for oy in 0..ph {
        let y0 = (oy as f64 * sy).floor() as usize;
        let y1 = (((oy as f64 + 1.0) * sy).ceil() as usize).min(h);
        let y1 = y1.max(y0 + 1); // ensure non-empty
        let mut line = String::with_capacity(pw);
        for ox in 0..pw {
            let x0 = (ox as f64 * sx).floor() as usize;
            let x1 = (((ox as f64 + 1.0) * sx).ceil() as usize).min(w);
            let x1 = x1.max(x0 + 1);

            let mut sum = 0usize;
            for yy in y0..y1 {
                let row = &mask[yy * w..yy * w + w];
                for &pixel in row.iter().take(x1).skip(x0) {
                    sum += pixel as usize;
                }
            }
            let area = (y1 - y0) * (x1 - x0);
            line.push(if (sum * 2) >= area { '#' } else { '.' });
        }
        rows.push(line);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_shape() {
        let w = 8u32;
        let h = 4u32;
        // Simple pattern (row-major): some zeros, then ones, etc.
        let mask: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ];
        let entry = encode_mask_to_entry(&mask, w, h, 0).unwrap();
        assert_eq!(entry.width, w);
        assert_eq!(entry.height, h);
        assert_eq!(entry.format, "rle-binary-v1 | gzip | base64");
        assert_eq!(entry.order, "row-major");
        assert_eq!(entry.start_value, 0);
        // Just sanity: data should be non-empty and base64-ish.
        assert!(!entry.data.is_empty());
        assert!(entry.data.is_ascii());
    }

    #[test]
    fn test_validation() {
        // Wrong length
        let mask = vec![0, 1, 0];
        let result = encode_mask_to_entry(&mask, 2, 2, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mask length 3 != 4"));

        // Invalid start_value
        let mask = vec![0, 1, 0, 1];
        let result = encode_mask_to_entry(&mask, 2, 2, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("start_value must be 0 or 1"));

        // Non-binary value
        let mask = vec![0, 1, 2, 1];
        let result = encode_mask_to_entry(&mask, 2, 2, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("mask contains non-binary value 2 at index 2"));
    }

    #[test]
    fn test_simple_rle() {
        // All zeros
        let mask = vec![0, 0, 0, 0];
        let entry = encode_mask_to_entry(&mask, 2, 2, 0).unwrap();
        assert_eq!(entry.start_value, 0);

        // All ones
        let mask = vec![1, 1, 1, 1];
        let entry = encode_mask_to_entry(&mask, 2, 2, 0).unwrap();
        assert_eq!(entry.start_value, 0);

        // Alternating pattern
        let mask = vec![0, 1, 0, 1];
        let entry = encode_mask_to_entry(&mask, 2, 2, 0).unwrap();
        assert_eq!(entry.start_value, 0);

        // Verify preview exists with aspect ratio-aware dimensions
        assert!(entry.preview.is_some());
        let preview = entry.preview.unwrap();
        assert_eq!(preview.format, "ascii");
        // For a 2x2 mask, both dimensions should be 40 (square aspect ratio)
        assert_eq!(preview.width, 40);
        assert_eq!(preview.height, 40);
        assert_eq!(preview.rows.len(), 40);
    }

    #[test]
    fn test_preview_generation() {
        // Create a simple 4x4 mask with a pattern
        let mask = vec![1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1];
        let entry = encode_mask_to_entry_with_preview(&mask, 4, 4, 0, Some((4, 4))).unwrap();

        assert!(entry.preview.is_some());
        let preview = entry.preview.unwrap();
        assert_eq!(preview.width, 4);
        assert_eq!(preview.height, 4);
        assert_eq!(preview.rows.len(), 4);

        // Should match the pattern approximately
        assert_eq!(preview.rows[0], "##..");
        assert_eq!(preview.rows[1], "##..");
        assert_eq!(preview.rows[2], "..##");
        assert_eq!(preview.rows[3], "..##");
    }

    #[test]
    fn test_preview_aspect_ratio_calculation() {
        // Test landscape aspect ratio (1280x960) -> should be (40, 30)
        let (pw, ph) = calculate_preview_dimensions(1280, 960);
        assert_eq!(pw, 40);
        assert_eq!(ph, 30);

        // Test portrait aspect ratio (960x1280) -> should be (30, 40)
        let (pw, ph) = calculate_preview_dimensions(960, 1280);
        assert_eq!(pw, 30);
        assert_eq!(ph, 40);

        // Test square aspect ratio (1000x1000) -> should be (40, 40)
        let (pw, ph) = calculate_preview_dimensions(1000, 1000);
        assert_eq!(pw, 40);
        assert_eq!(ph, 40);

        // Test extreme aspect ratio (1600x400) -> should be (40, 10)
        let (pw, ph) = calculate_preview_dimensions(1600, 400);
        assert_eq!(pw, 40);
        assert_eq!(ph, 10);
    }

    #[test]
    fn test_no_preview() {
        let mask = vec![0, 1, 0, 1];
        let entry = encode_mask_to_entry_with_preview(&mask, 2, 2, 0, None).unwrap();
        assert!(entry.preview.is_none());
    }
}
