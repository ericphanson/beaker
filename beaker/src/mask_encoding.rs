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
    pub format: String,      // e.g., "rle-binary-v1 | gzip | base64"
    pub start_value: u8,     // 0 or 1
    pub order: String,       // "row-major"
    pub data: String,        // base64(gzip(rle_csv))
}

/// Encode a binary mask (0/1 values) into the specified TOML-friendly format.
/// `mask` must be length == width*height, row-major (top-left to bottom-right).
pub fn encode_mask_to_entry(
    mask: &[u8],
    width: u32,
    height: u32,
    start_value: u8, // typically 0; if the first pixel is 1, we'll emit a leading 0-run
) -> Result<MaskEntry, String> {
    // Basic validation
    let expected_len = (width as usize) * (height as usize);
    if mask.len() != expected_len {
        return Err(format!(
            "mask length {} != width*height {}",
            mask.len(),
            expected_len
        ));
    }
    if start_value > 1 {
        return Err("start_value must be 0 or 1".into());
    }
    if let Some((i, bad)) = mask
        .iter()
        .enumerate()
        .find(|(_, &v)| v != 0 && v != 1)
    {
        return Err(format!("mask contains non-binary value {} at index {}", bad, i));
    }

    // RLE (binary, alternating runs starting at start_value)
    let mut runs: Vec<usize> = Vec::new();
    let mut current_val: u8 = start_value;
    let mut run_len: usize = 0;

    for &px in mask {
        if px == current_val {
            run_len += 1;
        } else {
            // close current run, flip value, start new run
            runs.push(run_len);
            current_val ^= 1;
            run_len = 1;
        }
    }
    // push the final run
    runs.push(run_len);

    // Convert runs to a comma-separated ASCII string
    let mut rle = String::new();
    // Reserve a bit to reduce reallocations (cheap heuristic)
    rle.reserve(runs.len() * 3);
    for (i, r) in runs.iter().enumerate() {
        if i > 0 {
            rle.push(',');
        }
        use std::fmt::Write as _;
        write!(&mut rle, "{}", r).unwrap();
    }

    // gzip the RLE string
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(rle.as_bytes())
        .map_err(|e| format!("gzip write failed: {}", e))?;
    let compressed = encoder
        .finish()
        .map_err(|e| format!("gzip finish failed: {}", e))?;

    // base64 encode
    let b64 = B64.encode(&compressed);

    Ok(MaskEntry {
        width,
        height,
        format: "rle-binary-v1 | gzip | base64".to_string(),
        start_value,
        order: "row-major".to_string(),
        data: b64,
    })
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
            0,0,0,0,0,0,0,0,
            0,1,1,1,1,1,0,0,
            0,1,0,0,0,1,0,0,
            0,0,0,0,0,0,0,0,
        ];
        let entry = encode_mask_to_entry(&mask, w, h, 0).unwrap();
        assert_eq!(entry.width, w);
        assert_eq!(entry.height, h);
        assert_eq!(entry.format, "rle-binary-v1 | gzip | base64");
        assert_eq!(entry.order, "row-major");
        assert_eq!(entry.start_value, 0);
        // Just sanity: data should be non-empty and base64-ish.
        assert!(entry.data.len() > 0);
        assert!(entry.data.chars().all(|c| c.is_ascii()));
    }

    #[test]
    fn test_validation() {
        // Wrong length
        let mask = vec![0, 1, 0];
        let result = encode_mask_to_entry(&mask, 2, 2, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mask length 3 != width*height 4"));

        // Invalid start_value
        let mask = vec![0, 1, 0, 1];
        let result = encode_mask_to_entry(&mask, 2, 2, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("start_value must be 0 or 1"));

        // Non-binary value
        let mask = vec![0, 1, 2, 1];
        let result = encode_mask_to_entry(&mask, 2, 2, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mask contains non-binary value 2 at index 2"));
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
    }
}