use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rawloader::{decode_file, RawImage, RawImageData, CFA};
use std::path::Path;

/// Represents a raw image that can be processed through the YOLO pipeline
pub struct RawImageProcessor {
    pub raw_data: RawImage,
    pub preview_image: DynamicImage,
}

impl RawImageProcessor {
    /// Load a raw image file and create a JPEG preview for YOLO processing
    pub fn load_raw_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Load the raw image using rawloader
        let raw_data = decode_file(&path)?;

        // Convert raw data to a preview image for YOLO processing
        let preview_image = raw_to_preview_image(&raw_data)?;

        Ok(RawImageProcessor {
            raw_data,
            preview_image,
        })
    }

    /// Get the preview image for YOLO processing
    pub fn get_preview_image(&self) -> &DynamicImage {
        &self.preview_image
    }

    /// Crop the raw image data based on bounding box coordinates and save as DNG
    pub fn crop_and_save_dng(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        output_path: &Path,
    ) -> Result<()> {
        // Scale the bounding box coordinates from the preview image to the raw image
        let (raw_width, raw_height) = (self.raw_data.width as u32, self.raw_data.height as u32);
        let (preview_width, preview_height) = self.preview_image.dimensions();

        let scale_x = raw_width as f64 / preview_width as f64;
        let scale_y = raw_height as f64 / preview_height as f64;

        let raw_x = (x as f64 * scale_x) as u32;
        let raw_y = (y as f64 * scale_y) as u32;
        let raw_width = (width as f64 * scale_x) as u32;
        let raw_height = (height as f64 * scale_y) as u32;

        // Ensure crop bounds are within the raw image
        let raw_x = raw_x.min(self.raw_data.width as u32);
        let raw_y = raw_y.min(self.raw_data.height as u32);
        let raw_width = raw_width.min(self.raw_data.width as u32 - raw_x);
        let raw_height = raw_height.min(self.raw_data.height as u32 - raw_y);

        // Create a cropped raw image
        let cropped_raw = crop_raw_image(&self.raw_data, raw_x, raw_y, raw_width, raw_height)?;

        // Save as DNG
        save_raw_as_dng(&cropped_raw, output_path)?;

        Ok(())
    }
}

/// Convert raw image data to a preview image for YOLO processing
fn raw_to_preview_image(raw: &RawImage) -> Result<DynamicImage> {
    // Get the raw pixel data based on the data type
    let (pixels, data_max) = match &raw.data {
        RawImageData::Integer(data) => {
            let max_val = *data.iter().max().unwrap_or(&0) as f64;
            (data.iter().map(|&x| x as f64).collect::<Vec<_>>(), max_val)
        }
        RawImageData::Float(data) => {
            let max_val = data.iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
            (data.iter().map(|&x| x as f64).collect::<Vec<_>>(), max_val)
        }
    };

    // Simple debayering and conversion to RGB
    let rgb_data = decode_raw_to_rgb(&pixels, raw.width, raw.height, data_max)?;

    // Create an ImageBuffer from the RGB data
    let img_buffer =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(raw.width as u32, raw.height as u32, rgb_data)
            .ok_or_else(|| anyhow!("Failed to create image buffer from raw data"))?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

/// Decode raw image data to RGB format
fn decode_raw_to_rgb(
    pixels: &[f64],
    width: usize,
    height: usize,
    data_max: f64,
) -> Result<Vec<u8>> {
    let mut rgb_data = Vec::with_capacity(width * height * 3);

    // Simple debayering - this is a basic implementation
    // For production use, you might want more sophisticated debayering
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = y * width + x;
            if pixel_idx >= pixels.len() {
                continue;
            }

            let pixel_value = pixels[pixel_idx];

            // Normalize to 8-bit
            let normalized = ((pixel_value / data_max) * 255.0).clamp(0.0, 255.0) as u8;

            // Simple color assignment based on Bayer pattern
            // This is a simplified approach - real debayering is more complex
            let (r, g, b) = match ((x % 2), (y % 2)) {
                (0, 0) => (normalized, normalized / 2, 0), // Red pixel
                (1, 0) => (normalized / 2, normalized, normalized / 2), // Green pixel
                (0, 1) => (normalized / 2, normalized, normalized / 2), // Green pixel
                (1, 1) => (0, normalized / 2, normalized), // Blue pixel
                _ => unreachable!(),
            };

            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }

    Ok(rgb_data)
}

/// Crop a raw image to the specified bounds
fn crop_raw_image(raw: &RawImage, x: u32, y: u32, width: u32, height: u32) -> Result<RawImage> {
    // Get the raw pixel data based on the data type
    let cropped_data = match &raw.data {
        RawImageData::Integer(data) => {
            let mut cropped = Vec::with_capacity((width * height) as usize);

            for row in y..(y + height) {
                if row >= raw.height as u32 {
                    break;
                }

                let start_idx = (row * raw.width as u32 + x) as usize;
                let end_idx = (row * raw.width as u32 + x + width) as usize;

                if start_idx < data.len() && end_idx <= data.len() {
                    cropped.extend_from_slice(&data[start_idx..end_idx]);
                }
            }

            RawImageData::Integer(cropped)
        }
        RawImageData::Float(data) => {
            let mut cropped = Vec::with_capacity((width * height) as usize);

            for row in y..(y + height) {
                if row >= raw.height as u32 {
                    break;
                }

                let start_idx = (row * raw.width as u32 + x) as usize;
                let end_idx = (row * raw.width as u32 + x + width) as usize;

                if start_idx < data.len() && end_idx <= data.len() {
                    cropped.extend_from_slice(&data[start_idx..end_idx]);
                }
            }

            RawImageData::Float(cropped)
        }
    };

    Ok(RawImage {
        data: cropped_data,
        width: width as usize,
        height: height as usize,
        cpp: raw.cpp,
        wb_coeffs: raw.wb_coeffs,
        xyz_to_cam: raw.xyz_to_cam,
        cfa: raw.cfa.clone(),
        crops: raw.crops,
        blacklevels: raw.blacklevels,
        whitelevels: raw.whitelevels,
        blackareas: raw.blackareas.clone(),
        clean_make: raw.clean_make.clone(),
        clean_model: raw.clean_model.clone(),
        make: raw.make.clone(),
        model: raw.model.clone(),
        orientation: raw.orientation,
    })
}

/// Save a raw image as DNG format
fn save_raw_as_dng(raw: &RawImage, output_path: &Path) -> Result<()> {
    use dng::ifd::{Ifd, IfdValue};
    use dng::tags::{ifd, IfdType};
    use dng::{DngWriter, FileType};
    use std::fs::File;
    use std::sync::Arc;

    // Convert the path to have .dng extension
    let dng_path = output_path.with_extension("dng");

    // Prepare the raw image data as bytes
    let raw_data_bytes = match &raw.data {
        RawImageData::Integer(data) => {
            // Convert u16 data to bytes (little-endian)
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &pixel in data {
                bytes.extend_from_slice(&pixel.to_le_bytes());
            }
            bytes
        }
        RawImageData::Float(data) => {
            // Convert f32 data to u16 and then to bytes
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &pixel in data {
                let pixel_u16 = (pixel.clamp(0.0, 1.0) * 65535.0) as u16;
                bytes.extend_from_slice(&pixel_u16.to_le_bytes());
            }
            bytes
        }
    };

    // Create output file
    let file = File::create(&dng_path)?;

    // Create the main IFD with all mandatory DNG metadata
    let mut main_ifd = Ifd::new(IfdType::Ifd);

    // Basic image properties
    main_ifd.insert(ifd::ImageWidth, raw.width as u32);
    main_ifd.insert(ifd::ImageLength, raw.height as u32);
    main_ifd.insert(ifd::BitsPerSample, 16u16);
    main_ifd.insert(ifd::SamplesPerPixel, 1u16); // Raw CFA data is single channel
    main_ifd.insert(ifd::PhotometricInterpretation, 32803u16); // CFA (Color Filter Array)
    main_ifd.insert(ifd::Compression, 1u16); // Uncompressed

    // DNG-specific tags (mandatory)
    main_ifd.insert(ifd::DNGVersion, &[1u8, 4, 0, 0]); // DNG version 1.4.0.0
    main_ifd.insert(ifd::DNGBackwardVersion, &[1u8, 4, 0, 0]);

    // CFA pattern from the actual raw file
    let cfa_pattern: &[u8] = match raw.cfa.pattern {
        Some(p) => match p {
            [[0, 1], [1, 2]] => &[0, 1, 1, 2], // RGGB
            [[2, 1], [1, 0]] => &[2, 1, 1, 0], // BGGR
            [[1, 0], [2, 1]] => &[1, 0, 2, 1], // GRBG
            [[1, 2], [0, 1]] => &[1, 2, 0, 1], // GBRG
            _ => &[0, 1, 1, 2],                // Default to RGGB
        },
        None => &[0, 1, 1, 2], // Default to RGGB
    };
    main_ifd.insert(ifd::CFAPattern, cfa_pattern);
    main_ifd.insert(ifd::CFARepeatPatternDim, &[2u16, 2]); // 2x2 repeat

    // Black and white levels from the actual raw file
    let black_level = if !raw.blacklevels.is_empty() {
        raw.blacklevels[0] as u16
    } else {
        64u16 // fallback value
    };

    let white_level = if !raw.whitelevels.is_empty() {
        raw.whitelevels[0] as u16
    } else {
        16383u16 // fallback value
    };

    main_ifd.insert(ifd::BlackLevel, black_level);
    main_ifd.insert(ifd::WhiteLevel, white_level);

    // Camera information
    if !raw.make.is_empty() {
        main_ifd.insert(ifd::Make, raw.make.as_str());
    } else {
        main_ifd.insert(ifd::Make, "Unknown");
    }

    if !raw.model.is_empty() {
        main_ifd.insert(ifd::Model, raw.model.as_str());
    } else {
        main_ifd.insert(ifd::Model, "Raw Camera");
    }

    main_ifd.insert(ifd::UniqueCameraModel, "Beaker Raw Processor");

    // Add orientation from the raw file if it's not the default
    match raw.orientation {
        rawloader::Orientation::Normal => {} // Don't add if normal (1)
        _ => {
            // Convert rawloader orientation to EXIF orientation value
            let exif_orientation = match raw.orientation {
                rawloader::Orientation::Normal => 1u16,
                rawloader::Orientation::HorizontalFlip => 2u16,
                rawloader::Orientation::Rotate180 => 3u16,
                rawloader::Orientation::VerticalFlip => 4u16,
                rawloader::Orientation::Transpose => 5u16,
                rawloader::Orientation::Rotate90 => 6u16,
                rawloader::Orientation::Transverse => 7u16,
                rawloader::Orientation::Rotate270 => 8u16,
                rawloader::Orientation::Unknown => 1u16, // Default to normal
            };
            main_ifd.insert(ifd::Orientation, exif_orientation);
        }
    }

    // Add the actual original raw sensor dimensions and crop bounds if available
    if raw.crops.len() >= 4 {
        // crops contains [left, top, right, bottom] in raw coordinates
        // Calculate the effective active area
        let _default_crop_origin = &[
            raw.crops[1] as u32, // top
            raw.crops[0] as u32, // left
            raw.crops[3] as u32, // bottom
            raw.crops[2] as u32, // right
        ];

        // Add as DefaultCropOrigin instead of ActiveArea if the tag exists
        // For now we'll just document this in a comment since we need to check if the tag is available
        // main_ifd.insert(ifd::DefaultCropOrigin, &[raw.crops[0] as u32, raw.crops[1] as u32]);
        // main_ifd.insert(ifd::DefaultCropSize, &[(raw.crops[2] - raw.crops[0]) as u32, (raw.crops[3] - raw.crops[1]) as u32]);
    }

    // Color information from the original raw file
    main_ifd.insert(ifd::CalibrationIlluminant1, 21u16); // D65 (standard)

    // Add software information
    main_ifd.insert(ifd::Software, "Beaker Raw Processor v1.0");

    // Use white balance coefficients from the raw file if available
    if raw.wb_coeffs.len() >= 3 {
        // Convert wb_coeffs to as-shot neutral (reciprocal of wb coefficients)
        let _wb_r = raw.wb_coeffs[0] as f64;
        let _wb_g = raw.wb_coeffs[1] as f64;
        let _wb_b = raw.wb_coeffs[2] as f64;

        // Normalize to green
        let _as_shot_r = _wb_g / _wb_r;
        let _as_shot_g = 1.0;
        let _as_shot_b = _wb_g / _wb_b;

        // For now, we'll skip the complex f64 arrays due to DNG crate limitations
        // In a full implementation, these would be included:
        // main_ifd.insert(ifd::AsShotNeutral, &[as_shot_r, as_shot_g, as_shot_b]);
    }

    // Use color matrix from raw file if available
    if raw.xyz_to_cam.len() >= 9 {
        // The raw file contains xyz_to_cam matrix, but DNG expects cam_to_xyz
        // For now, we'll skip this due to the f64 array limitations in the DNG crate
        // In a full implementation, we would invert the matrix and store it:
        // main_ifd.insert(ifd::ColorMatrix1, &inverted_matrix);
    }

    // Image data strip information - use proper DNG writer approach
    // The DNG writer will automatically handle positioning the image data
    main_ifd.insert(
        ifd::StripOffsets,
        IfdValue::Offsets(Arc::new(raw_data_bytes.clone())),
    );
    main_ifd.insert(ifd::StripByteCounts, raw_data_bytes.len() as u32);
    main_ifd.insert(ifd::RowsPerStrip, raw.height as u32);

    // Note: Additional EXIF metadata like ISO, exposure time, focal length, etc.
    // would ideally be copied from the original raw file, but the current DNG crate
    // has limited support for complex EXIF sub-IFDs. In a full implementation,
    // these would be preserved:
    // - ISO speed
    // - Exposure time
    // - Aperture (F-number)
    // - Focal length
    // - Date/time original
    // - Camera-specific settings

    // Write the DNG file with proper TIFF structure
    DngWriter::write_dng(file, true, FileType::Dng, vec![main_ifd])?;

    Ok(())
}
/// Check if a file is a supported raw format
pub fn is_raw_format<P: AsRef<Path>>(path: P) -> bool {
    if let Some(extension) = path.as_ref().extension() {
        let ext = extension.to_string_lossy().to_lowercase();
        matches!(
            ext.as_str(),
            "rw2" | "raw" | "arw" | "cr2" | "cr3" | "nef" | "orf" | "dng" | "raf" | "srw" | "x3f"
        )
    } else {
        false
    }
}
