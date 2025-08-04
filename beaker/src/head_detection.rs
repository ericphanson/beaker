use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

use crate::config::HeadDetectionConfig;
use crate::model_processing::{ModelProcessor, ModelResult};
use crate::onnx_session::ModelSource;
use crate::output_manager::OutputManager;
use crate::yolo_postprocessing::{postprocess_output, Detection};
use crate::yolo_preprocessing::preprocess_image;
use log::{debug, info};

// Embed the ONNX model at compile time
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bird-head-detector.onnx"));

// Get model version from build script
pub const MODEL_VERSION: &str =
    include_str!(concat!(env!("OUT_DIR"), "/bird-head-detector.version"));

/// Core results for enhanced metadata (without config duplication)
#[derive(Serialize)]
pub struct HeadCoreResult {
    pub model_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box_path: Option<String>,
    pub detections: Vec<DetectionWithPath>,
}

#[derive(Serialize, Clone)]
pub struct DetectionWithPath {
    #[serde(flatten)]
    pub detection: Detection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
}

/// Check if a file is a supported image format
/// Get the appropriate output extension based on input file
/// PNG files output PNG to preserve transparency, others output JPG
fn get_output_extension(input_path: &Path) -> &'static str {
    if let Some(ext) = input_path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        if ext == "png" {
            "png"
        } else {
            "jpg"
        }
    } else {
        "jpg"
    }
}

pub fn create_square_crop(
    img: &DynamicImage,
    bbox: &Detection,
    output_path: &Path,
    padding: f32,
) -> Result<()> {
    let (img_width, img_height) = img.dimensions();

    // Extract bounding box coordinates
    let x1 = bbox.x1.max(0.0) as u32;
    let y1 = bbox.y1.max(0.0) as u32;
    let x2 = bbox.x2.min(img_width as f32) as u32;
    let y2 = bbox.y2.min(img_height as f32) as u32;

    // Calculate center and dimensions with padding
    let bbox_width = x2 - x1;
    let bbox_height = y2 - y1;
    let expand_w = (bbox_width as f32 * padding) as u32;
    let expand_h = (bbox_height as f32 * padding) as u32;

    let expanded_x1 = x1.saturating_sub(expand_w);
    let expanded_y1 = y1.saturating_sub(expand_h);
    let expanded_x2 = (x2 + expand_w).min(img_width);
    let expanded_y2 = (y2 + expand_h).min(img_height);

    // Calculate center and make square
    let center_x = (expanded_x1 + expanded_x2) / 2;
    let center_y = (expanded_y1 + expanded_y2) / 2;
    let new_width = expanded_x2 - expanded_x1;
    let new_height = expanded_y2 - expanded_y1;
    let size = new_width.max(new_height);
    let half_size = size / 2;

    // Calculate square crop bounds
    let crop_x1 = center_x.saturating_sub(half_size);
    let crop_y1 = center_y.saturating_sub(half_size);
    let crop_x2 = (center_x + half_size).min(img_width);
    let crop_y2 = (center_y + half_size).min(img_height);

    // Adjust if we hit image boundaries to maintain square
    let actual_width = crop_x2 - crop_x1;
    let actual_height = crop_y2 - crop_y1;

    let (final_x1, final_y1, final_x2, final_y2) = if actual_width < actual_height {
        let diff = actual_height - actual_width;
        (
            crop_x1.saturating_sub(diff / 2),
            crop_y1,
            (crop_x2 + diff / 2).min(img_width),
            crop_y2,
        )
    } else if actual_height < actual_width {
        let diff = actual_width - actual_height;
        (
            crop_x1,
            crop_y1.saturating_sub(diff / 2),
            crop_x2,
            (crop_y2 + diff / 2).min(img_height),
        )
    } else {
        (crop_x1, crop_y1, crop_x2, crop_y2)
    };

    // Crop the image
    let cropped = image::imageops::crop_imm(
        img,
        final_x1,
        final_y1,
        final_x2 - final_x1,
        final_y2 - final_y1,
    );

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert to appropriate format based on output file extension
    let cropped_dynamic = DynamicImage::ImageRgba8(cropped.to_image());
    let output_img = if let Some(ext) = output_path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        if ext_lower == "png" {
            // For PNG, preserve alpha channel if present
            cropped_dynamic
        } else {
            // For JPEG and other formats, convert to RGB
            DynamicImage::ImageRgb8(cropped_dynamic.to_rgb8())
        }
    } else {
        // Default to RGB if no extension
        DynamicImage::ImageRgb8(cropped_dynamic.to_rgb8())
    };

    // Save the cropped image
    output_img.save(output_path)?;

    Ok(())
}

pub fn save_bounding_box_image(
    img: &DynamicImage,
    detections: &[Detection],
    output_path: &Path,
) -> Result<()> {
    // Create a copy of the image for drawing bounding boxes
    let mut output_img = img.clone();

    // Determine if we should preserve alpha channel based on output format
    let preserve_alpha = if let Some(ext) = output_path.extension() {
        ext.to_string_lossy().to_lowercase() == "png"
    } else {
        false
    };

    // Convert to appropriate format for drawing
    if preserve_alpha {
        // For PNG output, work with RGBA to preserve transparency
        let mut rgba_img = output_img.to_rgba8();

        // Draw bounding boxes on the image
        for detection in detections {
            let x1 = detection.x1.max(0.0) as u32;
            let y1 = detection.y1.max(0.0) as u32;
            let x2 = detection.x2.min(rgba_img.width() as f32) as u32;
            let y2 = detection.y2.min(rgba_img.height() as f32) as u32;

            // Draw bounding box - bright green with full opacity
            let green = image::Rgba([0, 255, 0, 255]);

            // Draw horizontal lines
            for x in x1..=x2 {
                if x < rgba_img.width() {
                    if y1 < rgba_img.height() {
                        rgba_img.put_pixel(x, y1, green);
                    }
                    if y2 < rgba_img.height() {
                        rgba_img.put_pixel(x, y2, green);
                    }
                }
            }

            // Draw vertical lines
            for y in y1..=y2 {
                if y < rgba_img.height() {
                    if x1 < rgba_img.width() {
                        rgba_img.put_pixel(x1, y, green);
                    }
                    if x2 < rgba_img.width() {
                        rgba_img.put_pixel(x2, y, green);
                    }
                }
            }
        }

        output_img = DynamicImage::ImageRgba8(rgba_img);
    } else {
        // For JPEG output, work with RGB
        let mut rgb_img = output_img.to_rgb8();

        // Draw bounding boxes on the image
        for detection in detections {
            let x1 = detection.x1.max(0.0) as u32;
            let y1 = detection.y1.max(0.0) as u32;
            let x2 = detection.x2.min(rgb_img.width() as f32) as u32;
            let y2 = detection.y2.min(rgb_img.height() as f32) as u32;

            // Draw bounding box - bright green
            let green = image::Rgb([0, 255, 0]);

            // Draw horizontal lines
            for x in x1..=x2 {
                if x < rgb_img.width() {
                    if y1 < rgb_img.height() {
                        rgb_img.put_pixel(x, y1, green);
                    }
                    if y2 < rgb_img.height() {
                        rgb_img.put_pixel(x, y2, green);
                    }
                }
            }

            // Draw vertical lines
            for y in y1..=y2 {
                if y < rgb_img.height() {
                    if x1 < rgb_img.width() {
                        rgb_img.put_pixel(x1, y, green);
                    }
                    if x2 < rgb_img.width() {
                        rgb_img.put_pixel(x2, y, green);
                    }
                }
            }
        }

        output_img = DynamicImage::ImageRgb8(rgb_img);
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the image with bounding boxes
    output_img.save(output_path)?;

    Ok(())
}

/// Handle outputs (crops, bounding boxes, metadata) for a single image
fn handle_image_outputs(
    img: &DynamicImage,
    detections: &[Detection],
    image_path: &Path,
    config: &HeadDetectionConfig,
) -> Result<(Option<String>, Vec<DetectionWithPath>)> {
    let source_path = image_path;
    let output_ext = get_output_extension(source_path);
    let output_manager = OutputManager::new(config, source_path);

    let mut detections_with_paths = Vec::new();

    // Create crops if requested
    if config.crop && !detections.is_empty() {
        for (i, detection) in detections.iter().enumerate() {
            let crop_filename = output_manager.generate_numbered_output(
                "crop",
                i + 1,
                detections.len(),
                output_ext,
            )?;

            create_square_crop(img, detection, &crop_filename, 0.1)?;

            // Make path relative to metadata file if metadata will be created
            let crop_path = output_manager.make_relative_to_metadata(&crop_filename)?;

            detections_with_paths.push(DetectionWithPath {
                detection: detection.clone(),
                crop_path: Some(crop_path),
            });
        }
    } else {
        // No crops, but still need to store detections for metadata
        for detection in detections {
            detections_with_paths.push(DetectionWithPath {
                detection: detection.clone(),
                crop_path: None,
            });
        }
    }

    // Create bounding box image if requested
    let mut bounding_box_path = None;
    if config.bounding_box && !detections.is_empty() {
        let bbox_filename = output_manager.generate_auxiliary_output("bounding-box", output_ext)?;
        save_bounding_box_image(img, detections, &bbox_filename)?;

        // Make path relative to metadata file if metadata will be created
        bounding_box_path = Some(output_manager.make_relative_to_metadata(&bbox_filename)?);
    }

    Ok((bounding_box_path, detections_with_paths))
}

pub fn run_head_detection(config: HeadDetectionConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<HeadProcessor>(config)
}

// Implementation of ModelProcessor trait for head detection
/// Implementation of ModelResult for HeadDetectionResult
pub struct HeadDetectionResult {
    pub detections: Vec<Detection>,
    pub processing_time_ms: f64,
    pub bounding_box_path: Option<String>,
    pub detections_with_paths: Vec<DetectionWithPath>,
}

impl ModelResult for HeadDetectionResult {
    fn result_summary(&self) -> String {
        format!("Found {} bird head(s)", self.detections.len())
    }

    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "head"
    }

    fn core_results(&self) -> Result<toml::Value> {
        let head_core_result = HeadCoreResult {
            model_version: MODEL_VERSION.to_string(),
            bounding_box_path: self.bounding_box_path.clone(),
            detections: self.detections_with_paths.clone(),
        };

        Ok(toml::Value::try_from(head_core_result)?)
    }

    fn output_summary(&self) -> String {
        let mut outputs = Vec::new();

        // Count crops
        let crop_count = self
            .detections_with_paths
            .iter()
            .filter(|d| d.crop_path.is_some())
            .count();
        if crop_count > 0 {
            outputs.push(format!("{crop_count} crop(s)"));
        }

        // Add bounding box if present
        if self.bounding_box_path.is_some() {
            outputs.push("bounding box".to_string());
        }

        if outputs.is_empty() {
            "metadata only".to_string()
        } else {
            format!("→ {}", outputs.join(" + "))
        }
    }
}

/// Head detection processor implementing the generic ModelProcessor trait
pub struct HeadProcessor;

impl ModelProcessor for HeadProcessor {
    type Config = HeadDetectionConfig;
    type Result = HeadDetectionResult;

    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        Ok(ModelSource::EmbeddedBytes(MODEL_BYTES))
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        let processing_start = Instant::now();

        // Load the image
        let img = image::open(image_path)?;
        let (orig_width, orig_height) = img.dimensions();

        debug!(
            "📷 Processing {}: {}x{}",
            image_path.display(),
            orig_width,
            orig_height
        );

        // Preprocess the image
        let model_size = 640; // Standard YOLO input size
        let input_tensor = preprocess_image(&img, model_size)?;

        // Run inference using ORT v2 API with timing
        let inference_start = Instant::now();
        let input_value = Value::from_array(input_tensor)
            .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;
        let outputs = session
            .run(ort::inputs!["images" => &input_value])
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;
        let inference_time = inference_start.elapsed();

        // Extract the output tensor using ORT v2 API and convert to owned array
        let output_view = outputs["output0"]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;
        let output_array =
            Array::from_shape_vec(output_view.shape(), output_view.iter().cloned().collect())?;

        // Postprocess to get detections
        let detections = postprocess_output(
            &output_array,
            config.confidence,
            config.iou_threshold,
            orig_width,
            orig_height,
            model_size,
        )?;

        debug!(
            "⚡ Inference completed in {:.3}ms",
            inference_time.as_secs_f64() * 1000.0
        );
        info!(
            "🎯 Found {} detection(s) after confidence filtering (>{}) and NMS (IoU>{})",
            detections.len(),
            config.confidence,
            config.iou_threshold
        );

        let total_processing_time = processing_start.elapsed().as_secs_f64() * 1000.0;

        // Handle outputs for this specific image
        let (bounding_box_path, detections_with_paths) =
            handle_image_outputs(&img, &detections, image_path, config)?;

        Ok(HeadDetectionResult {
            detections,
            processing_time_ms: total_processing_time,
            bounding_box_path,
            detections_with_paths,
        })
    }

    fn serialize_config(config: &Self::Config) -> Result<toml::Value> {
        Ok(toml::Value::try_from(config)?)
    }
}
