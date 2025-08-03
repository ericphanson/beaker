use anyhow::Result;
use chrono::{DateTime, Utc};
use image::{DynamicImage, GenericImageView};
use ndarray::Array;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

use crate::config::HeadDetectionConfig;
use crate::image_input::{collect_images_from_sources, ImageInputConfig};
use crate::onnx_session::{
    create_onnx_session, determine_optimal_device, ModelSource, SessionConfig,
};
use crate::shared_metadata::{get_metadata_path, load_or_create_metadata, save_metadata};
use crate::yolo_postprocessing::{postprocess_output, Detection};
use crate::yolo_preprocessing::preprocess_image;
use log::{debug, info};

// Embed the ONNX model at compile time
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bird-head-detector.onnx"));

// Get model version from build script
pub const MODEL_VERSION: &str =
    include_str!(concat!(env!("OUT_DIR"), "/bird-head-detector.version"));

#[derive(Serialize)]
pub struct HeadResult {
    pub timestamp: DateTime<Utc>,
    pub model_version: String,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box_path: Option<String>,
    pub detections: Vec<DetectionWithPath>,
}

#[derive(Serialize)]
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

pub fn make_path_relative_to_toml(file_path: &Path, toml_path: &Path) -> Result<String> {
    // Get the directory containing the TOML file
    let toml_dir = toml_path.parent().unwrap_or(Path::new("."));

    // Try to create a relative path from the TOML directory to the file
    match file_path.strip_prefix(toml_dir) {
        Ok(relative_path) => Ok(relative_path.to_string_lossy().to_string()),
        Err(_) => {
            // If they're not in a parent-child relationship, use absolute path
            Ok(file_path.to_string_lossy().to_string())
        }
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

/// Process a single image with an existing session
fn process_single_image(
    session: &mut Session,
    image_path: &Path,
    config: &HeadDetectionConfig,
) -> Result<(usize, f64)> {
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
        "🎯 Found {} detections after confidence filtering (>{}) and NMS (IoU>{})",
        detections.len(),
        config.confidence,
        config.iou_threshold
    );

    // Display detections
    for (i, detection) in detections.iter().enumerate() {
        debug!(
            "  Detection {}: bbox=({:.1}, {:.1}, {:.1}, {:.1}), confidence={:.3}",
            i + 1,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.confidence
        );
    }

    let total_processing_time = processing_start.elapsed().as_secs_f64() * 1000.0;

    // Handle outputs for this specific image
    handle_image_outputs(&img, &detections, image_path, config, total_processing_time)?;

    Ok((detections.len(), total_processing_time))
}

/// Handle outputs (crops, bounding boxes, metadata) for a single image
fn handle_image_outputs(
    img: &DynamicImage,
    detections: &[Detection],
    image_path: &Path,
    config: &HeadDetectionConfig,
    processing_time_ms: f64,
) -> Result<()> {
    let source_path = image_path;
    let input_stem = source_path.file_stem().unwrap().to_str().unwrap();
    let output_ext = get_output_extension(source_path);

    // Determine the metadata file path early so we can make paths relative to it
    let metadata_path = if !config.base.skip_metadata {
        Some(get_metadata_path(
            source_path,
            config.base.output_dir.as_deref(),
        )?)
    } else {
        None
    };

    let mut detections_with_paths = Vec::new();

    // Create crops if requested
    if config.crop && !detections.is_empty() {
        for (i, detection) in detections.iter().enumerate() {
            let crop_filename = if detections.len() == 1 {
                if let Some(output_dir) = &config.base.output_dir {
                    let output_dir = Path::new(output_dir);
                    std::fs::create_dir_all(output_dir)?;
                    output_dir.join(format!("{input_stem}.{output_ext}"))
                } else {
                    source_path
                        .parent()
                        .unwrap()
                        .join(format!("{input_stem}_crop.{output_ext}"))
                }
            } else if let Some(output_dir) = &config.base.output_dir {
                let output_dir = Path::new(output_dir);
                std::fs::create_dir_all(output_dir)?;
                if detections.len() >= 10 {
                    output_dir.join(format!("{input_stem}-{:02}.{output_ext}", i + 1))
                } else {
                    output_dir.join(format!("{input_stem}-{}.{output_ext}", i + 1))
                }
            } else if detections.len() >= 10 {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}_crop-{:02}.{output_ext}", i + 1))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}_crop-{}.{output_ext}", i + 1))
            };

            create_square_crop(img, detection, &crop_filename, 0.1)?;

            // Make path relative to metadata file if metadata will be created
            let crop_path = if let Some(ref toml_path) = metadata_path {
                make_path_relative_to_toml(&crop_filename, toml_path)?
            } else {
                crop_filename.to_string_lossy().to_string()
            };

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
        let bbox_filename = if let Some(output_dir) = &config.base.output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(format!("{input_stem}_bounding-box.{output_ext}"))
        } else {
            source_path
                .parent()
                .unwrap()
                .join(format!("{input_stem}_bounding-box.{output_ext}"))
        };

        save_bounding_box_image(img, detections, &bbox_filename)?;

        // Make path relative to metadata file if metadata will be created
        bounding_box_path = Some(if let Some(ref toml_path) = metadata_path {
            make_path_relative_to_toml(&bbox_filename, toml_path)?
        } else {
            bbox_filename.to_string_lossy().to_string()
        });
    }

    // Create metadata output unless skipped
    if !config.base.skip_metadata {
        let source_path = image_path;

        // Save using shared metadata system
        let metadata_path = get_metadata_path(source_path, config.base.output_dir.as_deref())?;
        let mut metadata = load_or_create_metadata(&metadata_path)?;

        let head_result = HeadResult {
            timestamp: Utc::now(),
            model_version: MODEL_VERSION.trim().to_string(),
            confidence_threshold: config.confidence,
            iou_threshold: config.iou_threshold,
            processing_time_ms,
            bounding_box_path,
            detections: detections_with_paths,
        };

        metadata.head = Some(serde_json::to_value(head_result)?);
        save_metadata(&metadata, &metadata_path)?;

        debug!("📝 Created TOML output: {}", metadata_path.display());
    }

    Ok(())
}

pub fn run_head_detection(config: HeadDetectionConfig) -> Result<usize> {
    // Collect all image files from the provided sources using the strict flag
    let image_config = ImageInputConfig::from_strict_flag(config.base.strict);
    let image_files = collect_images_from_sources(&config.base.sources, &image_config)?;

    if image_files.is_empty() {
        info!("🎯 No images found to process");
        return Ok(0);
    }

    // Log what we're processing
    if image_files.len() == 1 {
        log::info!("🔍 Processing single image: {}", image_files[0].display());
    } else {
        log::info!(
            "📁 Processing {} images from {} source{}",
            image_files.len(),
            config.base.sources.len(),
            if config.base.sources.len() == 1 {
                ""
            } else {
                "s"
            }
        );
    }

    // Determine optimal device based on number of images
    let device_selection = determine_optimal_device(&config.base.device, image_files.len());

    // Create session using unified ONNX session management
    let session_config = SessionConfig {
        device: &device_selection.device,
    };

    let (mut session, load_time) =
        create_onnx_session(ModelSource::EmbeddedBytes(MODEL_BYTES), &session_config)?;

    // Process all images with the same session
    let total_start = Instant::now();
    let mut total_detections = 0;

    for (i, image_path) in image_files.iter().enumerate() {
        if image_files.len() > 1 {
            log::debug!(
                "📷 Processing image {}/{}: {}",
                i + 1,
                image_files.len(),
                image_path.display()
            );
        }

        // Create individual config for each image
        let mut image_config = config.clone();
        image_config.base.sources = vec![image_path.to_string_lossy().to_string()];

        match process_single_image(&mut session, image_path, &image_config) {
            Ok((detections, _processing_time)) => {
                total_detections += detections;
            }
            Err(e) => {
                log::error!("❌ Error processing {}: {}", image_path.display(), e);
                continue;
            }
        }
    }

    let total_time = total_start.elapsed();

    // Print appropriate summary
    if image_files.len() == 1 {
        log::info!("🎉 Processing complete!");
        log::info!("🎯 Found {total_detections} bird head(s)");
        log::info!(
            "⚡ Total time: {:.1}ms (load: {:.1}ms, inference: {:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            load_time,
            (total_time.as_secs_f64() * 1000.0) - load_time
        );
    } else {
        let avg_time_per_image = total_time.as_secs_f64() / image_files.len() as f64;

        log::info!("🎉 Batch processing complete!");
        log::info!("📊 Summary:");
        log::info!("  • Images processed: {}", image_files.len());
        log::info!("  • Total detections: {total_detections}");
        log::info!("  • Model load time: {load_time:.1}ms");
        log::info!(
            "  • Total processing time: {:.3}s",
            total_time.as_secs_f64()
        );
        log::info!(
            "  • Average time per image: {:.1}ms",
            avg_time_per_image * 1000.0
        );
    }

    Ok(total_detections)
}
