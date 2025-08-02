use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgb};
use ndarray::Array;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;

use crate::yolo_postprocessing::{postprocess_output, Detection};
use crate::yolo_preprocessing::preprocess_image;

// Embed the ONNX model at compile time
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bird-head-detector.onnx"));

// Get model version from build script
pub const MODEL_VERSION: &str =
    include_str!(concat!(env!("OUT_DIR"), "/bird-head-detector.version"));

#[derive(Serialize, Clone)]
pub struct HeadDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
}

#[derive(Serialize)]
pub struct HeadSection {
    pub model_version: String,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box_path: Option<String>,
    pub detections: Vec<HeadDetection>,
}

#[derive(Serialize)]
pub struct BeakerOutput {
    pub head: HeadSection,
}

impl Detection {
    pub fn to_head_detection(&self) -> HeadDetection {
        HeadDetection {
            x1: self.x1,
            y1: self.y1,
            x2: self.x2,
            y2: self.y2,
            confidence: self.confidence,
            crop_path: None,
        }
    }
}

#[derive(Debug)]
pub struct HeadDetectionConfig<'a> {
    pub source: &'a str,
    pub confidence: f32,
    pub iou_threshold: f32,
    pub device: &'a str,
    pub output_dir: Option<&'a str>,
    pub crop: bool,
    pub bounding_box: bool,
    pub skip_toml: bool,
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
    let rgb_img = img.to_rgb8();
    let (img_width, img_height) = rgb_img.dimensions();

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
        &rgb_img,
        final_x1,
        final_y1,
        final_x2 - final_x1,
        final_y2 - final_y1,
    );

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the cropped image
    cropped.to_image().save(output_path)?;

    Ok(())
}

pub fn save_bounding_box_image(
    img: &DynamicImage,
    detections: &[Detection],
    output_path: &Path,
) -> Result<()> {
    let mut rgb_img = img.to_rgb8();

    // Draw bounding boxes on the image
    for detection in detections {
        let x1 = detection.x1.max(0.0) as u32;
        let y1 = detection.y1.max(0.0) as u32;
        let x2 = detection.x2.min(rgb_img.width() as f32) as u32;
        let y2 = detection.y2.min(rgb_img.height() as f32) as u32;

        // Draw bounding box (simple line drawing)
        let green = Rgb([0, 255, 0]);

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

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the image with bounding boxes
    rgb_img.save(output_path)?;

    Ok(())
}

pub fn run_head_detection(config: HeadDetectionConfig) -> Result<usize> {
    // Load the image
    let img = image::open(config.source)?;
    let (orig_width, orig_height) = img.dimensions();

    println!("📷 Loaded image: {orig_width}x{orig_height}");

    // Initialize ONNX Runtime
    let environment = Arc::new(Environment::builder().with_name("beaker").build()?);

    // Determine execution provider based on device
    let execution_providers = match config.device {
        "cpu" => vec![ExecutionProvider::CPU(Default::default())],
        "auto" => {
            #[cfg(target_os = "macos")]
            {
                vec![
                    ExecutionProvider::CoreML(Default::default()),
                    ExecutionProvider::CPU(Default::default()),
                ]
            }
            #[cfg(not(target_os = "macos"))]
            {
                vec![ExecutionProvider::CPU(Default::default())]
            }
        }
        _ => vec![ExecutionProvider::CPU(Default::default())],
    };

    // Load the embedded model
    let session = SessionBuilder::new(&environment)?
        .with_execution_providers(execution_providers)?
        .with_model_from_memory(MODEL_BYTES)?;

    println!(
        "🤖 Loaded embedded ONNX model ({} bytes)",
        MODEL_BYTES.len()
    );

    // Preprocess the image
    let model_size = 640; // Standard YOLO input size
    let input_tensor = preprocess_image(&img, model_size)?;

    println!("🔄 Preprocessed image to {model_size}x{model_size}");

    // Create input for ONNX Runtime
    let input_view = input_tensor.view();
    let input_cow = ndarray::CowArray::from(input_view);
    let input_value = Value::from_array(session.allocator(), &input_cow)?;

    // Run inference
    let outputs = session.run(vec![input_value])?;

    println!("⚡ Inference completed");

    // Extract output tensor
    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_view = output_tensor.view();

    // Convert to ndarray for easier manipulation
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

    let detection_count = detections.len();
    let confidence_threshold = config.confidence;
    let iou_threshold = config.iou_threshold;
    println!(
        "🎯 Found {detection_count} detections after confidence filtering (>{confidence_threshold}) and NMS (IoU>{iou_threshold})"
    );

    // Print detection details
    for (i, detection) in detections.iter().enumerate() {
        let detection_num = i + 1;
        let x1 = detection.x1;
        let y1 = detection.y1;
        let x2 = detection.x2;
        let y2 = detection.y2;
        let confidence = detection.confidence;
        println!("  Detection {detection_num}: bbox=({x1:.1}, {y1:.1}, {x2:.1}, {y2:.1}), confidence={confidence:.3}");
    }

    // Process detections for output generation
    let mut crops_created = 0;
    let mut bbox_images_created = 0;
    let mut bounding_box_path: Option<String> = None;

    // Sort detections by confidence (highest first) and convert to HeadDetection
    let mut sorted_detections = detections.clone();
    sorted_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut head_detections: Vec<HeadDetection> = sorted_detections
        .iter()
        .map(|d| d.to_head_detection())
        .collect();

    if !detections.is_empty() {
        let source_path = Path::new(config.source);
        let input_stem = source_path.file_stem().unwrap().to_str().unwrap();
        let input_ext = source_path
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap();

        // Determine the TOML file path early so we can make paths relative to it
        let toml_filename = if !config.skip_toml {
            Some(if let Some(output_dir) = config.output_dir {
                let output_dir = Path::new(output_dir);
                output_dir.join(format!("{input_stem}-beaker.toml"))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-beaker.toml"))
            })
        } else {
            None
        };

        // Create crops if requested (one per detection)
        if config.crop {
            // Determine if we need zero-padding for filenames
            let use_padding = sorted_detections.len() >= 10;
            let use_suffix = sorted_detections.len() > 1;

            for (i, detection) in sorted_detections.iter().enumerate() {
                let crop_filename = if let Some(output_dir) = config.output_dir {
                    let output_dir = Path::new(output_dir);
                    if use_suffix {
                        if use_padding {
                            output_dir.join(format!("{input_stem}-crop-{:02}.{input_ext}", i + 1))
                        } else {
                            output_dir.join(format!("{input_stem}-crop-{}.{input_ext}", i + 1))
                        }
                    } else {
                        output_dir.join(format!("{input_stem}-crop.{input_ext}"))
                    }
                } else if use_suffix {
                    if use_padding {
                        source_path
                            .parent()
                            .unwrap()
                            .join(format!("{input_stem}-crop-{:02}.{input_ext}", i + 1))
                    } else {
                        source_path
                            .parent()
                            .unwrap()
                            .join(format!("{input_stem}-crop-{}.{input_ext}", i + 1))
                    }
                } else {
                    source_path
                        .parent()
                        .unwrap()
                        .join(format!("{input_stem}-crop.{input_ext}"))
                };

                match create_square_crop(&img, detection, &crop_filename, 0.25) {
                    Ok(()) => {
                        crops_created += 1;
                        if sorted_detections.len() == 1 {
                            println!("✂️  Created crop: {}", crop_filename.display());
                        } else {
                            println!(
                                "✂️  Created crop {}: {} (confidence: {:.3})",
                                i + 1,
                                crop_filename.display(),
                                detection.confidence
                            );
                        }

                        // Update the corresponding detection with crop path
                        // Make path relative to TOML file if TOML will be created
                        let crop_path = if let Some(ref toml_path) = toml_filename {
                            make_path_relative_to_toml(&crop_filename, toml_path)?
                        } else {
                            crop_filename.to_string_lossy().to_string()
                        };
                        head_detections[i].crop_path = Some(crop_path);
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to create crop {}: {e}", i + 1);
                    }
                }
            }
        }

        // Save bounding box image if requested
        if config.bounding_box {
            let bbox_filename = if let Some(output_dir) = config.output_dir {
                let output_dir = Path::new(output_dir);
                output_dir.join(format!("{input_stem}-bounding-box.{input_ext}"))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-bounding-box.{input_ext}"))
            };

            // Pass the original unsorted detections to maintain all detections in the image
            match save_bounding_box_image(&img, &detections, &bbox_filename) {
                Ok(()) => {
                    bbox_images_created += 1;
                    if detections.len() == 1 {
                        println!("📦 Created bounding box image: {}", bbox_filename.display());
                    } else {
                        println!(
                            "📦 Created bounding box image: {} ({} detections)",
                            bbox_filename.display(),
                            detections.len()
                        );
                    }

                    // Set the bounding box path at the top level (includes all detections)
                    // Make path relative to TOML file if TOML will be created
                    bounding_box_path = Some(if let Some(ref toml_path) = toml_filename {
                        make_path_relative_to_toml(&bbox_filename, toml_path)?
                    } else {
                        bbox_filename.to_string_lossy().to_string()
                    });
                }
                Err(e) => {
                    eprintln!("❌ Failed to save bounding box image: {e}");
                }
            }
        }
    }

    // Create TOML output unless skipped
    if !config.skip_toml {
        let source_path = Path::new(config.source);
        let input_stem = source_path.file_stem().unwrap().to_str().unwrap();

        let toml_filename = if let Some(output_dir) = config.output_dir {
            let output_dir = Path::new(output_dir);
            output_dir.join(format!("{input_stem}-beaker.toml"))
        } else {
            source_path
                .parent()
                .unwrap()
                .join(format!("{input_stem}-beaker.toml"))
        };

        let beaker_output = BeakerOutput {
            head: HeadSection {
                model_version: MODEL_VERSION.trim().to_string(),
                confidence_threshold: config.confidence,
                iou_threshold: config.iou_threshold,
                bounding_box_path,
                detections: head_detections,
            },
        };

        match toml::to_string_pretty(&beaker_output) {
            Ok(toml_content) => {
                // Create output directory if it doesn't exist
                if let Some(parent) = toml_filename.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                match std::fs::write(&toml_filename, toml_content) {
                    Ok(()) => {
                        println!("📝 Created TOML output: {}", toml_filename.display());
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to write TOML file: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to serialize TOML: {e}");
            }
        }
    }

    // Print summary
    if config.crop && crops_created > 0 {
        if let Some(output_dir) = config.output_dir {
            println!("✂️  Created {crops_created} head crop in: {output_dir}");
        } else {
            println!("✂️  Created {crops_created} head crop next to original image");
        }
    }

    if config.bounding_box && bbox_images_created > 0 {
        if let Some(output_dir) = config.output_dir {
            println!("📦 Created {bbox_images_created} bounding box image in: {output_dir}");
        } else {
            println!("📦 Created {bbox_images_created} bounding box image next to original image");
        }
    }

    Ok(detections.len())
}
