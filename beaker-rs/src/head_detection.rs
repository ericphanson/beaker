use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgb};
use ndarray::Array;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::Session,
    value::Value,
};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

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
pub struct HeadDetectionOutput {
    pub head: HeadResult,
}

#[derive(Serialize)]
pub struct HeadResult {
    pub model_version: String,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
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

#[derive(Serialize)]
pub struct BeakerOutput {
    pub head: HeadResult,
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

#[derive(Debug, Clone)]
pub struct HeadDetectionConfig {
    pub source: String,
    pub confidence: f32,
    pub iou_threshold: f32,
    pub device: String,
    pub output_dir: Option<String>,
    pub crop: bool,
    pub bounding_box: bool,
    pub skip_metadata: bool,
}

/// Check if a file is a supported image format
fn is_image_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        matches!(
            ext.as_str(),
            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "tif" | "webp"
        )
    } else {
        false
    }
}

/// Find all image files in a directory (non-recursive)
fn find_image_files(dir_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut image_files = Vec::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && is_image_file(&path) {
            image_files.push(path);
        }
    }

    // Sort for consistent ordering
    image_files.sort();
    Ok(image_files)
}

/// Determine optimal device based on number of images and user preference
fn determine_optimal_device(device: &str, num_images: usize) -> String {
    const COREML_THRESHOLD: usize = 3; // Use CoreML for 3+ images

    match device {
        "auto" => {
            if num_images >= COREML_THRESHOLD {
                // Check if CoreML is available
                let coreml = CoreMLExecutionProvider::default();
                match coreml.is_available() {
                    Ok(true) => {
                        println!("📊 Processing {num_images} images - using CoreML for better batch performance");
                        "coreml".to_string()
                    }
                    _ => {
                        println!(
                            "📊 Processing {num_images} images - CoreML not available, using CPU"
                        );
                        "cpu".to_string()
                    }
                }
            } else {
                println!("📊 Processing {num_images} images - using CPU for small batch");
                "cpu".to_string()
            }
        }
        other => other.to_string(), // User explicitly chose a device
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

/// Create an ONNX Runtime session with the specified device
fn create_session(device: &str) -> Result<(Session, f64)> {
    // Determine execution provider based on device with availability checking
    let execution_providers = match device {
        "cpu" => {
            println!("🖥️  Using CPU execution provider");
            vec![CPUExecutionProvider::default().build()]
        }
        "auto" => {
            // Check CoreML availability (works on all platforms but only available on macOS)
            let coreml = CoreMLExecutionProvider::default();
            match coreml.is_available() {
                Ok(true) => {
                    println!("🍎 CoreML execution provider is available");
                    println!("🍎 Using CoreML + CPU execution providers (auto mode)");
                    vec![coreml.build(), CPUExecutionProvider::default().build()]
                }
                Ok(false) => {
                    println!("⚠️  CoreML execution provider not available on this platform");
                    println!("🖥️  Using CPU execution provider (auto mode fallback)");
                    vec![CPUExecutionProvider::default().build()]
                }
                Err(e) => {
                    println!("⚠️  Error checking CoreML availability: {e}");
                    println!("🖥️  Using CPU execution provider (auto mode fallback)");
                    vec![CPUExecutionProvider::default().build()]
                }
            }
        }
        "coreml" => {
            // Explicit CoreML request - check availability and error if not available
            let coreml = CoreMLExecutionProvider::default();
            match coreml.is_available() {
                Ok(true) => {
                    println!("🍎 Using CoreML + CPU execution providers (explicit)");
                    vec![coreml.build(), CPUExecutionProvider::default().build()]
                }
                Ok(false) => {
                    return Err(anyhow::anyhow!(
                        "CoreML execution provider requested but not available on this platform"
                    ));
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Error checking CoreML availability: {}", e));
                }
            }
        }
        _ => {
            println!("🖥️  Using CPU execution provider (fallback)");
            vec![CPUExecutionProvider::default().build()]
        }
    };

    // Store EP info for logging before moving the vector
    let ep_names: Vec<String> = execution_providers
        .iter()
        .enumerate()
        .map(|(i, _)| format!("EP{}", i + 1))
        .collect();

    // Load the embedded model using ORT v2 API
    let session_start = Instant::now();
    let session = Session::builder()?
        .with_execution_providers(execution_providers)?
        .commit_from_memory(MODEL_BYTES)?;
    let session_load_time = session_start.elapsed();

    println!(
        "🤖 Loaded embedded ONNX model ({} bytes) in {:.3}ms",
        MODEL_BYTES.len(),
        session_load_time.as_secs_f64() * 1000.0
    );

    // Log execution provider information
    println!(
        "⚙️  Execution providers registered: {}",
        ep_names.join(" -> ")
    );

    Ok((session, session_load_time.as_secs_f64() * 1000.0))
}

/// Process a single image with an existing session
fn process_single_image(
    session: &mut Session,
    image_path: &Path,
    config: &HeadDetectionConfig,
) -> Result<usize> {
    // Load the image
    let img = image::open(image_path)?;
    let (orig_width, orig_height) = img.dimensions();

    println!(
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
    let input_value = Value::from_array(input_tensor)?;
    let outputs = session.run(ort::inputs!["images" => &input_value])?;
    let inference_time = inference_start.elapsed();

    // Extract the output tensor using ORT v2 API and convert to owned array
    let output_view = outputs["output0"].try_extract_array::<f32>()?;
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

    println!(
        "⚡ Inference completed in {:.3}ms",
        inference_time.as_secs_f64() * 1000.0
    );
    println!(
        "🎯 Found {} detections after confidence filtering (>{}) and NMS (IoU>{})",
        detections.len(),
        config.confidence,
        config.iou_threshold
    );

    // Display detections
    for (i, detection) in detections.iter().enumerate() {
        println!(
            "  Detection {}: bbox=({:.1}, {:.1}, {:.1}, {:.1}), confidence={:.3}",
            i + 1,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.confidence
        );
    }

    // Handle outputs for this specific image
    handle_image_outputs(&img, &detections, image_path, config)?;

    Ok(detections.len())
}

/// Handle outputs (crops, bounding boxes, metadata) for a single image
fn handle_image_outputs(
    img: &DynamicImage,
    detections: &[Detection],
    image_path: &Path,
    config: &HeadDetectionConfig,
) -> Result<()> {
    let source_path = image_path;
    let input_stem = source_path.file_stem().unwrap().to_str().unwrap();

    // Determine the metadata file path early so we can make paths relative to it
    let toml_filename = if !config.skip_metadata {
        Some(if let Some(output_dir) = &config.output_dir {
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

    let mut detections_with_paths = Vec::new();

    // Create crops if requested
    if config.crop && !detections.is_empty() {
        for (i, detection) in detections.iter().enumerate() {
            let crop_filename = if detections.len() == 1 {
                if let Some(output_dir) = &config.output_dir {
                    let output_dir = Path::new(output_dir);
                    std::fs::create_dir_all(output_dir)?;
                    output_dir.join(format!("{input_stem}-crop.jpg"))
                } else {
                    source_path
                        .parent()
                        .unwrap()
                        .join(format!("{input_stem}-crop.jpg"))
                }
            } else if let Some(output_dir) = &config.output_dir {
                let output_dir = Path::new(output_dir);
                std::fs::create_dir_all(output_dir)?;
                if detections.len() >= 10 {
                    output_dir.join(format!("{input_stem}-crop-{:02}.jpg", i + 1))
                } else {
                    output_dir.join(format!("{input_stem}-crop-{}.jpg", i + 1))
                }
            } else if detections.len() >= 10 {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-crop-{:02}.jpg", i + 1))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-crop-{}.jpg", i + 1))
            };

            create_square_crop(img, detection, &crop_filename, 0.1)?;

            // Make path relative to metadata file if metadata will be created
            let crop_path = if let Some(ref toml_path) = toml_filename {
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
        let bbox_filename = if let Some(output_dir) = &config.output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(format!("{input_stem}-bounding-box.jpg"))
        } else {
            source_path
                .parent()
                .unwrap()
                .join(format!("{input_stem}-bounding-box.jpg"))
        };

        save_bounding_box_image(img, detections, &bbox_filename)?;

        // Make path relative to metadata file if metadata will be created
        bounding_box_path = Some(if let Some(ref toml_path) = toml_filename {
            make_path_relative_to_toml(&bbox_filename, toml_path)?
        } else {
            bbox_filename.to_string_lossy().to_string()
        });
    }

    // Create metadata output unless skipped
    if !config.skip_metadata {
        let source_path = Path::new(&config.source);
        let input_stem = source_path.file_stem().unwrap().to_str().unwrap();

        let toml_filename = if let Some(output_dir) = &config.output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(format!("{input_stem}-beaker.toml"))
        } else {
            source_path
                .parent()
                .unwrap()
                .join(format!("{input_stem}-beaker.toml"))
        };

        let output = HeadDetectionOutput {
            head: HeadResult {
                model_version: MODEL_VERSION.trim().to_string(),
                confidence_threshold: config.confidence,
                iou_threshold: config.iou_threshold,
                bounding_box_path,
                detections: detections_with_paths,
            },
        };

        let toml_content = toml::to_string_pretty(&output)?;
        std::fs::write(&toml_filename, toml_content)?;

        println!("📝 Created TOML output: {}", toml_filename.display());
    }

    Ok(())
}

pub fn run_head_detection(config: HeadDetectionConfig) -> Result<usize> {
    let source_path = Path::new(&config.source);

    // Get list of images to process (either single file or directory)
    let image_files = if source_path.is_file() {
        if is_image_file(source_path) {
            vec![source_path.to_path_buf()]
        } else {
            return Err(anyhow::anyhow!(
                "File is not a supported image format: {}",
                source_path.display()
            ));
        }
    } else if source_path.is_dir() {
        find_image_files(source_path)?
    } else {
        return Err(anyhow::anyhow!(
            "Source path does not exist or is not a file/directory: {}",
            source_path.display()
        ));
    };

    if image_files.is_empty() {
        println!("⚠️  No image files found in: {}", source_path.display());
        return Ok(0);
    }

    // Log what we're processing
    if image_files.len() == 1 {
        println!("� Processing single image: {}", image_files[0].display());
    } else {
        println!(
            "📁 Processing {} images from directory: {}",
            image_files.len(),
            source_path.display()
        );
    }

    // Determine optimal device based on number of images
    let optimal_device = determine_optimal_device(&config.device, image_files.len());
    let (mut session, load_time) = create_session(&optimal_device)?;

    // Process all images with the same session
    let total_start = Instant::now();
    let mut total_detections = 0;

    for (i, image_path) in image_files.iter().enumerate() {
        if image_files.len() > 1 {
            println!(
                "\n📷 Processing image {}/{}: {}",
                i + 1,
                image_files.len(),
                image_path.display()
            );
        }

        // Create individual config for each image
        let mut image_config = config.clone();
        image_config.source = image_path.to_string_lossy().to_string();

        match process_single_image(&mut session, image_path, &image_config) {
            Ok(detections) => {
                total_detections += detections;
            }
            Err(e) => {
                println!("❌ Error processing {}: {}", image_path.display(), e);
                continue;
            }
        }
    }

    let total_time = total_start.elapsed();

    // Print appropriate summary
    if image_files.len() == 1 {
        println!("\n🎉 Processing complete!");
        println!("🎯 Found {total_detections} bird head(s)");
        println!(
            "⚡ Total time: {:.1}ms (load: {:.1}ms, inference: {:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            load_time,
            (total_time.as_secs_f64() * 1000.0) - load_time
        );
    } else {
        let avg_time_per_image = total_time.as_secs_f64() / image_files.len() as f64;

        println!("\n🎉 Batch processing complete!");
        println!("📊 Summary:");
        println!("  • Images processed: {}", image_files.len());
        println!("  • Total detections: {total_detections}");
        println!("  • Model load time: {load_time:.1}ms");
        println!(
            "  • Total processing time: {:.3}s",
            total_time.as_secs_f64()
        );
        println!(
            "  • Average time per image: {:.1}ms",
            avg_time_per_image * 1000.0
        );
    }

    Ok(total_detections)
}
