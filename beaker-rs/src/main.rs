use anyhow::Result;
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, Rgb};
use ndarray::Array;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;

// Embed the ONNX model at compile time
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bird-head-detector.onnx"));

#[derive(Parser)]
#[command(name = "beaker")]
#[command(about = "Detect bird heads in images using YOLOv8")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input image path (when not using subcommands)
    #[arg(short, long)]
    image: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run bird head detection on an image
    Detect {
        /// Input image or directory path
        source: String,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<String>,

        /// Confidence threshold
        #[arg(short, long, default_value = "0.25")]
        confidence: f32,

        /// IoU threshold for NMS
        #[arg(long, default_value = "0.45")]
        iou_threshold: f32,

        /// Device to use for inference
        #[arg(short, long, default_value = "auto")]
        device: String,

        /// Skip creating square crops around detected heads
        #[arg(long)]
        skip_crop: bool,

        /// Save detection results with bounding boxes
        #[arg(long)]
        save_bounding_box: bool,
    },
    /// Run benchmark tests
    Benchmark,
    /// Show version information
    Version,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Detect {
            source,
            output,
            confidence,
            iou_threshold,
            device,
            skip_crop,
            save_bounding_box,
        }) => {
            println!("🔍 Would detect bird heads in: {source}");
            println!("   Model: embedded ONNX model");
            println!("   Confidence threshold: {confidence}");
            println!("   IoU threshold: {iou_threshold}");
            println!("   Device: {device}");
            if *skip_crop {
                println!("   Skipping crop generation");
            }
            if *save_bounding_box {
                println!("   Will save bounding box images");
            }
            if let Some(output) = output {
                println!("   Output directory: {output}");
            }

            // Run actual detection
            let config = DetectionConfig {
                source,
                confidence: *confidence,
                iou_threshold: *iou_threshold,
                device,
                output_dir: output.as_deref(),
                skip_crop: *skip_crop,
                save_bounding_box: *save_bounding_box,
            };
            match run_detection(config) {
                Ok(detections) => {
                    println!("✅ Found {detections} detections");
                }
                Err(e) => {
                    eprintln!("❌ Detection failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Benchmark) => {
            println!("🏃 Would run benchmark tests");
            println!("   [Not implemented yet - this is a skeleton]");
        }
        Some(Commands::Version) => {
            println!("beaker {}", env!("CARGO_PKG_VERSION"));
            println!("Rust implementation of YOLOv8 bird head detection");
        }
        None => {
            if let Some(image) = &cli.image {
                println!("🔍 Would detect bird heads in: {image}");
                println!("   [Not implemented yet - this is a skeleton]");
                println!("   Hint: Use 'beaker detect {image}' for more options");
            } else {
                // Show help if no command or image specified
                use clap::CommandFactory;
                let mut cmd = Cli::command();
                cmd.print_help().unwrap();
            }
        }
    }
}

fn preprocess_image(img: &DynamicImage, target_size: u32) -> Result<Array<f32, ndarray::IxDyn>> {
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

#[derive(Debug, Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
}

impl Detection {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersection_area(&self, other: &Detection) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        if x2 > x1 && y2 > y1 {
            (x2 - x1) * (y2 - y1)
        } else {
            0.0
        }
    }

    fn iou(&self, other: &Detection) -> f32 {
        let intersection = self.intersection_area(other);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    // Sort by confidence score in descending order
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i].clone());

        // Suppress overlapping detections
        for j in (i + 1)..detections.len() {
            if !suppressed[j] && detections[i].iou(&detections[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn create_square_crop(
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

fn save_bounding_box_image(
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

fn postprocess_output(
    output: &Array<f32, ndarray::IxDyn>,
    confidence_threshold: f32,
    iou_threshold: f32,
    img_width: u32,
    img_height: u32,
    model_size: u32,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    // Output shape should be [1, num_classes + 4, num_boxes]
    let shape = output.shape();
    if shape.len() != 3 {
        return Err(anyhow::anyhow!("Expected 3D output, got {}D", shape.len()));
    }
    let num_boxes = shape[2];

    for i in 0..num_boxes {
        // Get box coordinates (first 4 values)
        let x_center = output[[0, 0, i]];
        let y_center = output[[0, 1, i]];
        let width = output[[0, 2, i]];
        let height = output[[0, 3, i]];

        // Get confidence (5th value, index 4)
        let confidence = output[[0, 4, i]];

        if confidence > confidence_threshold {
            // Convert from center coordinates to corner coordinates
            let x1 = x_center - width / 2.0;
            let y1 = y_center - height / 2.0;
            let x2 = x_center + width / 2.0;
            let y2 = y_center + height / 2.0;

            // Scale coordinates back to original image size
            let scale_x = img_width as f32 / model_size as f32;
            let scale_y = img_height as f32 / model_size as f32;

            detections.push(Detection {
                x1: x1 * scale_x,
                y1: y1 * scale_y,
                x2: x2 * scale_x,
                y2: y2 * scale_y,
                confidence,
            });
        }
    }

    // Apply Non-Maximum Suppression
    let nms_detections = nms(detections, iou_threshold);

    Ok(nms_detections)
}

#[derive(Debug)]
struct DetectionConfig<'a> {
    source: &'a str,
    confidence: f32,
    iou_threshold: f32,
    device: &'a str,
    output_dir: Option<&'a str>,
    skip_crop: bool,
    save_bounding_box: bool,
}

fn run_detection(config: DetectionConfig) -> Result<usize> {
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

    if !detections.is_empty() {
        let source_path = Path::new(config.source);
        let input_stem = source_path.file_stem().unwrap().to_str().unwrap();
        let input_ext = source_path
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap();

        // Get the highest confidence detection for cropping
        let best_detection = detections
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        // Create crop if not skipped
        if !config.skip_crop {
            let crop_filename = if let Some(output_dir) = config.output_dir {
                let output_dir = Path::new(output_dir);
                output_dir.join(format!("{input_stem}.{input_ext}"))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-crop.{input_ext}"))
            };

            match create_square_crop(&img, best_detection, &crop_filename, 0.25) {
                Ok(()) => {
                    crops_created += 1;
                    if detections.len() == 1 {
                        println!("✂️  Created crop: {}", crop_filename.display());
                    } else {
                        println!(
                            "✂️  Created crop: {} (used highest confidence: {:.3}, {} total detections)",
                            crop_filename.display(),
                            best_detection.confidence,
                            detections.len()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to create crop: {e}");
                }
            }
        }

        // Save bounding box image if requested
        if config.save_bounding_box {
            let bbox_filename = if let Some(output_dir) = config.output_dir {
                let output_dir = Path::new(output_dir);
                output_dir.join(format!("{input_stem}.{input_ext}"))
            } else {
                source_path
                    .parent()
                    .unwrap()
                    .join(format!("{input_stem}-bounding-box.{input_ext}"))
            };

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
                }
                Err(e) => {
                    eprintln!("❌ Failed to save bounding box image: {e}");
                }
            }
        }
    } else {
        // No detections found
        if !config.skip_crop || config.save_bounding_box {
            println!("ℹ️  No detections found, no outputs created");
        }
    }

    // Print summary
    if !config.skip_crop && crops_created > 0 {
        if let Some(output_dir) = config.output_dir {
            println!("✂️  Created {crops_created} square head crop in: {output_dir}");
        } else {
            println!("✂️  Created {crops_created} square head crop next to original image");
        }
    }

    if config.save_bounding_box && bbox_images_created > 0 {
        if let Some(output_dir) = config.output_dir {
            println!("� Created {bbox_images_created} bounding box image in: {output_dir}");
        } else {
            println!("📦 Created {bbox_images_created} bounding box image next to original image");
        }
    }

    Ok(detections.len())
}
