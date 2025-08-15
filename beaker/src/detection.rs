use crate::color_utils::symbols;
use crate::config::DetectionConfig;
use crate::detection_obj::Detection;
use crate::model_access::{ModelAccess, ModelInfo};
use crate::model_processing::{ModelProcessor, ModelResult};
use crate::onnx_session::ModelSource;
use crate::output_manager::OutputManager;
use crate::rfdetr;
use crate::shared_metadata::IoTiming;
use crate::yolo;
use ab_glyph::{FontRef, PxScale};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use imageproc::drawing::{draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut, text_size};
use log::debug;
use ndarray::Array;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

static FONT_BYTES: &[u8] = include_bytes!("../fonts/NotoSans-Regular.ttf");

pub fn get_default_detect_model_info() -> ModelInfo {
    ModelInfo {
        name: "bird-orientation-detector-v1.0.0".to_string(),
        url: "https://github.com/ericphanson/beaker/releases/download/bird-orientation-detector-v1.0.0/rfdetr-medium-dynamic-int8.onnx".to_string(),
        md5_checksum: "f1e20fe90da342a7529c89f4d39b7ff9".to_string(),
        filename: "bird-orientation-detector-v1.0.0.onnx".to_string(),
    }
}

#[derive(Debug)]
enum DetectionModelVariants {
    HeadOnly,
    MultiDetect,
    Orientation,
}

impl DetectionModelVariants {
    /// Determine the model variant from output dimensions
    fn from_outputs(output_dimensions: &[i64], n_outputs: usize) -> Self {
        // Check if legacy model based on output channels
        let is_legacy_model = output_dimensions[1] < 8;

        if is_legacy_model {
            DetectionModelVariants::HeadOnly
        } else if n_outputs == 1 {
            DetectionModelVariants::MultiDetect
        } else {
            DetectionModelVariants::Orientation
        }
    }

    /// Convert to boolean for compatibility with existing postprocessing function
    fn is_legacy_model(&self) -> bool {
        matches!(self, DetectionModelVariants::HeadOnly)
    }
}

/// Head detection model access implementation.
pub struct HeadAccess;

impl ModelAccess for HeadAccess {
    fn get_embedded_bytes() -> Option<&'static [u8]> {
        None
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_DETECT_MODEL_PATH"
    }

    // Head models support remote download for CLI --model-url usage
    // but prefer embedded bytes by default
    fn get_default_model_info() -> Option<ModelInfo> {
        // Only provide this for CLI usage when --model-url is specified
        // The get_model_source_with_cli_and_env_override will use embedded bytes
        // unless CLI args or env vars override it
        Some(get_default_detect_model_info())
    }
}

#[derive(Serialize)]
pub struct DetectionResult {
    pub model_version: String,
    #[serde(skip_serializing)]
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box_path: Option<String>,
    pub detections: Vec<DetectionWithPath>,
    #[serde(skip_serializing)]
    pub io_timing: IoTiming,
    pub input_img_width: u32,
    pub input_img_height: u32,
}

#[derive(Serialize, Clone)]
pub struct DetectionWithPath {
    #[serde(flatten)]
    pub detection: Detection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
}

/// Process multiple images sequentially
pub fn run_detection(config: DetectionConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<DetectionProcessor>(config)
}

impl ModelResult for DetectionResult {
    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "detect"
    }

    fn core_results(&self) -> Result<toml::Value> {
        Ok(toml::Value::try_from(self)?)
    }

    fn output_summary(&self) -> String {
        let mut outputs = Vec::new();

        // Count crops
        let crop_count = self
            .detections
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
            "".to_string()
        } else {
            format!("→ {}", outputs.join(" + "))
        }
    }

    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming {
        self.io_timing.clone()
    }
}

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

/// Dispatch preprocessing based on model variant
fn preprocess_image_for_model(
    model_variant: &DetectionModelVariants,
    img: &DynamicImage,
    model_size: u32,
) -> Result<Array<f32, ndarray::IxDyn>> {
    match model_variant {
        DetectionModelVariants::HeadOnly | DetectionModelVariants::MultiDetect => {
            // YOLO models use letterboxing with gray padding
            yolo::preprocess_image(img, model_size)
        }
        DetectionModelVariants::Orientation => {
            // RF-DETR models use square resize with ImageNet normalization
            rfdetr::preprocess_image(img, model_size)
        }
    }
}

/// Handle outputs (crops, bounding boxes, metadata) for a single image with I/O timing
fn handle_image_outputs_with_timing(
    img: &DynamicImage,
    detections: &[Detection],
    image_path: &Path,
    config: &DetectionConfig,
    io_timing: &mut IoTiming,
    output_manager: &OutputManager,
) -> Result<(Option<String>, Vec<DetectionWithPath>)> {
    let source_path = image_path;
    let output_ext = get_output_extension(source_path);

    let mut detections_with_paths = Vec::new();

    let total_per_class =
        detections
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, detection| {
                *acc.entry(detection.class_name.clone()).or_insert(0) += 1;
                acc
            });
    // Create crops if requested
    if !config.crop_classes.is_empty() && !detections.is_empty() {
        for (i, detection) in detections.iter().enumerate() {
            // Check if this detection's class should be cropped
            let should_crop = config
                .crop_classes
                .iter()
                .any(|class| class.to_string() == detection.class_name);

            if !should_crop {
                // Still add to output so we see it in the metadata
                detections_with_paths.push(DetectionWithPath {
                    detection: detection.clone(),
                    crop_path: None,
                });
                continue; // Skip this detection if its class is not in crop_classes
            }

            let suffix = format!("crop_{}", detection.class_name);
            let total_count = total_per_class
                .get(&detection.class_name)
                .cloned()
                .unwrap_or(1); // Default to 1 if not found
            let crop_filename = output_manager.generate_numbered_output_with_tracking(
                &suffix,
                i + 1,
                total_count,
                output_ext,
                true,
            )?;

            // Time the crop creation and save
            io_timing
                .time_save_operation(|| create_square_crop(img, detection, &crop_filename, 0.0))?;

            debug!(
                "{} Crop saved to: {}",
                symbols::completed_successfully(),
                crop_filename.display()
            );

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
        let bbox_filename = output_manager.generate_auxiliary_output_with_tracking(
            "bounding-box",
            output_ext,
            true,
        )?;

        // Time the bounding box image save
        io_timing
            .time_save_operation(|| save_bounding_box_image(img, detections, &bbox_filename))?;

        debug!(
            "{} Bounding box image saved to: {}",
            symbols::completed_successfully(),
            bbox_filename.display()
        );

        // Make path relative to metadata file if metadata will be created
        bounding_box_path = Some(output_manager.make_relative_to_metadata(&bbox_filename)?);
    }

    Ok((bounding_box_path, detections_with_paths))
}

/// Detection processor implementing the generic ModelProcessor trait
pub struct DetectionProcessor;

impl ModelProcessor for DetectionProcessor {
    type Config = DetectionConfig;
    type Result = DetectionResult;

    fn get_model_source<'a>(
        config: &Self::Config,
    ) -> Result<(
        ModelSource<'a>,
        Option<crate::shared_metadata::OnnxCacheStats>,
    )> {
        // Create CLI model info from config
        let cli_model_info = crate::model_access::CliModelInfo {
            model_path: config.model_path.clone(),
            model_url: config.model_url.clone(),
            model_checksum: config.model_checksum.clone(),
        };

        // Use CLI-aware model access
        HeadAccess::get_model_source_with_cli(&cli_model_info)
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
        output_manager: &crate::output_manager::OutputManager,
    ) -> Result<Self::Result> {
        let processing_start = Instant::now();
        let mut io_timing = IoTiming::new();

        // Load the image with timing
        let img = io_timing.time_image_read(image_path)?;
        let (orig_width, orig_height) = img.dimensions();

        debug!(
            "📷 Processing {}: {}x{}",
            image_path.display(),
            orig_width,
            orig_height
        );

        // Preprocess the image
        // Query the model size from the session:
        let input_md = &session.inputs[0];

        let dimensions = match &input_md.input_type {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => shape.to_vec(),
            _ => {
                debug!(
                    "Unexpected input type: {:?}. Defaulting to 640x640",
                    input_md.input_type
                );
                vec![1, 3, 640, 640] // Default to 1 batch, 3 channels, 640x640
            }
        };

        debug!("Input: {}, shape: {:?}", input_md.name, dimensions);
        let model_size = dimensions[3] as u32; // Assuming square input, use width

        let output0_name = session.outputs[0].name.clone();
        let output_dimensions = match &session.outputs[0].output_type {
            ort::value::ValueType::Tensor {
                ty,
                shape,
                dimension_symbols: _,
            } => {
                debug!("Output: {output0_name}, type: {ty:?}, shape: {shape:?}");
                shape.to_vec()
            }
            _ => {
                debug!("Unexpected output type for {output0_name}. Defaulting to 1x1x1x1");
                vec![1, 1, 1, 1]
            }
        };

        for (i, output) in session.outputs.iter().enumerate() {
            match &output.output_type {
                ort::value::ValueType::Tensor {
                    ty,
                    shape,
                    dimension_symbols,
                } => {
                    debug!(
                        "Output {}: {}, type: {:?}, shape: {:?}, dimension_symbols: {:?}",
                        i, output.name, ty, shape, dimension_symbols
                    );
                }
                other => {
                    debug!("Unexpected output type {:?} for {}", other, output.name);
                }
            };
        }
        let n_outputs = session.outputs.len();
        let model_variant = DetectionModelVariants::from_outputs(&output_dimensions, n_outputs);
        debug!("Output dimensions: {output_dimensions:?}, model variant: {model_variant:?}");

        let input_tensor = preprocess_image_for_model(&model_variant, &img, model_size)?;

        // Run inference using ORT v2 API with timing
        let inference_start = Instant::now();
        let input_value = Value::from_array(input_tensor)
            .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;
        let outputs = session
            .run(ort::inputs![input_md.name.clone() => &input_value])
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;
        let inference_time = inference_start.elapsed();

        // Postprocess to get detections
        let detections = postprocess_output(
            model_variant,
            config,
            &outputs,
            orig_width,
            orig_height,
            model_size,
        )?;

        debug!(
            "⚡ Inference completed in {:.1} ms",
            inference_time.as_secs_f64() * 1000.0
        );

        let total_processing_time = processing_start.elapsed().as_secs_f64() * 1000.0;

        // Handle outputs for this specific image (includes file I/O)
        let (bounding_box_path, detections_with_paths) = handle_image_outputs_with_timing(
            &img,
            &detections,
            image_path,
            config,
            &mut io_timing,
            output_manager,
        )?;

        Ok(DetectionResult {
            model_version: get_default_detect_model_info().name,
            processing_time_ms: total_processing_time,
            bounding_box_path,
            detections: detections_with_paths,
            io_timing,
            input_img_width: orig_width,
            input_img_height: orig_height,
        })
    }

    fn serialize_config(config: &Self::Config) -> Result<toml::Value> {
        Ok(toml::Value::try_from(config)?)
    }
}

fn postprocess_output(
    model_variant: DetectionModelVariants,
    config: &DetectionConfig,
    outputs: &ort::session::SessionOutputs,
    orig_width: u32,
    orig_height: u32,
    model_size: u32,
) -> Result<Vec<Detection>> {
    match model_variant {
        DetectionModelVariants::HeadOnly | DetectionModelVariants::MultiDetect => {
            // Extract the output tensor using ORT v2 API and convert to owned array
            let output_view: ndarray::ArrayBase<
                ndarray::ViewRepr<&f32>,
                ndarray::Dim<ndarray::IxDynImpl>,
            > = outputs["output0"]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;
            let output_array =
                Array::from_shape_vec(output_view.shape(), output_view.iter().cloned().collect())?;

            yolo::postprocess_output(
                &output_array,
                config.confidence,
                config.iou_threshold,
                orig_width,
                orig_height,
                model_size,
                model_variant.is_legacy_model(), // Pass legacy model flag
            )
        }
        DetectionModelVariants::Orientation => {
            rfdetr::postprocess_output(outputs, orig_width, orig_height, model_size, config)
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

    // Filter detections to only include bird and head classes
    let filtered_detections: Vec<&Detection> = detections
        .iter()
        .filter(|detection| detection.class_name == "bird" || detection.class_name == "head")
        .collect();

    // Determine if we should preserve alpha channel based on output format
    let preserve_alpha = if let Some(ext) = output_path.extension() {
        ext.to_string_lossy().to_lowercase() == "png"
    } else {
        false
    };

    // Always work in RGBA for drawing (preserves translucent text backgrounds)
    let mut rgba_img = output_img.to_rgba8();
    draw_detections(&mut rgba_img, &filtered_detections);

    // Convert back to appropriate format for output
    if preserve_alpha {
        output_img = DynamicImage::ImageRgba8(rgba_img);
    } else {
        // Convert back to RGB for JPEG output
        output_img = DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(rgba_img).to_rgb8());
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the image with bounding boxes
    output_img.save(output_path)?;

    Ok(())
}

/// Unified function to draw detections on an RGBA image
fn draw_detections(rgba_img: &mut image::RgbaImage, filtered_detections: &[&Detection]) {
    for detection in filtered_detections {
        let x1 = detection.x1.max(0.0) as u32;
        let y1 = detection.y1.max(0.0) as u32;
        let x2 = detection.x2.min(rgba_img.width() as f32) as u32;
        let y2 = detection.y2.min(rgba_img.height() as f32) as u32;

        // Choose color based on class_name: bird = forest green, head = blue
        let box_color = if detection.class_name == "bird" {
            image::Rgba([34, 139, 34, 255]) // Forest green for bird (nicer than bright green)
        } else {
            image::Rgba([0, 100, 255, 255]) // Bright blue for head
        };

        // Draw thick bounding box using imageproc (3 pixels thick)
        for thickness_offset in 0..3i32 {
            let thick_rect = imageproc::rect::Rect::at(
                (x1 as i32) - thickness_offset,
                (y1 as i32) - thickness_offset,
            )
            .of_size(
                (x2 - x1) + (thickness_offset * 2) as u32,
                (y2 - y1) + (thickness_offset * 2) as u32,
            );
            draw_hollow_rect_mut(rgba_img, thick_rect, box_color);
        }

        // Draw confidence text at top-left corner of bounding box
        let confidence_text = format!("{:.2}", detection.confidence);
        let text_x = x1 + 2; // Position text 2 pixels from left edge
        let text_y = y1.saturating_sub(25).max(2); // Position text above box, adjusted for larger text

        let font = FontRef::try_from_slice(FONT_BYTES).expect("Font load failed");
        let scale = PxScale::from(20.0); // Larger font size
        let text_color = image::Rgba([255, 255, 255, 255]); // White text
        let bg_color = image::Rgba([0, 0, 0, 120]); // More transparent black background
        let (text_width, text_height) = text_size(scale, &font, &confidence_text);

        // Draw background rectangle (adjusted to properly center around text)
        let y_offset: i32 = 4;
        let bg_x = text_x;
        let bg_y = text_y + 2 + (y_offset as u32); // empirical offsets
        for dx in 0..(text_width + 4) {
            for dy in 0..(text_height + 4) {
                let px = bg_x + dx;
                let py = bg_y + dy;
                if px < rgba_img.width() && py < rgba_img.height() {
                    rgba_img.put_pixel(px, py, bg_color);
                }
            }
        }
        draw_text_mut(
            rgba_img,
            text_color,
            text_x as i32 + 2,
            text_y as i32 + y_offset,
            scale,
            &font,
            &confidence_text,
        );

        // Draw angle line if angle is not NaN (make it same thickness as box)
        if !detection.angle_radians.is_nan() {
            let center_x = (x1 + x2) as f32 / 2.0;
            let center_y = (y1 + y2) as f32 / 2.0;
            let box_width = (x2 - x1) as f32;
            let box_height = (y2 - y1) as f32;
            let line_length = box_width.min(box_height) / 2.0; // half the smaller dimension

            let end_x = center_x + line_length * detection.angle_radians.cos();
            let end_y = center_y + line_length * detection.angle_radians.sin();

            // Draw thick line (3 pixels thick like the box)
            for thickness_offset in -1..=1i32 {
                // Draw horizontal thick lines
                for extra_thickness in 0..3i32 {
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            center_x + thickness_offset as f32,
                            center_y + extra_thickness as f32,
                        ),
                        (
                            end_x + thickness_offset as f32,
                            end_y + extra_thickness as f32,
                        ),
                        box_color,
                    );
                }
                // Draw vertical thick lines
                for extra_thickness in 0..3i32 {
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            center_x + extra_thickness as f32,
                            center_y + thickness_offset as f32,
                        ),
                        (
                            end_x + extra_thickness as f32,
                            end_y + thickness_offset as f32,
                        ),
                        box_color,
                    );
                }
            }

            // Draw arrowhead at the end of the line
            let arrowhead_length = 8.0; // Length of arrowhead sides
            let arrowhead_angle = std::f32::consts::PI / 6.0; // 30 degrees

            // Calculate arrowhead points
            let arrow_angle1 = detection.angle_radians + std::f32::consts::PI - arrowhead_angle;
            let arrow_angle2 = detection.angle_radians + std::f32::consts::PI + arrowhead_angle;

            let arrow_x1 = end_x + arrowhead_length * arrow_angle1.cos();
            let arrow_y1 = end_y + arrowhead_length * arrow_angle1.sin();
            let arrow_x2 = end_x + arrowhead_length * arrow_angle2.cos();
            let arrow_y2 = end_y + arrowhead_length * arrow_angle2.sin();

            // Draw arrowhead lines with same thickness as main line
            for thickness_offset in -1..=1i32 {
                for extra_thickness in 0..3i32 {
                    // First arrowhead line
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            end_x + thickness_offset as f32,
                            end_y + extra_thickness as f32,
                        ),
                        (
                            arrow_x1 + thickness_offset as f32,
                            arrow_y1 + extra_thickness as f32,
                        ),
                        box_color,
                    );
                    // Second arrowhead line
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            end_x + thickness_offset as f32,
                            end_y + extra_thickness as f32,
                        ),
                        (
                            arrow_x2 + thickness_offset as f32,
                            arrow_y2 + extra_thickness as f32,
                        ),
                        box_color,
                    );
                }
                for extra_thickness in 0..3i32 {
                    // First arrowhead line (vertical thickness)
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            end_x + extra_thickness as f32,
                            end_y + thickness_offset as f32,
                        ),
                        (
                            arrow_x1 + extra_thickness as f32,
                            arrow_y1 + thickness_offset as f32,
                        ),
                        box_color,
                    );
                    // Second arrowhead line (vertical thickness)
                    draw_line_segment_mut(
                        rgba_img,
                        (
                            end_x + extra_thickness as f32,
                            end_y + thickness_offset as f32,
                        ),
                        (
                            arrow_x2 + extra_thickness as f32,
                            arrow_y2 + thickness_offset as f32,
                        ),
                        box_color,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Environment variable tests are now in integration tests to avoid race conditions

    #[test]
    fn test_head_access_env_var_name() {
        assert_eq!(HeadAccess::get_env_var_name(), "BEAKER_DETECT_MODEL_PATH");
    }

    #[test]
    fn test_preprocessing_dispatch() {
        use image::{Rgb, RgbImage};

        // Create a test image
        let img = RgbImage::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let dynamic_img = DynamicImage::ImageRgb8(img);

        // Test YOLO preprocessing (should use letterboxing)
        let yolo_result =
            preprocess_image_for_model(&DetectionModelVariants::HeadOnly, &dynamic_img, 640)
                .unwrap();
        assert_eq!(yolo_result.shape(), &[1, 3, 640, 640]);

        // Test RF-DETR preprocessing (should use square resize)
        let rfdetr_result =
            preprocess_image_for_model(&DetectionModelVariants::Orientation, &dynamic_img, 640)
                .unwrap();
        assert_eq!(rfdetr_result.shape(), &[1, 3, 640, 640]);

        // The results should be different because the preprocessing methods differ
        // (YOLO uses letterboxing with gray padding, RF-DETR uses square resize with ImageNet normalization)
        assert_ne!(yolo_result.as_slice(), rfdetr_result.as_slice());
    }
}
