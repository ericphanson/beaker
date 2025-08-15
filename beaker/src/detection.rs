use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

use crate::color_utils::symbols;
use crate::config::DetectionConfig;
use crate::detection_obj::Detection;
use crate::model_access::{ModelAccess, ModelInfo};
use crate::model_processing::{ModelProcessor, ModelResult};
use crate::onnx_session::ModelSource;
use crate::output_manager::OutputManager;
use crate::rfdetr_postprocessing;
use crate::rfdetr_preprocessing;
use crate::shared_metadata::IoTiming;
use crate::yolo_postprocessing;
use crate::yolo_postprocessing::{create_square_crop, save_bounding_box_image};
use crate::yolo_preprocessing;
use log::debug;

pub fn get_default_detect_model_info() -> ModelInfo {
    ModelInfo {
        name: "bird-orientation-detector-v0.1.0".to_string(),
        url: "https://github.com/ericphanson/beaker/releases/download/bird-orientation-detector-v0.1.0/rfdetr-dynamic-int8.onnx".to_string(),
        md5_checksum: "256548ae718eb5da232f2c9823828447".to_string(),
        filename: "bird-orientation-detector-v0.1.0.onnx".to_string(),
    }
}

#[derive(Debug)]
enum DetectionModelVariants {
    HeadOnlyModelVariant,
    MultiDetectModelVariant,
    OrientationModelVariant,
}

impl DetectionModelVariants {
    /// Determine the model variant from output dimensions
    fn from_outputs(output_dimensions: &[i64], n_outputs: usize) -> Self {
        // Check if legacy model based on output channels
        let is_legacy_model = output_dimensions[1] < 8;

        if is_legacy_model {
            DetectionModelVariants::HeadOnlyModelVariant
        } else if n_outputs == 1 {
            DetectionModelVariants::MultiDetectModelVariant
        } else {
            DetectionModelVariants::OrientationModelVariant
        }
    }

    /// Convert to boolean for compatibility with existing postprocessing function
    fn is_legacy_model(&self) -> bool {
        matches!(self, DetectionModelVariants::HeadOnlyModelVariant)
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
        DetectionModelVariants::HeadOnlyModelVariant
        | DetectionModelVariants::MultiDetectModelVariant => {
            // YOLO models use letterboxing with gray padding
            yolo_preprocessing::preprocess_image(img, model_size)
        }
        DetectionModelVariants::OrientationModelVariant => {
            // RF-DETR models use square resize with ImageNet normalization
            rfdetr_preprocessing::preprocess_image(img, model_size)
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
        DetectionModelVariants::HeadOnlyModelVariant
        | DetectionModelVariants::MultiDetectModelVariant => {
            // Extract the output tensor using ORT v2 API and convert to owned array
            let output_view: ndarray::ArrayBase<
                ndarray::ViewRepr<&f32>,
                ndarray::Dim<ndarray::IxDynImpl>,
            > = outputs["output0"]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;
            let output_array =
                Array::from_shape_vec(output_view.shape(), output_view.iter().cloned().collect())?;

            yolo_postprocessing::postprocess_output(
                &output_array,
                config.confidence,
                config.iou_threshold,
                orig_width,
                orig_height,
                model_size,
                model_variant.is_legacy_model(), // Pass legacy model flag
            )
        }
        DetectionModelVariants::OrientationModelVariant => {
            rfdetr_postprocessing::postprocess_output(
                outputs,
                orig_width,
                orig_height,
                model_size,
                config,
            )
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
        let yolo_result = preprocess_image_for_model(
            &DetectionModelVariants::HeadOnlyModelVariant,
            &dynamic_img,
            640,
        )
        .unwrap();
        assert_eq!(yolo_result.shape(), &[1, 3, 640, 640]);

        // Test RF-DETR preprocessing (should use square resize)
        let rfdetr_result = preprocess_image_for_model(
            &DetectionModelVariants::OrientationModelVariant,
            &dynamic_img,
            640,
        )
        .unwrap();
        assert_eq!(rfdetr_result.shape(), &[1, 3, 640, 640]);

        // The results should be different because the preprocessing methods differ
        // (YOLO uses letterboxing with gray padding, RF-DETR uses square resize with ImageNet normalization)
        assert_ne!(yolo_result.as_slice(), rfdetr_result.as_slice());
    }

    #[test]
    fn test_head_access_embedded_bytes_available() {
        let bytes = HeadAccess::get_embedded_bytes();
        assert!(bytes.is_some(), "Head model should have embedded bytes");
        assert!(
            !bytes.unwrap().is_empty(),
            "Embedded model bytes should not be empty"
        );
    }
}
