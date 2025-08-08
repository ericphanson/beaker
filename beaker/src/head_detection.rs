use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

use crate::color_utils::symbols;
use crate::config::HeadDetectionConfig;
use crate::model_access::{get_model_source_with_env_override, ModelAccess};
use crate::model_processing::{ModelProcessor, ModelResult};
use crate::onnx_session::ModelSource;
use crate::output_manager::OutputManager;
use crate::shared_metadata::IoTiming;
use crate::yolo_postprocessing::{
    create_square_crop, postprocess_output, save_bounding_box_image, Detection,
};
use crate::yolo_preprocessing::preprocess_image;
use log::debug;

// Embed the ONNX model at compile time
pub const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bird-head-detector.onnx"));

// Get model version from build script
pub const MODEL_VERSION: &str =
    include_str!(concat!(env!("OUT_DIR"), "/bird-head-detector.version"));

/// Head detection model access implementation.
pub struct HeadAccess;

impl ModelAccess for HeadAccess {
    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        get_model_source_with_env_override::<Self>()
    }

    fn get_embedded_bytes() -> Option<&'static [u8]> {
        // Reference to the embedded model bytes
        Some(MODEL_BYTES)
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_HEAD_MODEL_PATH"
    }

    // Currently, head models don't support remote download (embedded only)
    // But this could be added in the future by uncommenting the following:
    // fn get_default_model_info() -> Option<ModelInfo> {
    //     Some(HEAD_MODEL_INFO)
    // }
}

#[derive(Serialize)]
pub struct HeadDetectionResult {
    pub model_version: String,
    #[serde(skip_serializing)]
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box_path: Option<String>,
    pub detections: Vec<DetectionWithPath>,
    #[serde(skip_serializing)]
    pub io_timing: IoTiming,
}

#[derive(Serialize, Clone)]
pub struct DetectionWithPath {
    #[serde(flatten)]
    pub detection: Detection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_path: Option<String>,
}

/// Process multiple images sequentially
pub fn run_head_detection(config: HeadDetectionConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<HeadProcessor>(config)
}

impl ModelResult for HeadDetectionResult {
    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "head"
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

    fn get_io_timing(&self) -> Option<crate::shared_metadata::IoTiming> {
        Some(self.io_timing.clone())
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

/// Handle outputs (crops, bounding boxes, metadata) for a single image with I/O timing
fn handle_image_outputs_with_timing(
    img: &DynamicImage,
    detections: &[Detection],
    image_path: &Path,
    config: &HeadDetectionConfig,
    io_timing: &mut IoTiming,
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

            // Time the crop creation and save
            io_timing
                .time_save_operation(|| create_square_crop(img, detection, &crop_filename, 0.1))?;

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
        let bbox_filename = output_manager.generate_auxiliary_output("bounding-box", output_ext)?;

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

/// Head detection processor implementing the generic ModelProcessor trait
pub struct HeadProcessor;

impl ModelProcessor for HeadProcessor {
    type Config = HeadDetectionConfig;
    type Result = HeadDetectionResult;

    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        HeadAccess::get_model_source()
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
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
        )?;

        Ok(HeadDetectionResult {
            model_version: MODEL_VERSION.to_string(),
            processing_time_ms: total_processing_time,
            bounding_box_path,
            detections: detections_with_paths,
            io_timing,
        })
    }

    fn serialize_config(config: &Self::Config) -> Result<toml::Value> {
        Ok(toml::Value::try_from(config)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::NamedTempFile;

    #[test]
    fn test_head_access_embedded_default() {
        // Ensure env var is not set
        env::remove_var("BEAKER_HEAD_MODEL_PATH");

        let source = HeadAccess::get_model_source().unwrap();

        match source {
            ModelSource::EmbeddedBytes(bytes) => {
                assert!(
                    !bytes.is_empty(),
                    "Embedded model bytes should not be empty"
                );
            }
            _ => panic!("Expected embedded bytes when no env var is set"),
        }
    }

    #[test]
    fn test_head_access_env_override() {
        // Create a temporary file to act as a model
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        // Set environment variable
        env::set_var("BEAKER_HEAD_MODEL_PATH", temp_path);

        let source = HeadAccess::get_model_source().unwrap();

        match source {
            ModelSource::FilePath(path) => {
                assert_eq!(path, temp_path);
            }
            _ => panic!("Expected file path when env var is set"),
        }

        // Clean up
        env::remove_var("BEAKER_HEAD_MODEL_PATH");
    }

    #[test]
    fn test_head_access_invalid_path() {
        // Use a path that's guaranteed to not exist on any platform
        // Use a path in a directory that doesn't exist with invalid characters
        let non_existent_path = if cfg!(windows) {
            // On Windows, use a path with invalid characters
            "C:\\this\\path\\definitely\\does\\not\\exist\\model.onnx"
        } else {
            // On Unix-like systems, use a deeply nested non-existent path
            "/this/path/definitely/does/not/exist/model.onnx"
        };

        // Set environment variable to non-existent path
        env::set_var("BEAKER_HEAD_MODEL_PATH", non_existent_path);

        let result = HeadAccess::get_model_source();
        assert!(result.is_err(), "Should fail with non-existent path");

        let error_msg = result.err().unwrap().to_string();
        assert!(
            error_msg.contains("does not exist"),
            "Error should mention non-existent path"
        );

        // Clean up
        env::remove_var("BEAKER_HEAD_MODEL_PATH");
    }

    #[test]
    fn test_head_access_env_var_name() {
        assert_eq!(HeadAccess::get_env_var_name(), "BEAKER_HEAD_MODEL_PATH");
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
