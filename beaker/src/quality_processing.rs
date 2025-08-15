use crate::color_utils::symbols;
use crate::config::QualityConfig;
use crate::model_access::{ModelAccess, ModelInfo};
use crate::onnx_session::ModelSource;
use crate::shared_metadata::IoTiming;
use anyhow::Result;
use image::GenericImageView;
use log::debug;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

/// Quality model default model information
pub fn get_default_quality_model_info() -> ModelInfo {
    ModelInfo {
        name: "quality-model-v1".to_string(),
        url: "https://github.com/ericphanson/beaker/releases/download/quality-model-v1/quality-dynamic-int8.onnx".to_string(),
        md5_checksum: "8691fda05ee55c8552c412e5e62551cb".to_string(),
        filename: "quality-model-v1.onnx".to_string(),
    }
}

/// Quality model access implementation.
pub struct QualityAccess;

impl ModelAccess for QualityAccess {
    fn get_embedded_bytes() -> Option<&'static [u8]> {
        // Quality models are not embedded, they are downloaded
        None
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_QUALITY_MODEL_PATH"
    }

    fn get_url_env_var_name() -> Option<&'static str> {
        Some("BEAKER_QUALITY_MODEL_URL")
    }

    fn get_checksum_env_var_name() -> Option<&'static str> {
        Some("BEAKER_QUALITY_MODEL_CHECKSUM")
    }

    fn get_default_model_info() -> Option<ModelInfo> {
        Some(get_default_quality_model_info())
    }
}

/// Core results for quality assessment metadata
#[derive(Serialize)]
pub struct QualityResult {
    pub model_version: String,
    #[serde(skip_serializing)]
    pub processing_time_ms: f64,
    pub quality_score: f64,
    pub local_quality_grid: [[u8; 20]; 20], // Fixed-size 20x20 grid of integers 0-100
    #[serde(skip_serializing)]
    pub io_timing: IoTiming,
    pub input_img_width: u32,  // Original input image width
    pub input_img_height: u32, // Original input image height
}

/// Process multiple images sequentially
pub fn run_quality_processing(config: QualityConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<QualityProcessor>(config)
}

// Implementation of ModelProcessor trait for quality processing
use crate::model_processing::{ModelProcessor, ModelResult};

impl ModelResult for QualityResult {
    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "quality"
    }

    fn core_results(&self) -> Result<toml::Value> {
        Ok(toml::Value::try_from(self)?)
    }

    fn output_summary(&self) -> String {
        format!("quality: {:.3}", self.quality_score)
    }

    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming {
        self.io_timing.clone()
    }

    fn get_mask_entry(&self) -> Option<crate::mask_encoding::MaskEntry> {
        // Quality assessment doesn't produce masks
        None
    }
}

/// Quality processor implementing the generic ModelProcessor trait
pub struct QualityProcessor;

impl ModelProcessor for QualityProcessor {
    type Config = QualityConfig;
    type Result = QualityResult;

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
        QualityAccess::get_model_source_with_cli(&cli_model_info)
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        _config: &Self::Config,
        _output_manager: &crate::output_manager::OutputManager,
    ) -> Result<Self::Result> {
        let start_time = Instant::now();
        let mut io_timing = IoTiming::new();

        debug!("🖼️  Processing: {}", image_path.display());

        // Load and preprocess the image with timing
        let img = io_timing.time_image_read(image_path)?;
        let original_size = img.dimensions();

        // Placeholder preprocessing - replace with actual implementation
        let input_array = preprocess_image_for_quality(&img)?;

        // Prepare input for the model
        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();
        debug!("🖼️  Running inference: {input_name} -> {output_name}");
        let input_value = Value::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;

        // Run inference
        let outputs = session
            .run(ort::inputs![input_name.as_str() => &input_value])
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

        // Extract the output tensor using ORT v2 API
        let output_view = outputs[output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;

        // Placeholder postprocessing - replace with actual implementation
        let (quality_score, local_quality_grid) = postprocess_quality_output(&output_view)?;

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "{} Quality assessment completed: score={:.3}",
            symbols::completed_successfully(),
            quality_score
        );

        // Create result with timing information
        let quality_result = QualityResult {
            model_version: get_default_quality_model_info().name,
            processing_time_ms: processing_time,
            quality_score,
            local_quality_grid,
            io_timing,
            input_img_width: original_size.0,
            input_img_height: original_size.1,
        };

        Ok(quality_result)
    }

    fn serialize_config(config: &Self::Config) -> Result<toml::Value> {
        Ok(toml::Value::try_from(config)?)
    }
}

fn preprocess_image_for_quality(img: &image::DynamicImage) -> Result<ndarray::Array4<f32>> {
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let rgb_img = resized.to_rgb8();

    // Convert to normalized float array [1, 3, 224, 224]
    let mut array = ndarray::Array4::<f32>::zeros((1, 3, 224, 224));

    log::debug!("Got RGB image with dimensions: {:?}", rgb_img.dimensions());

    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }

    log::debug!("Converted image to array with shape: {:?}", array.shape());

    Ok(array)
}

/// Placeholder postprocessing function for quality assessment
fn postprocess_quality_output(
    output: &ndarray::ArrayView<f32, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Result<(f64, [[u8; 20]; 20])> {
    // Placeholder implementation - extract quality score and confidence
    // Assumes output is a single value or pair of values
    log::debug!("Quality model output: {:?}", output.shape());
    let output_data = output
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("Failed to get output slice"))?;

    if output_data.is_empty() {
        return Err(anyhow::anyhow!("Empty output data"));
    }

    // Placeholder logic - replace with actual model output interpretation
    let quality_score = output_data[0] as f64;

    // the next 400 values are a 20x20 grid of local quality scores
    let mut local_quality_grid = [[0u8; 20]; 20];
    if output_data.len() >= 401 {
        for (i, &value) in output_data[1..401].iter().enumerate() {
            // Values are already between 0 and 100, just round to nearest integer
            let rounded_value = value.round().clamp(0.0, 100.0) as u8;
            local_quality_grid[i / 20][i % 20] = rounded_value;
        }
    } else {
        return Err(anyhow::anyhow!("Insufficient output data for 20x20 grid"));
    }

    Ok((quality_score, local_quality_grid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_access_env_vars() {
        // Test environment variable names
        assert_eq!(
            QualityAccess::get_env_var_name(),
            "BEAKER_QUALITY_MODEL_PATH"
        );
        assert_eq!(
            QualityAccess::get_url_env_var_name(),
            Some("BEAKER_QUALITY_MODEL_URL")
        );
        assert_eq!(
            QualityAccess::get_checksum_env_var_name(),
            Some("BEAKER_QUALITY_MODEL_CHECKSUM")
        );
    }

    #[test]
    fn test_quality_access_no_embedded_bytes() {
        let bytes = QualityAccess::get_embedded_bytes();
        assert!(
            bytes.is_none(),
            "Quality model should not have embedded bytes"
        );
    }

    #[test]
    fn test_quality_access_has_default_info() {
        let model_info = QualityAccess::get_default_model_info();
        assert!(
            model_info.is_some(),
            "Quality model should have default model info"
        );

        let info = model_info.unwrap();
        assert_eq!(info.name, "quality-model-v1");
        assert!(info.url.contains("quality-model.onnx"));
        assert!(!info.md5_checksum.is_empty());
        assert_eq!(info.filename, "quality-model.onnx");
    }

    #[test]
    fn test_get_default_quality_model_info() {
        let model_info = get_default_quality_model_info();
        assert_eq!(model_info.name, "quality-model-v1");
        assert!(model_info.url.contains("quality-model.onnx"));
        assert!(!model_info.md5_checksum.is_empty());
        assert_eq!(model_info.filename, "quality-model.onnx");
    }

    #[test]
    fn test_runtime_model_info_with_quality_overrides() {
        use crate::model_access::RuntimeModelInfo;

        // Test RuntimeModelInfo creation without env var modification
        let default_info = get_default_quality_model_info();

        // Test without any env vars (should use default info)
        let runtime_info = RuntimeModelInfo::from_model_info_with_overrides(
            &default_info,
            Some("NONEXISTENT_TEST_URL"),
            Some("NONEXISTENT_TEST_CHECKSUM"),
        );

        assert_eq!(runtime_info.name, default_info.name);
        assert_eq!(runtime_info.url, default_info.url);
        assert_eq!(runtime_info.md5_checksum, default_info.md5_checksum);
        assert_eq!(runtime_info.filename, default_info.filename);
    }
}
