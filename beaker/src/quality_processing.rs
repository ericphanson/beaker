use crate::blur_detection::compute_raw_tenengrad;
use crate::color_utils::symbols;
use crate::config::QualityConfig;
use crate::model_access::{ModelAccess, ModelInfo};
use crate::onnx_session::ModelSource;
use crate::quality_types::QualityRawData;
use crate::shared_metadata::IoTiming;
use anyhow::{Context, Result};
use cached::proc_macro::cached;
use log::debug;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::path::Path;
use std::time::{Instant, SystemTime};

/// Quality model default model information
pub fn get_default_quality_model_info() -> ModelInfo {
    ModelInfo {
        name: "quality-model-v1".to_string(),
        url: "https://github.com/ericphanson/beaker/releases/download/quality-model-v1/quality-dynamic-int8.onnx".to_string(),
        md5_checksum: Some("8691fda05ee55c8552c412e5e62551cb".to_string()),
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
#[derive(Serialize, Clone, Debug)]
pub struct QualityResult {
    pub model_version: String,
    #[serde(skip_serializing)]
    pub processing_time_ms: f64,
    pub global_quality_score: f32,
    pub global_paq2piq_score: f32,
    pub global_blur_score: f32,
    pub local_paq2piq_grid: [[u8; 20]; 20], // Fixed-size 20x20 grid of integers 0-100
    pub local_blur_weights: [[f32; 20]; 20], // Fixed-size 20x20 grid of floats 0.0-1.0
    pub local_fused_probability: [[f32; 20]; 20], // Fixed-size 20x20 grid of floats 0.0-1.0
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
        format!("quality: {:.3}", self.global_paq2piq_score)
    }

    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming {
        self.io_timing.clone()
    }

    fn get_quality_result(&self) -> Option<QualityResult> {
        Some(self.clone())
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
        config: &Self::Config,
        output_manager: &crate::output_manager::OutputManager,
    ) -> Result<Self::Result> {
        let start_time = Instant::now();
        let mut io_timing = IoTiming::new();

        debug!("Processing: {}", image_path.display());

        // Record image load time
        let _img = io_timing.time_image_read(image_path)?;

        // Level 1: Compute raw data (cached automatically)
        let raw = compute_quality_raw(image_path, session)?;

        // Level 2: Compute scores from raw data with parameters
        let params = config.params.clone().unwrap_or_default();
        let scores = crate::quality_types::QualityScores::compute(&raw, &params);

        // Generate debug visualizations if requested
        if config.debug_dump_images {
            let input_stem = output_manager.input_stem();
            let output_dir = output_manager
                .get_output_dir()?
                .join(format!("quality_debug_images_{input_stem}"));

            // Load and preprocess the image for debug visualization
            let img =
                image::open(image_path).context("Failed to open image for debug visualization")?;
            let input_array = preprocess_image_for_quality(&img)?;

            // Generate debug images using blur_weights_from_nchw
            let _ = crate::blur_detection::blur_weights_from_nchw(&input_array, Some(output_dir));
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "{} Quality assessment completed: score={:.3}",
            symbols::completed_successfully(),
            scores.final_score
        );

        // Convert QualityScores to QualityResult format for metadata output
        let quality_result = QualityResult {
            model_version: raw.model_version,
            processing_time_ms: processing_time,
            global_quality_score: scores.final_score,
            global_paq2piq_score: scores.paq2piq_score,
            global_blur_score: scores.blur_score,
            local_paq2piq_grid: raw.paq2piq_local,
            local_blur_weights: scores.blur_weights,
            local_fused_probability: scores.blur_probability,
            io_timing,
            input_img_width: raw.input_width,
            input_img_height: raw.input_height,
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

/// Load ONNX session with default model path
#[allow(dead_code)]
pub fn load_onnx_session_default() -> Result<Session> {
    let model_dir = std::env::var("ONNX_MODEL_CACHE_DIR").unwrap_or_else(|_| "models".to_string());
    let model_path = Path::new(&model_dir).join("quality_model.onnx");

    // Read model bytes
    let bytes = std::fs::read(&model_path)
        .with_context(|| format!("Failed to read model file: {}", model_path.display()))?;

    // Create session from bytes
    Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
        .commit_from_memory(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))
}

/// Compute parameter-independent quality data (expensive: ~60ms, cached)
#[cached(
    size = 1000,
    key = "String",
    convert = r#"{ format!("{}", path.as_ref().display()) }"#,
    result = true
)]
pub fn compute_quality_raw(
    path: impl AsRef<Path>,
    session: &mut Session,
) -> Result<QualityRawData> {
    // Load and preprocess image
    let img = image::open(path.as_ref()).context("Failed to open image")?;

    let input_width = img.width();
    let input_height = img.height();

    // Preprocess for ONNX
    let input_array = preprocess_image_for_quality(&img)?;

    // Run ONNX inference
    let input_name = session.inputs[0].name.clone();
    let output_name = session.outputs[0].name.clone();
    let input_value = Value::from_array(input_array.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;
    let outputs = session
        .run(ort::inputs![input_name.as_str() => &input_value])
        .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

    // Extract outputs
    let output_view = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;

    // Parse ONNX outputs (same as current code)
    let output_data = output_view
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("Failed to get output slice"))?;

    let global_idx = 400;
    let paq2piq_global = output_data[global_idx].clamp(0.0, 100.0);

    let mut paq2piq_local = [[0u8; 20]; 20];
    for (i, row) in paq2piq_local.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let idx = i * 20 + j;
            let val = output_data[idx].clamp(0.0, 100.0);
            *cell = val as u8;
        }
    }

    // Compute raw Tenengrad
    let raw_tenengrad = compute_raw_tenengrad(&input_array)?;

    // Convert ndarray to fixed array
    let mut tenengrad_224 = [[0.0f32; 20]; 20];
    let mut tenengrad_112 = [[0.0f32; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            tenengrad_224[i][j] = raw_tenengrad.t224[[i, j]];
            tenengrad_112[i][j] = raw_tenengrad.t112[[i, j]];
        }
    }

    Ok(QualityRawData {
        input_width,
        input_height,
        paq2piq_global,
        paq2piq_local,
        tenengrad_224,
        tenengrad_112,
        median_tenengrad_224: raw_tenengrad.median_224,
        scale_ratio: raw_tenengrad.scale_ratio,
        model_version: "quality-model-v1".to_string(),
        computed_at: SystemTime::now(),
    })
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
