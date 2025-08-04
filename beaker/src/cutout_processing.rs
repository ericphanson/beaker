use anyhow::Result;
use chrono::{DateTime, Utc};
use image::GenericImageView;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::config::CutoutConfig;
use crate::cutout_postprocessing::{
    apply_alpha_matting, create_cutout, create_cutout_with_background, postprocess_mask,
};
use crate::cutout_preprocessing::preprocess_image_for_isnet_v2;
use crate::model_cache::{get_or_download_model, ISNET_GENERAL_MODEL};
use crate::output_manager::OutputManager;
use log::{debug, info};

#[derive(Serialize, Clone)]
pub struct CutoutResult {
    pub input_path: String,
    pub output_path: String,
    pub model_version: String,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
}

/// Core results for enhanced metadata (without config duplication)
#[derive(Serialize)]
pub struct CutoutCoreResult {
    pub timestamp: DateTime<Utc>,
    pub model_version: String,
    pub input_path: String,
    pub output_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
}

/// Check if a file is a supported image format
/// Process a single image with an existing session
fn process_single_image(
    config: &CutoutConfig,
    session: &mut Session,
    image_path: &Path,
) -> Result<CutoutResult> {
    let start_time = Instant::now();

    debug!("🖼️  Processing: {}", image_path.display());

    // Load and preprocess the image
    let img = image::open(image_path)?;
    let original_size = img.dimensions();

    let input_array = preprocess_image_for_isnet_v2(&img)?;

    // Prepare input for the model
    let input_name = session.inputs[0].name.clone();
    let output_name = session.outputs[0].name.clone();
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

    // Extract the mask from the output (shape should be [1, 1, 1024, 1024])
    let mask_2d = output_view.slice(ndarray::s![0, 0, .., ..]);

    // Post-process the mask
    let mask = postprocess_mask(&mask_2d, original_size, config.post_process_mask)?;

    // Generate output paths using OutputManager
    let output_manager = OutputManager::new(config, image_path);
    let output_path = output_manager.generate_main_output_path("cutout", "png")?;
    let mask_path = if config.save_mask {
        Some(output_manager.generate_auxiliary_output("mask", "png")?)
    } else {
        None
    };

    // Create the cutout
    let cutout_result = if config.alpha_matting {
        apply_alpha_matting(
            &img,
            &mask,
            config.alpha_matting_foreground_threshold,
            config.alpha_matting_background_threshold,
            config.alpha_matting_erode_size,
        )?
    } else if let Some(bg_color) = config.background_color {
        create_cutout_with_background(&img, &mask, bg_color)?
    } else {
        create_cutout(&img, &mask)?
    };

    // Save the cutout (always PNG for transparency)
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    cutout_result.save(&output_path)?;

    // Save mask if requested
    if let Some(mask_path_val) = &mask_path {
        if let Some(parent) = Path::new(mask_path_val).parent() {
            fs::create_dir_all(parent)?;
        }
        mask.save(mask_path_val)?;
    }

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    info!(
        "✅ Processed {} in {:.1}ms → {}",
        image_path.display(),
        processing_time,
        output_path.display()
    );

    let cutout_result = CutoutResult {
        input_path: image_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        model_version: "isnet-general-use".to_string(),
        processing_time_ms: processing_time,
        mask_path: mask_path.map(|p| p.to_string_lossy().to_string()),
    };

    Ok(cutout_result)
}

/// Process multiple images sequentially
pub fn run_cutout_processing(config: CutoutConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<CutoutProcessor>(config)
}

// Implementation of ModelProcessor trait for cutout processing
use crate::model_processing::{ModelProcessor, ModelResult};

impl ModelResult for CutoutResult {
    fn result_summary(&self) -> String {
        if self.mask_path.is_some() {
            "Generated cutout and mask".to_string()
        } else {
            "Generated cutout".to_string()
        }
    }

    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "cutout"
    }

    fn core_results(&self) -> Result<serde_json::Value> {
        let cutout_core_result = CutoutCoreResult {
            timestamp: chrono::Utc::now(),
            model_version: self.model_version.clone(),
            input_path: self.input_path.clone(),
            output_path: self.output_path.clone(),
            mask_path: self.mask_path.clone(),
        };

        Ok(serde_json::to_value(cutout_core_result)?)
    }
}

/// Cutout processor implementing the generic ModelProcessor trait
pub struct CutoutProcessor;

impl ModelProcessor for CutoutProcessor {
    type Config = CutoutConfig;
    type Result = CutoutResult;

    fn create_session_with_device(device: &str) -> Result<Session> {
        use crate::model_processing::create_session_with_device;
        use crate::onnx_session::ModelSource;

        // Download model if needed
        let model_path = get_or_download_model(&ISNET_GENERAL_MODEL)?;

        create_session_with_device(ModelSource::FilePath(model_path.to_str().unwrap()), device)
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        // Use the existing process_single_image function
        process_single_image(config, session, image_path)
    }

    fn serialize_config(config: &Self::Config) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(config)?)
    }
}
