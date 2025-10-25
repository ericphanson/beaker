//! Simplified API for GUI integration
//!
//! This module provides a simplified, single-image API for GUI applications,
//! wrapping the more complex batch processing infrastructure.

use crate::config::{BaseModelConfig, DetectionClass, DetectionConfig};
use crate::detection::{DetectionProcessor, DetectionResult};
use crate::image_input::ImageInput;
use crate::model_processing::ModelProcessor;
use crate::onnx_session::create_session;
use crate::output_manager::OutputManager;
use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;

/// Simplified parameters for single-image detection
pub struct SimpleDetectionParams {
    pub image_path: PathBuf,
    pub confidence: f32,
    pub iou_threshold: f32,
    pub crop_classes: HashSet<DetectionClass>,
    pub bounding_box: bool,
    pub output_dir: Option<PathBuf>,
}

/// Detect objects in a single image with simplified parameters
pub fn detect_single_image(params: SimpleDetectionParams) -> Result<DetectionResult> {
    // Create base config
    let sources = ImageInput::parse_sources(&[params.image_path.to_string_lossy().to_string()])?;
    let base_config = BaseModelConfig {
        sources,
        output_dir: params.output_dir.map(|p| p.to_string_lossy().to_string()),
        ..Default::default()
    };

    // Create detection config
    let config = DetectionConfig {
        base: base_config,
        confidence: params.confidence,
        iou_threshold: params.iou_threshold,
        crop_classes: params.crop_classes,
        bounding_box: params.bounding_box,
        model_path: None,
        model_url: None,
        model_checksum: None,
        quality_results: None,
    };

    // Get model source
    let (model_source, _cache_stats) = DetectionProcessor::get_model_source(&config)?;

    // Create session
    let mut session = create_session(&model_source, &config.base)?;

    // Create output manager
    let output_manager = OutputManager::new(
        &params.image_path,
        config.base.output_dir.as_deref(),
        config.base.metadata.as_ref(),
        "detect",
    )?;

    // Process the single image
    DetectionProcessor::process_single_image(
        &mut session,
        &params.image_path,
        &config,
        &output_manager,
    )
}
