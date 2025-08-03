//! Model processing framework providing unified interface for all models.
//!
//! This module defines the core `ModelProcessor` trait that all models must implement,
//! along with generic processing functions that handle the common patterns across
//! different model types.

use anyhow::Result;
use ort::session::Session;
use std::path::Path;

use crate::config::BaseModelConfig;
use crate::image_input::{collect_images_from_sources, ImageInputConfig};
use crate::onnx_session::{
    create_onnx_session, determine_optimal_device, ModelSource, SessionConfig,
};

/// Configuration trait for models that can be processed generically
pub trait ModelConfig {
    fn base(&self) -> &BaseModelConfig;
}

/// Result trait for model outputs that can be handled generically
pub trait ModelResult {
    /// Get a human-readable summary of the results for logging
    fn result_summary(&self) -> String;

    /// Get the processing time in milliseconds
    fn processing_time_ms(&self) -> f64;
}

/// Core trait that all models must implement
pub trait ModelProcessor {
    /// Configuration type for this model
    type Config: ModelConfig;

    /// Result type returned by this model
    type Result: ModelResult;

    /// Create an ONNX session for this model
    fn create_session(config: &Self::Config) -> Result<Session>;

    /// Process a single image through the complete pipeline
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result>;
}

/// Helper function to create session with common pattern
pub fn create_session_with_source(
    config: &impl ModelConfig,
    model_source: ModelSource,
) -> Result<Session> {
    // Determine optimal device
    let device_selection = determine_optimal_device(&config.base().device);

    // Create session using unified ONNX session management
    let session_config = SessionConfig {
        device: &device_selection.device,
    };

    let (session, _load_time) = create_onnx_session(model_source, &session_config)?;

    Ok(session)
}

/// Generic batch processing function that works with any model
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    // Create image input configuration from model config
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);

    // Collect images from input sources
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    if image_files.is_empty() {
        log::warn!("No valid images found to process");
        return Ok(0);
    }

    log::info!("🎯 Found {} image(s) to process", image_files.len());

    // Create session
    let mut session = P::create_session(&config)?;

    // Process each image
    let mut successful_count = 0;
    let total_start = std::time::Instant::now();

    for (index, image_path) in image_files.iter().enumerate() {
        match P::process_single_image(&mut session, image_path, &config) {
            Ok(result) => {
                successful_count += 1;
                log::info!(
                    "✅ Processed {} ({}/{}) in {:.1}ms",
                    image_path.display(),
                    index + 1,
                    image_files.len(),
                    result.processing_time_ms()
                );
                log::debug!("📊 {}", result.result_summary());
            }
            Err(e) => {
                if config.base().strict {
                    return Err(e);
                } else {
                    log::warn!(
                        "⚠️  Failed to process {} ({}/{}): {}",
                        image_path.display(),
                        index + 1,
                        image_files.len(),
                        e
                    );
                }
            }
        }
    }

    let total_time = total_start.elapsed();

    if successful_count > 0 {
        log::info!(
            "✅ Processed {} images in {:.1}s",
            successful_count,
            total_time.as_secs_f64()
        );
    }

    if successful_count < image_files.len() {
        log::warn!(
            "⚠️  {} of {} images failed to process",
            image_files.len() - successful_count,
            image_files.len()
        );
    }

    Ok(successful_count)
}
