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
use crate::onnx_session::{create_onnx_session, ModelSource, SessionConfig};

use crate::shared_metadata::{InputProcessing, SystemInfo};
/// Configuration trait for models that can be processed generically
pub trait ModelConfig {
    fn base(&self) -> &BaseModelConfig;
}

/// Result trait for model outputs that can be handled generically
pub trait ModelResult {
    /// Get the processing time in milliseconds
    fn processing_time_ms(&self) -> f64;

    /// Get the tool name for metadata sections (e.g., "head", "cutout")
    fn tool_name(&self) -> &'static str;

    /// Get the serializable core results for the main tool section
    fn core_results(&self) -> Result<toml::Value>;

    /// Get a summary of all output files created
    fn output_summary(&self) -> String;
}

/// Core trait that all models must implement
pub trait ModelProcessor {
    /// Configuration type for this model
    type Config: ModelConfig;

    /// Result type returned by this model
    type Result: ModelResult;

    /// Get the model source for loading the ONNX model
    fn get_model_source<'a>() -> Result<ModelSource<'a>>;

    /// Process a single image through the complete pipeline
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result>;

    /// Get serializable configuration for metadata
    fn serialize_config(config: &Self::Config) -> Result<toml::Value>;
}

/// Generic batch processing function that works with any model
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    use crate::onnx_session::determine_optimal_device;
    use chrono::Utc;
    use std::time::Instant;

    let start_timestamp = Utc::now();

    // Collect command line for metadata
    let command_line: Vec<String> = std::env::args().collect();

    // Create image input configuration from model config
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);

    // Collect images from input sources
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    if image_files.is_empty() {
        log::warn!("No valid images found to process");
        return Ok(0);
    }

    if image_files.len() > 1 {
        log::info!(
            "{} Found {} images to process",
            crate::color_utils::symbols::resources_found(),
            image_files.len()
        );
    }

    // Create progress bar for batch processing if appropriate
    let progress_bar = crate::color_utils::progress::create_batch_progress_bar(image_files.len());

    // Collect device information for metadata
    let device_selection = determine_optimal_device(&config.base().device);
    let device_selected = device_selection.device.clone();
    let device_selection_reason = device_selection.reason.clone();

    // Create session with timing
    let session_start = Instant::now();
    let model_source = P::get_model_source();

    let session_config = SessionConfig {
        device: &device_selected,
    };

    let (mut session, model_info) = create_onnx_session(model_source.unwrap(), &session_config)?;

    let model_load_time_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    let megabytes_str = format!("{} MB", model_info.model_size_bytes / 1_048_576);

    let pretty_device_name = if device_selected == "coreml" {
        "CoreML".to_string()
    } else {
        "CPU".to_string()
    };

    log::info!(
        "{} Model loaded ({}, {}) in {:.3}ms",
        crate::color_utils::symbols::model_loaded(),
        megabytes_str,
        pretty_device_name,
        model_load_time_ms
    );

    let system = SystemInfo {
        device_requested: Some(config.base().device.clone()),
        device_selected: Some(device_selected.to_string()),
        device_selection_reason: Some(device_selection_reason.to_string()),
        execution_providers: model_info.execution_providers,
        model_source: Some(model_info.model_source),
        model_path: model_info.model_path,
        model_size_bytes: Some(model_info.model_size_bytes.try_into().unwrap()),
        model_load_time_ms: Some(model_load_time_ms),
    };

    // Collect source type information
    let source_types: Vec<String> = config
        .base()
        .sources
        .iter()
        .map(|source| {
            let path = std::path::Path::new(source);
            if path.is_dir() {
                "directory".to_string()
            } else if source.contains('*') || source.contains('?') {
                "glob".to_string()
            } else {
                "file".to_string()
            }
        })
        .collect();

    // Process each image and collect results
    let mut successful_count = 0;
    let mut failed_count = 0;

    // Create input processing info
    let input = InputProcessing {
        sources: config.base().sources.to_vec(),
        source_types: source_types.to_vec(),
        strict_mode: config.base().strict,
    };

    // Create vector to contain failed image paths
    let mut failed_images = Vec::new();
    for (index, image_path) in image_files.iter().enumerate() {
        match P::process_single_image(&mut session, image_path, &config) {
            Ok(result) => {
                successful_count += 1;

                if !config.base().skip_metadata {
                    save_enhanced_metadata_for_file::<P>(
                        &result,
                        &config,
                        image_path,
                        &command_line,
                        system.clone(),
                        input.clone(),
                        start_timestamp,
                    )?;
                }

                // Use progress bar or log message based on what's available
                if let Some(ref pb) = progress_bar {
                    pb.inc(1);
                    pb.set_message(format!(
                        "Processed {}",
                        image_path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                } else {
                    // Log comprehensive processing result for single files or non-interactive
                    log::info!(
                        "{} Processed {} ({}/{}) in {:.1}ms {}",
                        crate::color_utils::symbols::completed_successfully(),
                        image_path.display(),
                        index + 1,
                        image_files.len(),
                        result.processing_time_ms(),
                        result.output_summary()
                    );
                }
            }
            Err(e) => {
                failed_count += 1;
                if let Some(ref pb) = progress_bar {
                    pb.inc(1);
                    pb.set_message(format!(
                        "Failed: {}",
                        image_path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }

                failed_images.push(image_path.to_str().unwrap_or_default().to_string());

                if config.base().strict {
                    if let Some(pb) = progress_bar {
                        pb.finish_and_clear();
                    }
                    return Err(e);
                } else {
                    log::warn!(
                        "{} Failed to process {} ({}/{}): {}",
                        crate::color_utils::symbols::warning(),
                        image_path.display(),
                        index + 1,
                        image_files.len(),
                        e
                    );
                }
            }
        }
    }

    // Finish progress bar if it exists
    if let Some(pb) = progress_bar {
        pb.finish_with_message("Processing complete");
    }
    // if failed_images > 20, we will truncate and add an ellipsis
    let failed_images_str = if failed_images.len() > 20 {
        format!(
            "{} and {} more...",
            failed_images[..20].join(", "),
            failed_images.len() - 20
        )
    } else {
        failed_images.join(", ")
    };
    if failed_count > 0 {
        log::warn!(
            "{} {} images failed to process: {}",
            crate::color_utils::symbols::warning(),
            failed_count,
            failed_images_str
        );
    }

    if image_files.len() > 1 {
        if failed_count > 0 {
            log::info!(
                "{} Processed {} images with {} successes and {} failures",
                crate::color_utils::symbols::completed_partially_successfully(),
                successful_count + failed_count,
                successful_count,
                failed_count
            );
        } else {
            log::info!(
                "{} Processed {} images successfully",
                crate::color_utils::symbols::completed_successfully(),
                successful_count
            );
        }
    }

    Ok(successful_count)
}

/// Save enhanced metadata for a single processed file
#[allow(clippy::too_many_arguments)]
fn save_enhanced_metadata_for_file<P: ModelProcessor>(
    result: &P::Result,
    config: &P::Config,
    image_path: &std::path::Path,
    command_line: &[String],
    system: SystemInfo,
    input: InputProcessing,
    start_timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    use crate::output_manager::OutputManager;
    use crate::shared_metadata::{CutoutSections, ExecutionContext, HeadSections};

    let output_manager = OutputManager::new(config, image_path);

    // Create execution context
    let execution = ExecutionContext {
        timestamp: Some(start_timestamp),
        beaker_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command_line: Some(command_line.to_vec()),
        exit_code: Some(0),
        model_processing_time_ms: Some(result.processing_time_ms()),
    };

    // Get core results and config
    let core_results = result.core_results()?;
    let config_value = P::serialize_config(config)?;

    // Create the appropriate sections based on tool type
    match result.tool_name() {
        "head" => {
            let head_sections = HeadSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
            };
            output_manager.save_complete_metadata(Some(head_sections), None)?;
        }
        "cutout" => {
            let cutout_sections = CutoutSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
            };
            output_manager.save_complete_metadata(None, Some(cutout_sections))?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown tool name: {}", result.tool_name()));
        }
    }

    Ok(())
}
