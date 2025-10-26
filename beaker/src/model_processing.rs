//! Model processing framework providing unified interface for all models.
//!
//! This module defines the core `ModelProcessor` trait that all models must implement,
//! along with generic processing functions that handle the common patterns across
//! different model types.

use crate::progress::{add_progress_bar, remove_progress_bar};
use crate::quality_processing::QualityResult;
use anyhow::Result;
use colored::Colorize;
use log::warn;
use ort::session::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use crate::config::BaseModelConfig;
use crate::image_input::{collect_images_from_sources, ImageInputConfig};
use crate::onnx_session::{create_onnx_session, ModelSource, SessionConfig};

use crate::color_utils::maybe_dim_stderr;
use crate::shared_metadata::{InputProcessing, SystemInfo};

/// Progress events emitted during processing
#[derive(Debug, Clone)]
pub enum ProcessingEvent {
    /// Processing started for an image
    ImageStart {
        path: PathBuf,
        index: usize,
        total: usize,
        stage: ProcessingStage,
    },

    /// Image processing completed (success or failure)
    ImageComplete {
        path: PathBuf,
        index: usize,
        result: ProcessingResultInfo,
    },

    /// Stage transition (quality → detection)
    StageChange {
        stage: ProcessingStage,
        images_total: usize,
    },

    /// Overall progress update
    Progress {
        completed: usize,
        total: usize,
        stage: ProcessingStage,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingStage {
    Quality,
    Detection,
}

#[derive(Debug, Clone)]
pub enum ProcessingResultInfo {
    Success {
        detections_count: usize,
        good_count: usize,
        bad_count: usize,
        unknown_count: usize,
        processing_time_ms: f64,
    },
    Error {
        error_message: String,
    },
}

/// Configuration trait for models that can be processed generically
pub trait ModelConfig: std::any::Any {
    fn base(&self) -> &BaseModelConfig;

    /// Get the tool name for this config (e.g., "detect", "cutout")
    fn tool_name(&self) -> &'static str;
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

    /// Get file I/O timing information
    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming;

    /// Get mask entry for cutout results (only applicable for cutout tools)
    fn get_mask_entry(&self) -> Option<crate::mask_encoding::MaskEntry> {
        None
    }

    fn get_quality_result(&self) -> Option<QualityResult> {
        None
    }
}

/// Core trait that all models must implement
pub trait ModelProcessor {
    /// Configuration type for this model
    type Config: ModelConfig;

    /// Result type returned by this model
    type Result: ModelResult;

    /// Get the model source for loading the ONNX model
    /// Returns (ModelSource, OnnxCacheStats) where OnnxCacheStats emerge from cache operations
    fn get_model_source<'a>(
        config: &Self::Config,
    ) -> Result<(
        ModelSource<'a>,
        Option<crate::shared_metadata::OnnxCacheStats>,
    )>;

    /// Process a single image through the complete pipeline
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
        output_manager: &crate::output_manager::OutputManager,
    ) -> Result<Self::Result>;

    /// Get serializable configuration for metadata
    fn serialize_config(config: &Self::Config) -> Result<toml::Value>;
}

pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    run_model_processing_with_options::<P>(config, None, None, None)
}

pub fn run_model_processing_with_options<P: ModelProcessor>(
    config: P::Config,
    progress_tx: Option<Sender<ProcessingEvent>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    stage: Option<ProcessingStage>,
) -> Result<usize> {
    match run_model_processing_with_quality_outputs::<P>(config, progress_tx, cancel_flag, stage) {
        Ok((successful_count, _quality_results)) => {
            // Process successful results
            // Do something with _quality_results
            Ok(successful_count)
        }
        Err(errors) => {
            // Handle errors
            Err(errors)
        }
    }
}
/// Generic batch processing function that works with any model
pub fn run_model_processing_with_quality_outputs<P: ModelProcessor>(
    config: P::Config,
    progress_tx: Option<Sender<ProcessingEvent>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    stage: Option<ProcessingStage>,
) -> Result<(usize, HashMap<String, QualityResult>)> {
    use crate::onnx_session::determine_optimal_device;
    use chrono::Utc;
    use std::time::Instant;

    // Hashmap from image path to local quality grid
    let mut quality_results: HashMap<String, QualityResult> = HashMap::new();

    let start_timestamp = Utc::now();
    let total_processing_start = Instant::now();

    // Collect command line for metadata
    let command_line: Vec<String> = std::env::args().collect();

    // Create image input configuration from model config
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);

    // Collect images from input sources
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    if image_files.is_empty() {
        log::warn!("No valid images found to process");
        return Ok((0, quality_results));
    }

    if image_files.len() > 1 {
        log::info!(
            "{} Found {} images to process",
            crate::color_utils::symbols::resources_found(),
            image_files.len()
        );
    }

    let tool = config.tool_name();
    // Collect device information for metadata
    let device_selection = determine_optimal_device(&config.base().device, tool);
    let device_selected = device_selection.device.clone();
    let device_selection_reason = device_selection.reason.clone();

    // Create session with timing
    let session_start = Instant::now();
    // It sometimes takes a while to load the model, so we'll use a spinner when on TTY
    // it will show "model loading" while the model is being loaded, then switch to "model loaded" when done
    let spinner = indicatif::ProgressBar::new_spinner();
    add_progress_bar(spinner.clone());
    spinner.set_message(" Loading model...");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let (model_source, onnx_cache_stats) = P::get_model_source(&config)?;

    let session_config = SessionConfig {
        device: &device_selected,
    };
    let (mut session, model_info, coreml_cache_stats) =
        create_onnx_session(model_source, &session_config)?;

    spinner.finish_and_clear();
    remove_progress_bar(&spinner);

    let model_load_time_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    let megabytes_str = format!("{} MB", model_info.model_size_bytes / 1_048_576);

    let pretty_device_name = if device_selected == "coreml" {
        "CoreML".to_string()
    } else {
        "CPU".to_string()
    };

    let timing_str: String = maybe_dim_stderr(&format!("in {model_load_time_ms:.0}ms"));
    log::info!(
        "{} Model loaded ({}, {}) {}",
        crate::color_utils::symbols::model_loaded(),
        megabytes_str,
        pretty_device_name,
        timing_str
    );

    let system = SystemInfo {
        device_requested: Some(config.base().device.clone()),
        device_selected: Some(device_selected.to_string()),
        device_selection_reason: Some(device_selection_reason.to_string()),
        execution_providers: model_info.execution_providers,
        model_source: Some(model_info.model_source),
        model_path: model_info.model_path.clone(),
        model_size_bytes: Some(model_info.model_size_bytes.try_into().unwrap()),
        model_load_time_ms: Some(model_load_time_ms),
        model_checksum: Some(model_info.model_checksum),
        onnx_cache: onnx_cache_stats,
        coreml_cache: coreml_cache_stats,
    };

    // Pre-check for output path collisions if output_dir is set
    if config.base().output_dir.is_some() && !config.base().force {
        let mut seen_stems = std::collections::HashMap::new();
        for image_path in image_files.keys() {
            let stem = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");

            if let Some(first_path) = seen_stems.get(stem) {
                // Error on collision unless --force is used
                return Err(anyhow::anyhow!(
                    "Output path collision detected between:\n\
                     1. {}\n\
                     2. {}\n\
                     \n\
                     Both would generate the same output filename: {}_{}.jpg\n\
                     \n\
                     This happens when processing multiple files with the same basename\n\
                     from different directories to a single output directory.\n\
                     \n\
                     Solutions:\n\
                     1. Use --force flag to allow overwriting\n\
                     2. Process files separately\n\
                     3. Rename input files to have unique basenames\n\
                     4. Don't use --output-dir (files will go to their source directories)",
                    first_path,
                    image_path.display(),
                    stem,
                    config.tool_name()
                ));
            }
            seen_stems.insert(stem, image_path.display().to_string());
        }
    }

    // Process each image and collect results
    let mut successful_count = 0;
    let mut failed_count = 0;

    // Create progress bar for batch processing if appropriate
    let progress_bar: Option<indicatif::ProgressBar> =
        crate::color_utils::progress::create_batch_progress_bar(image_files.len());

    // Create vector to contain failed image paths
    let mut failed_images = Vec::new();

    for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Processing cancelled by user");
                return Ok((successful_count, quality_results));
            }
        }

        // Emit ImageStart event
        if let Some(ref tx) = progress_tx {
            let current_stage = stage.unwrap_or(ProcessingStage::Detection);
            let _ = tx.send(ProcessingEvent::ImageStart {
                path: image_path.clone(),
                index,
                total: image_files.len(),
                stage: current_stage,
            });
        }

        // Create input processing info
        let input = InputProcessing {
            image_path: image_path.to_string_lossy().to_string(),
            source: source_string.to_string(),
            source_type: source_type.to_string(),
            strict_mode: config.base().strict,
        };

        // Create OutputManager for this image
        let output_manager = crate::output_manager::OutputManager::new(&config, image_path);

        if let Some(ref pb) = progress_bar {
            // we will style the filename with bold:
            let filename = crate::color_utils::maybe_color_stderr(
                &image_path.file_name().unwrap_or_default().to_string_lossy(),
                |s| s.bold(),
            );
            pb.set_prefix(format!(
                "[{}/{}] Processing {}",
                index + 1,
                image_files.len(),
                filename,
            ));
            // we are using msg as ETA
            if index > 0 {
                // Calculate ETA based on elapsed time and number of processed images
                // This is a simple linear estimate, not perfect but works for most cases
                let elapsed = pb.elapsed().as_secs_f64();
                let total = image_files.len() as f64;
                let processed = index as f64; // no +1 since we haven't processed this one yet
                let eta = (elapsed / processed) * (total - processed);
                pb.set_message(format!("ETA: {eta:.1}s"));
            }
        }
        match P::process_single_image(&mut session, image_path, &config, &output_manager) {
            Ok(result) => {
                successful_count += 1;
                if let Some(quality) = result.get_quality_result() {
                    quality_results.insert(image_path.to_string_lossy().to_string(), quality);
                };

                // Emit ImageComplete success event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: ProcessingResultInfo::Success {
                            detections_count: 0, // Will be populated by detection stage
                            good_count: 0,
                            bad_count: 0,
                            unknown_count: 0,
                            processing_time_ms: result.processing_time_ms(),
                        },
                    });
                }

                if !config.base().skip_metadata {
                    save_enhanced_metadata_for_file::<P>(
                        &result,
                        &config,
                        &command_line,
                        system.clone(),
                        input.clone(),
                        start_timestamp,
                        &output_manager,
                    )?;
                }
                if progress_bar.is_none() {
                    let val = result.processing_time_ms();
                    let timing_str: String = maybe_dim_stderr(&format!("in {val:.0} ms"));
                    // Log comprehensive processing result for single files or non-interactive
                    log::info!(
                        "{} Processed {} ({}/{}) {} {}",
                        crate::color_utils::symbols::completed_successfully(),
                        image_path.display(),
                        index + 1,
                        image_files.len(),
                        timing_str,
                        result.output_summary()
                    );
                }
            }
            Err(e) => {
                failed_count += 1;

                failed_images.push(image_path.to_str().unwrap_or_default().to_string());

                // Emit ImageComplete error event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: ProcessingResultInfo::Error {
                            error_message: e.to_string(),
                        },
                    });
                }

                let colored_error = crate::color_utils::colors::error_level(&e.to_string());
                warn!(
                    "{} Failed to process {}:\n            {}",
                    crate::color_utils::symbols::warning(),
                    image_path.display(),
                    colored_error
                );
            }
        }
        if let Some(ref pb) = progress_bar {
            pb.inc(1);
        }
    }

    // Finish progress bar if it exists
    if let Some(ref pb) = progress_bar {
        pb.finish_and_clear();
        remove_progress_bar(pb);
    }
    let total_processing_time = total_processing_start.elapsed();

    // The case n=1 doesn't use a progress bar and already got a direct log message
    if image_files.len() > 1 {
        if failed_count > 0 {
            let timing_str = maybe_dim_stderr(&format!(
                "({:.1}s, {:.0} ms/success)",
                total_processing_time.as_millis() as f64 / 1000.0,
                total_processing_time.as_millis() as f64 / successful_count as f64
            ));
            log::info!(
                "{} Processed {} images with {} successes and {} failures {}",
                crate::color_utils::symbols::completed_partially_successfully(),
                successful_count + failed_count,
                successful_count,
                failed_count,
                timing_str
            );
        } else {
            let timing_str = maybe_dim_stderr(&format!(
                "({:.1}s, {:.0} ms/image)",
                total_processing_time.as_millis() as f64 / 1000.0,
                total_processing_time.as_millis() as f64 / successful_count as f64
            ));
            log::info!(
                "{} Processed {} images successfully {}",
                crate::color_utils::symbols::completed_successfully(),
                successful_count,
                timing_str
            );
        }
    }

    // If strict mode is enabled, fail if any images failed
    if config.base().strict && failed_count > 0 {
        return Err(anyhow::anyhow!(
            "{} image(s) failed to process (without `--permissive` flag)",
            failed_count
        ));
    }
    Ok((successful_count, quality_results))
}

/// Save enhanced metadata for a single processed file
#[allow(clippy::too_many_arguments)]
fn save_enhanced_metadata_for_file<P: ModelProcessor>(
    result: &P::Result,
    config: &P::Config,
    command_line: &[String],
    system: SystemInfo,
    input: InputProcessing,
    start_timestamp: chrono::DateTime<chrono::Utc>,
    output_manager: &crate::output_manager::OutputManager,
) -> Result<()> {
    use crate::shared_metadata::{
        CutoutSections, DetectSections, ExecutionContext, QualitySections,
    };

    // Create execution context
    let execution = ExecutionContext {
        timestamp: Some(start_timestamp),
        beaker_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command_line: Some(command_line.to_vec()),
        exit_code: Some(0),
        model_processing_time_ms: Some(result.processing_time_ms()),
        file_io: Some(result.get_io_timing()),
        beaker_env_vars: crate::shared_metadata::collect_beaker_env_vars(),
    };

    // Get core results and config
    let core_results = result.core_results()?;
    let config_value = P::serialize_config(config)?;

    // Create the appropriate sections based on tool type
    match result.tool_name() {
        "detect" => {
            let detect_sections = DetectSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
            };
            output_manager.save_complete_metadata(Some(detect_sections), None, None)?;
        }
        "cutout" => {
            let cutout_sections = CutoutSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
                mask: result.get_mask_entry(),
            };
            output_manager.save_complete_metadata(None, Some(cutout_sections), None)?;
        }
        "quality" => {
            let quality_sections = QualitySections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
            };
            output_manager.save_complete_metadata(None, None, Some(quality_sections))?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown tool name: {}", result.tool_name()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_event_types() {
        // Test that we can create processing events
        let event = ProcessingEvent::ImageStart {
            path: PathBuf::from("test.jpg"),
            index: 0,
            total: 10,
            stage: ProcessingStage::Quality,
        };

        match event {
            ProcessingEvent::ImageStart { path, index, total, stage } => {
                assert_eq!(path, PathBuf::from("test.jpg"));
                assert_eq!(index, 0);
                assert_eq!(total, 10);
                assert_eq!(stage, ProcessingStage::Quality);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_processing_stage_equality() {
        assert_eq!(ProcessingStage::Quality, ProcessingStage::Quality);
        assert_eq!(ProcessingStage::Detection, ProcessingStage::Detection);
        assert_ne!(ProcessingStage::Quality, ProcessingStage::Detection);
    }

    #[test]
    fn test_processing_result_success() {
        let result = ProcessingResultInfo::Success {
            detections_count: 5,
            good_count: 3,
            bad_count: 1,
            unknown_count: 1,
            processing_time_ms: 123.45,
        };

        match result {
            ProcessingResultInfo::Success {
                detections_count,
                good_count,
                bad_count,
                unknown_count,
                processing_time_ms
            } => {
                assert_eq!(detections_count, 5);
                assert_eq!(good_count, 3);
                assert_eq!(bad_count, 1);
                assert_eq!(unknown_count, 1);
                assert_eq!(processing_time_ms, 123.45);
            }
            _ => panic!("Wrong result type"),
        }
    }

    #[test]
    fn test_processing_result_error() {
        let result = ProcessingResultInfo::Error {
            error_message: "Test error".to_string(),
        };

        match result {
            ProcessingResultInfo::Error { error_message } => {
                assert_eq!(error_message, "Test error");
            }
            _ => panic!("Wrong result type"),
        }
    }

    #[test]
    fn test_cancellation_flag() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let cancel_flag = Arc::new(AtomicBool::new(false));
        assert!(!cancel_flag.load(Ordering::Relaxed));

        cancel_flag.store(true, Ordering::Relaxed);
        assert!(cancel_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_channel_send_events() {
        use std::sync::mpsc::channel;

        let (tx, rx) = channel();

        // Send an event
        let event = ProcessingEvent::ImageStart {
            path: PathBuf::from("test.jpg"),
            index: 0,
            total: 1,
            stage: ProcessingStage::Quality,
        };

        tx.send(event.clone()).unwrap();

        // Receive the event
        let received = rx.recv().unwrap();

        match received {
            ProcessingEvent::ImageStart { path, .. } => {
                assert_eq!(path, PathBuf::from("test.jpg"));
            }
            _ => panic!("Wrong event type received"),
        }
    }

    #[test]
    fn test_stage_change_event() {
        let event = ProcessingEvent::StageChange {
            stage: ProcessingStage::Detection,
            images_total: 42,
        };

        match event {
            ProcessingEvent::StageChange { stage, images_total } => {
                assert_eq!(stage, ProcessingStage::Detection);
                assert_eq!(images_total, 42);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_progress_event() {
        let event = ProcessingEvent::Progress {
            completed: 5,
            total: 10,
            stage: ProcessingStage::Quality,
        };

        match event {
            ProcessingEvent::Progress { completed, total, stage } => {
                assert_eq!(completed, 5);
                assert_eq!(total, 10);
                assert_eq!(stage, ProcessingStage::Quality);
            }
            _ => panic!("Wrong event type"),
        }
    }
}
