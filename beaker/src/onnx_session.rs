use anyhow::Result;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    logging::LogLevel,
    session::Session,
};
use std::fs;
use std::time::Instant;

// Import model cache function
use crate::model_cache::get_coreml_cache_dir;

/// Configuration for creating ONNX sessions
pub struct SessionConfig<'a> {
    pub device: &'a str,
}

/// Model source for loading ONNX models
pub enum ModelSource<'a> {
    EmbeddedBytes(&'a [u8]),
    FilePath(&'a str),
}

/// Device selection result
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    pub device: String,
    pub reason: String,
}

/// Determine optimal device based on number of images and user preference
pub fn determine_optimal_device(requested_device: &str, num_images: usize) -> DeviceSelection {
    const COREML_THRESHOLD: usize = 3; // Use CoreML for 3+ images

    match requested_device {
        "auto" => {
            if num_images >= COREML_THRESHOLD {
                // Check if CoreML is available
                let coreml = CoreMLExecutionProvider::default();
                match coreml.is_available() {
                    Ok(true) => {
                        let reason = format!("Processing {num_images} images - using CoreML for better batch performance");
                        log::info!("📊 {reason}");
                        DeviceSelection {
                            device: "coreml".to_string(),
                            reason,
                        }
                    }
                    _ => {
                        let reason = format!(
                            "Processing {num_images} images - CoreML not available, using CPU"
                        );
                        log::info!("📊 {reason}");
                        DeviceSelection {
                            device: "cpu".to_string(),
                            reason,
                        }
                    }
                }
            } else {
                let reason = format!("Processing {num_images} images - using CPU for small batch");
                log::info!("📊 {reason}");
                DeviceSelection {
                    device: "cpu".to_string(),
                    reason,
                }
            }
        }
        other => DeviceSelection {
            device: other.to_string(),
            reason: format!("User explicitly chose {other}"),
        },
    }
}

/// Create an ONNX Runtime session with the specified configuration
pub fn create_onnx_session(
    model_source: ModelSource,
    config: &SessionConfig,
) -> Result<(Session, f64)> {
    // Set up CoreML cache directory if using CoreML
    let coreml_cache_dir = if config.device == "coreml" {
        match get_coreml_cache_dir() {
            Ok(cache_dir) => {
                // Create the cache directory if it doesn't exist
                if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                    log::warn!("⚠️  Failed to create CoreML cache directory: {e}");
                    None
                } else {
                    log::debug!("📂 Using CoreML cache directory: {}", cache_dir.display());
                    Some(cache_dir)
                }
            }
            Err(e) => {
                log::warn!("⚠️  Failed to get CoreML cache directory: {e}");
                None
            }
        }
    } else {
        None
    };

    // Create execution providers (simplified from cutout pattern)
    let execution_providers = match config.device {
        "coreml" => match CoreMLExecutionProvider::default().is_available() {
            Ok(true) => {
                let coreml_provider = if let Some(cache_dir) = &coreml_cache_dir {
                    if let Some(cache_path_str) = cache_dir.to_str() {
                        log::debug!("🗂️  Configuring CoreML model cache: {cache_path_str}");
                        CoreMLExecutionProvider::default().with_model_cache_dir(cache_path_str)
                    } else {
                        CoreMLExecutionProvider::default()
                    }
                } else {
                    CoreMLExecutionProvider::default()
                };

                vec![
                    coreml_provider.build(),
                    CPUExecutionProvider::default().build(),
                ]
            }
            _ => {
                log::warn!("⚠️  CoreML not available, falling back to CPU");
                vec![CPUExecutionProvider::default().build()]
            }
        },
        "cpu" => {
            log::info!("🖥️  Using CPU execution provider");
            vec![CPUExecutionProvider::default().build()]
        }
        _ => {
            log::warn!("⚠️  Unknown device '{}', using CPU", config.device);
            vec![CPUExecutionProvider::default().build()]
        }
    };

    // Store EP info for logging before moving the vector
    let ep_names: Vec<String> = execution_providers
        .iter()
        .map(|ep| format!("{ep:?}"))
        .collect();

    // Set log level to suppress CoreML warnings unless debug logging is enabled
    let log_level = if log::log_enabled!(log::Level::Debug) {
        LogLevel::Warning // Show warnings in debug mode
    } else {
        LogLevel::Error // Suppress warnings in normal mode
    };

    // Load the model using ORT v2 API
    let session_start = Instant::now();
    let session = match model_source {
        ModelSource::EmbeddedBytes(bytes) => Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_log_level(log_level)
            .map_err(|e| anyhow::anyhow!("Failed to set log level: {}", e))?
            .with_execution_providers(execution_providers)
            .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {}", e))?
            .commit_from_memory(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load model from memory: {}", e))?,
        ModelSource::FilePath(path) => {
            let model_bytes =
                fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;
            Session::builder()
                .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                .with_log_level(log_level)
                .map_err(|e| anyhow::anyhow!("Failed to set log level: {}", e))?
                .with_execution_providers(execution_providers)
                .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {}", e))?
                .commit_from_memory(&model_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to load model from memory: {}", e))?
        }
    };
    let session_load_time = session_start.elapsed();

    let model_info = match model_source {
        ModelSource::EmbeddedBytes(bytes) => format!("embedded ONNX model ({} bytes)", bytes.len()),
        ModelSource::FilePath(path) => format!("ONNX model from {path}"),
    };

    log::info!(
        "🤖 Loaded {} in {:.3}ms",
        model_info,
        session_load_time.as_secs_f64() * 1000.0
    );

    // Log execution provider information
    log::debug!(
        "⚙️  Execution providers registered: {}",
        ep_names.join(" -> ")
    );

    Ok((session, session_load_time.as_secs_f64() * 1000.0))
}
