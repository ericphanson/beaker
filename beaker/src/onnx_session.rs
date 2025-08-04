use anyhow::Result;
use log::Level;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    logging::LogLevel,
    session::Session,
};
use serde::Serialize;
use std::fs;

// Import model cache function
use crate::model_cache::get_coreml_cache_dir;

fn log_level_from_ort(level: LogLevel) -> Level {
    match level {
        LogLevel::Verbose => Level::Trace,
        LogLevel::Info => Level::Trace,
        LogLevel::Warning => Level::Debug,
        LogLevel::Error => Level::Info,
        LogLevel::Fatal => Level::Error,
    }
}
fn ort_level_from_log(level: Level) -> LogLevel {
    match level {
        // we skip mapping to info because ONNX's info is so verbose
        // that it is more like debug or trace
        Level::Trace => LogLevel::Verbose,
        Level::Debug => LogLevel::Warning,
        Level::Info => LogLevel::Error,
        Level::Warn => LogLevel::Error,
        Level::Error => LogLevel::Fatal,
    }
}

/// Configuration for creating ONNX sessions
pub struct SessionConfig<'a> {
    pub device: &'a str,
}

/// Model source for loading ONNX models
pub enum ModelSource<'a> {
    EmbeddedBytes(&'a [u8]),
    FilePath(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub model_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    pub model_size_bytes: usize,
    pub description: String,
    pub execution_providers: Vec<String>,
}

/// Device selection result
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    pub device: String,
    #[allow(dead_code)]
    pub reason: String,
}

/// Determine optimal device based on user preference
pub fn determine_optimal_device(requested_device: &str) -> DeviceSelection {
    match requested_device {
        "auto" => {
            // For auto, prefer CoreML if available, otherwise CPU
            let coreml = CoreMLExecutionProvider::default();
            match coreml.is_available() {
                Ok(true) => {
                    log::info!("🖥️  Using CoreML execution provider");
                    DeviceSelection {
                        device: "coreml".to_string(),
                        reason: "Auto-selected CoreML (available)".to_string(),
                    }
                }
                _ => {
                    log::info!("🖥️  Using CPU execution provider");
                    DeviceSelection {
                        device: "cpu".to_string(),
                        reason: "Auto-selected CPU (CoreML not available)".to_string(),
                    }
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
) -> Result<(Session, ModelInfo)> {
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

    // Choose the ORT log level based on what is enabled for us
    let ort_log_level = [
        Level::Trace,
        Level::Debug,
        Level::Info,
        Level::Warn,
        Level::Error,
    ]
    .into_iter()
    .find(|&lvl| log::log_enabled!(lvl))
    .map(ort_level_from_log)
    .unwrap_or(LogLevel::Fatal);

    let build_session = |bytes: &[u8]| {
        Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_logger(Box::new(|level, _, _, _, msg| {
                // we will just relog to our standard logger with `log!`
                // after choosing the appropriate log level
                let log_level = log_level_from_ort(level);
                log::log!(log_level, "[onnx] {msg}")
            }))
            .map_err(|e| anyhow::anyhow!("Failed to set logger: {}", e))?
            .with_log_level(ort_log_level)
            .map_err(|e| anyhow::anyhow!("Failed to set log level: {}", e))?
            .with_execution_providers(execution_providers)
            .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {}", e))?
            .commit_from_memory(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load model from memory: {}", e))
    };

    let (session, model_info) = match model_source {
        ModelSource::EmbeddedBytes(bytes) => {
            let session = build_session(bytes)?;
            let info: ModelInfo = ModelInfo {
                model_source: "embedded".to_string(),
                model_path: None,
                model_size_bytes: bytes.len(),
                description: format!("embedded ONNX model ({} bytes)", bytes.len()),
                execution_providers: ep_names,
            };
            (session, info)
        }
        ModelSource::FilePath(path) => {
            let model_bytes =
                fs::read(&path).map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;
            let session = build_session(&model_bytes)?;
            let info = ModelInfo {
                model_source: "file".to_string(),
                model_path: Some(path.clone()),
                model_size_bytes: model_bytes.len(),
                description: format!("ONNX model from {path}"),
                execution_providers: ep_names,
            };
            (session, info)
        }
    };

    // Log execution provider information
    log::debug!(
        "⚙️  Execution providers registered: {}",
        model_info.execution_providers.join(" -> ")
    );

    Ok((session, model_info))
}
