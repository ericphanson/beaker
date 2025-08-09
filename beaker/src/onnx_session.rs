use crate::cache_common;
use crate::color_utils::symbols;
use anyhow::Result;
use log::Level;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    logging::LogLevel,
    session::Session,
};
use serde::Serialize;
use std::fs;

/// Generate stable CoreML cache directory based on model content and ORT version
fn get_stable_coreml_cache_dir(model_bytes: &[u8]) -> Result<std::path::PathBuf> {
    let base_dir = crate::model_access::get_coreml_cache_dir()?;

    // Create MD5 hash of model content using shared function
    let model_hash = cache_common::calculate_md5_bytes(model_bytes);

    // Get ORT version for cache versioning
    let ort_version = env!("CARGO_PKG_VERSION"); // Use beaker version as proxy

    // Create stable cache key: model_hash + ort_version
    let cache_key = format!("{}_{}", &model_hash[..8], ort_version.replace('.', "_"));
    let stable_dir = base_dir.join(cache_key);

    log::debug!(
        "🔑 Generated stable CoreML cache key: {}",
        stable_dir.display()
    );
    Ok(stable_dir)
}

/// Generate a unique CoreML cache directory to avoid conflicts (fallback only)
fn get_unique_coreml_cache_dir() -> Result<std::path::PathBuf> {
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    let base_dir = crate::model_access::get_coreml_cache_dir()?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let pid = process::id();

    // Create a unique subdirectory using timestamp and process ID
    let unique_dir = base_dir.join(format!("session_{pid}_{timestamp}"));

    log::debug!(
        "🆔 Generated unique CoreML cache directory: {}",
        unique_dir.display()
    );
    Ok(unique_dir)
}

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
    pub model_checksum: String,
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
                Ok(true) => DeviceSelection {
                    device: "coreml".to_string(),
                    reason: "Auto-selected CoreML (available)".to_string(),
                },
                _ => DeviceSelection {
                    device: "cpu".to_string(),
                    reason: "Auto-selected CPU (CoreML not available)".to_string(),
                },
            }
        }
        other => DeviceSelection {
            device: other.to_string(),
            reason: format!("User explicitly chose {other}"),
        },
    }
}

/// Create an ONNX Runtime session with the specified configuration
/// Returns (Session, ModelInfo, CacheStats) where CacheStats contain CoreML cache information
pub fn create_onnx_session(
    model_source: ModelSource,
    config: &SessionConfig,
) -> Result<(Session, ModelInfo, crate::shared_metadata::CacheStats)> {
    let mut cache_stats = crate::shared_metadata::CacheStats::new();

    // Get model bytes for cache key generation and session creation
    let (bytes, model_info_base) = match model_source {
        ModelSource::EmbeddedBytes(bytes) => {
            let model_checksum = cache_common::calculate_md5_bytes(bytes);
            let model_info = ModelInfo {
                model_source: "Embedded".to_string(),
                model_path: None,
                model_size_bytes: bytes.len(),
                description: "Embedded model bytes".to_string(),
                execution_providers: vec![], // Will be populated later
                model_checksum,
            };
            (bytes.to_vec(), model_info)
        }
        ModelSource::FilePath(path) => {
            let bytes = fs::read(&path)?;
            let model_checksum = cache_common::calculate_md5_bytes(&bytes);
            let model_info = ModelInfo {
                model_source: "File".to_string(),
                model_path: Some(path.clone()),
                model_size_bytes: bytes.len(),
                description: format!("Model loaded from: {path}"),
                execution_providers: vec![], // Will be populated later
                model_checksum,
            };
            (bytes, model_info)
        }
    };

    // Set up stable CoreML cache directory if using CoreML
    let coreml_cache_dir = if config.device == "coreml" {
        match get_stable_coreml_cache_dir(&bytes) {
            Ok(cache_dir) => {
                // Create the cache directory if it doesn't exist
                if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                    let colored_error: String =
                        crate::color_utils::colors::error_level(&e.to_string());
                    log::warn!(
                        "{}Failed to create CoreML cache directory: {colored_error}",
                        symbols::warning()
                    );
                    None
                } else {
                    // Check if cache already exists - this determines cache hit/miss
                    let compiled_model_path = cache_dir.join("compiled_model.mlmodelc");
                    let cache_hit = compiled_model_path.exists();

                    if cache_hit {
                        log::debug!("♻️  Reusing existing CoreML cache: {}", cache_dir.display());
                    } else {
                        log::debug!("🆕 Creating new CoreML cache: {}", cache_dir.display());
                    }

                    // Collect CoreML cache statistics (single traversal) - only when CoreML is used
                    let mut coreml_cache_stats =
                        crate::shared_metadata::CoremlCacheStats::default();
                    coreml_cache_stats.cache_hit = Some(cache_hit);

                    if let Ok(base_coreml_cache) = crate::model_access::get_coreml_cache_dir() {
                        if let Ok((count, size_mb)) =
                            crate::shared_metadata::get_cache_info(&base_coreml_cache)
                        {
                            coreml_cache_stats.cache_count = Some(count);
                            coreml_cache_stats.cache_size_mb = Some(size_mb);
                        }
                    }

                    // Set CoreML cache stats in overall cache stats
                    cache_stats = cache_stats.with_coreml_cache(coreml_cache_stats);

                    Some(cache_dir)
                }
            }
            Err(e) => {
                let colored_error: String = crate::color_utils::colors::error_level(&e.to_string());
                log::warn!(
                    "{}Failed to get CoreML cache directory: {colored_error}",
                    symbols::warning()
                );
                None
            }
        }
    } else {
        None
    };

    // Create execution providers (simplified from cutout pattern)
    let mut execution_providers = match config.device {
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
                log::warn!(
                    "{}CoreML not available, falling back to CPU",
                    symbols::warning()
                );
                vec![CPUExecutionProvider::default().build()]
            }
        },
        "cpu" => {
            log::info!("🖥️  Using CPU execution provider");
            vec![CPUExecutionProvider::default().build()]
        }
        _ => {
            log::warn!(
                "{}Unknown device '{}', using CPU",
                symbols::warning(),
                config.device
            );
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

    let mut build_session_with_retry = |bytes: &[u8]| -> Result<Session> {
        for retry_count in 0..=3 {
            let result = Session::builder()
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
                .with_execution_providers(execution_providers.clone())
                .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {}", e))?
                .commit_from_memory(bytes);

            match result {
                Ok(session) => return Ok(session),
                Err(e) => {
                    let error_msg = e.to_string();

                    // Check if this is a CoreML cache conflict and we can retry
                    if error_msg.contains("item with the same name already exists")
                        && error_msg.contains("compiled_model.mlmodelc")
                        && retry_count < 3
                    {
                        log::warn!("🔄 CoreML cache conflict detected, switching to unique cache directory (attempt {}/3)", retry_count + 1);
                        log::warn!("   Error: {error_msg}");

                        // Generate a new unique cache directory for this retry
                        match get_unique_coreml_cache_dir() {
                            Ok(unique_cache_dir) => {
                                log::debug!(
                                    "🔄 Using unique cache directory: {}",
                                    unique_cache_dir.display()
                                );

                                // Create the unique cache directory
                                if let Err(e) = std::fs::create_dir_all(&unique_cache_dir) {
                                    log::warn!(
                                        "{}Failed to create unique cache directory: {e}",
                                        symbols::warning()
                                    );
                                    return Err(anyhow::anyhow!(
                                        "Failed to create unique cache directory: {e}"
                                    ));
                                }

                                // Update the execution providers with the new cache directory
                                execution_providers = match config.device {
                                    "coreml" => {
                                        match CoreMLExecutionProvider::default().is_available() {
                                            Ok(true) => {
                                                if let Some(cache_path_str) =
                                                    unique_cache_dir.to_str()
                                                {
                                                    log::debug!("🗂️  Configuring CoreML with unique cache: {cache_path_str}");
                                                    vec![CoreMLExecutionProvider::default()
                                                        .with_model_cache_dir(cache_path_str)
                                                        .build()]
                                                } else {
                                                    vec![CoreMLExecutionProvider::default().build()]
                                                }
                                            }
                                            Ok(false) => {
                                                log::warn!("{} CoreML is not available, falling back to CPU", symbols::operation_failed());
                                                vec![CPUExecutionProvider::default().build()]
                                            }
                                            Err(e) => {
                                                log::warn!("{} Failed to check CoreML availability: {e}, falling back to CPU", symbols::operation_failed());
                                                vec![CPUExecutionProvider::default().build()]
                                            }
                                        }
                                    }
                                    _ => vec![CPUExecutionProvider::default().build()],
                                };

                                continue; // Retry the operation with new cache directory
                            }
                            Err(e) => {
                                log::warn!(
                                    "{}Failed to generate unique cache directory: {e}",
                                    symbols::warning()
                                );
                                return Err(anyhow::anyhow!(
                                    "Failed to generate unique cache directory: {e}"
                                ));
                            }
                        }
                    }

                    // If not a retryable error or max retries reached
                    return Err(anyhow::anyhow!("Failed to load model from memory: {}", e));
                }
            }
        }
        unreachable!("Loop should have returned before this point")
    };

    let session = build_session_with_retry(&bytes)?;

    // Update model info with execution providers
    let mut model_info = model_info_base;
    model_info.execution_providers = ep_names;

    // Log execution provider information
    log::debug!(
        "{} Execution providers registered: {}",
        crate::color_utils::symbols::system_setup(),
        model_info.execution_providers.join(" -> ")
    );

    Ok((session, model_info, cache_stats))
}
