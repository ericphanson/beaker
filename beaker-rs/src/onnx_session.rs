use anyhow::Result;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    logging::LogLevel,
    session::Session,
};
use std::fs;
use std::time::Instant;

/// Configuration for creating ONNX sessions
pub struct SessionConfig<'a> {
    pub device: &'a str,
    pub num_images: usize,
    pub verbose: bool,
    pub suppress_warnings: bool,
}

/// Trait for verbose output - allows different config types to provide verbose logging
pub trait VerboseOutput {
    fn verbose_println(&self, msg: String);
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
pub fn determine_optimal_device(
    requested_device: &str,
    num_images: usize,
    verbose_output: &dyn VerboseOutput,
) -> DeviceSelection {
    const COREML_THRESHOLD: usize = 3; // Use CoreML for 3+ images

    match requested_device {
        "auto" => {
            if num_images >= COREML_THRESHOLD {
                // Check if CoreML is available
                let coreml = CoreMLExecutionProvider::default();
                match coreml.is_available() {
                    Ok(true) => {
                        let reason = format!("Processing {num_images} images - using CoreML for better batch performance");
                        verbose_output.verbose_println(format!("📊 {reason}"));
                        DeviceSelection {
                            device: "coreml".to_string(),
                            reason,
                        }
                    }
                    _ => {
                        let reason = format!(
                            "Processing {num_images} images - CoreML not available, using CPU"
                        );
                        verbose_output.verbose_println(format!("📊 {reason}"));
                        DeviceSelection {
                            device: "cpu".to_string(),
                            reason,
                        }
                    }
                }
            } else {
                let reason = format!("Processing {num_images} images - using CPU for small batch");
                verbose_output.verbose_println(format!("📊 {reason}"));
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
    verbose_output: &dyn VerboseOutput,
) -> Result<(Session, f64)> {
    // Create execution providers (simplified from cutout pattern)
    let execution_providers = match config.device {
        "coreml" => match CoreMLExecutionProvider::default().is_available() {
            Ok(true) => vec![
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ],
            _ => {
                verbose_output
                    .verbose_println("⚠️  CoreML not available, falling back to CPU".to_string());
                vec![CPUExecutionProvider::default().build()]
            }
        },
        "cpu" => {
            verbose_output.verbose_println("🖥️  Using CPU execution provider".to_string());
            vec![CPUExecutionProvider::default().build()]
        }
        _ => {
            verbose_output
                .verbose_println(format!("⚠️  Unknown device '{}', using CPU", config.device));
            vec![CPUExecutionProvider::default().build()]
        }
    };

    // Store EP info for logging before moving the vector
    let ep_names: Vec<String> = execution_providers
        .iter()
        .map(|ep| format!("{ep:?}"))
        .collect();

    // Set log level to suppress CoreML warnings unless verbose mode is enabled
    let log_level = if config.verbose && !config.suppress_warnings {
        LogLevel::Warning // Show warnings in verbose mode
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

    verbose_output.verbose_println(format!(
        "🤖 Loaded {} in {:.3}ms",
        model_info,
        session_load_time.as_secs_f64() * 1000.0
    ));

    // Log execution provider information
    verbose_output.verbose_println(format!(
        "⚙️  Execution providers registered: {}",
        ep_names.join(" -> ")
    ));

    Ok((session, session_load_time.as_secs_f64() * 1000.0))
}
