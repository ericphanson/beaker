use anyhow::Result;
use chrono::{DateTime, Utc};
use image::DynamicImage;
use log::warn;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Generic utility to track file I/O timing for any model
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct IoTiming {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_time_ms: Option<f64>,
}

impl IoTiming {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn time_image_read<P: AsRef<Path>>(&mut self, path: P) -> Result<DynamicImage> {
        let start = Instant::now();
        let img = image::open(path)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.read_time_ms = Some(self.read_time_ms.unwrap_or(0.0) + elapsed_ms);
        Ok(img)
    }

    pub fn time_cutout_save<P: AsRef<Path>>(
        &mut self,
        img: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        path: P,
    ) -> Result<()> {
        self.time_save_operation(|| Ok(img.save(path)?))
    }

    pub fn time_mask_save<P: AsRef<Path>>(
        &mut self,
        img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
        path: P,
    ) -> Result<()> {
        self.time_save_operation(|| Ok(img.save(path)?))
    }

    /// Add timing for a generic save operation
    pub fn time_save_operation<F>(&mut self, operation: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let start = Instant::now();
        operation()?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.write_time_ms = Some(self.write_time_ms.unwrap_or(0.0) + elapsed_ms);
        Ok(())
    }
}

/// Centralized list of relevant environment variables for version command and metadata collection
pub const RELEVANT_ENV_VARS: &[&str] = &[
    "BEAKER_HEAD_MODEL_PATH",
    "BEAKER_CUTOUT_MODEL_PATH",
    "BEAKER_CUTOUT_MODEL_URL",
    "BEAKER_CUTOUT_MODEL_CHECKSUM",
    "BEAKER_NO_COLOR",
    "BEAKER_DEBUG",
    "NO_COLOR",
    "RUST_LOG",
];

/// Shared metadata structure that can contain both head and cutout results
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct BeakerMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<HeadSections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cutout: Option<CutoutSections>,
}

/// All sections for head detection tool
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HeadSections {
    // Core results (backwards compatibility - flatten the existing head results)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub core: Option<toml::Value>,

    // New subsections
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<InputProcessing>,
}

/// All sections for cutout processing tool
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CutoutSections {
    // Core results (backwards compatibility - flatten the existing cutout results)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub core: Option<toml::Value>,

    // New subsections
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<InputProcessing>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<crate::mask_encoding::MaskEntry>,
}

/// Execution context for a tool invocation
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ExecutionContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beaker_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_line: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_processing_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_io: Option<IoTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beaker_env_vars: Option<std::collections::BTreeMap<String, String>>,
}

/// System information for a tool invocation
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SystemInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_requested: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_selected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_selection_reason: Option<String>,
    pub execution_providers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_load_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
}

/// Input processing statistics for a tool invocation
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct InputProcessing {
    pub image_path: String,
    pub source: String,
    pub source_type: String,
    pub strict_mode: bool,
}

/// Load existing metadata from a file, or create new empty metadata
pub fn load_or_create_metadata(path: &Path) -> Result<BeakerMetadata> {
    if path.exists() {
        let content = fs::read_to_string(path)?;
        match toml::from_str::<BeakerMetadata>(&content) {
            Ok(metadata) => Ok(metadata),
            Err(e) => {
                let colored_error = crate::color_utils::colors::warning_level(&e.to_string());
                warn!(
                    "{} Dropping existing metadata from {}:\n{}",
                    crate::color_utils::symbols::warning(),
                    path.display(),
                    colored_error
                );

                Ok(BeakerMetadata::default())
            }
        }
    } else {
        Ok(BeakerMetadata::default())
    }
}

/// Save metadata to a file
pub fn save_metadata(metadata: &BeakerMetadata, path: &Path) -> Result<()> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let toml_content = match toml::to_string_pretty(metadata) {
        Ok(content) => content,
        Err(e) => {
            log::debug!("About to serialize metadata: {metadata:#?}");
            // Try to serialize each section individually to isolate the problem
            if let Some(ref head) = metadata.head {
                log::error!("Head section: {head:#?}");
                if let Err(head_err) = toml::to_string_pretty(head) {
                    log::error!("Head section serialization failed: {head_err}");
                }
            }
            if let Some(ref cutout) = metadata.cutout {
                log::error!("Cutout section: {cutout:#?}");
                if let Err(cutout_err) = toml::to_string_pretty(cutout) {
                    log::error!("Cutout section serialization failed: {cutout_err}");
                }
            }
            return Err(anyhow::anyhow!(
                "Failed to serialize metadata to TOML: {}. This usually means a field contains a value that cannot be represented in TOML format.",
                e
            ));
        }
    };

    fs::write(path, toml_content)?;
    Ok(())
}

/// Generate metadata file path for an input image
pub fn get_metadata_path(
    input_path: &Path,
    output_dir: Option<&str>,
) -> Result<std::path::PathBuf> {
    let input_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid input filename"))?;

    let metadata_filename = format!("{input_stem}.beaker.toml");

    let metadata_path = if let Some(output_dir) = output_dir {
        Path::new(output_dir).join(metadata_filename)
    } else {
        input_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(metadata_filename)
    };

    Ok(metadata_path)
}

/// Collect relevant environment variables that are present and non-empty
pub fn collect_beaker_env_vars() -> Option<std::collections::BTreeMap<String, String>> {
    let mut env_vars = std::collections::BTreeMap::new();

    for env_name in RELEVANT_ENV_VARS {
        if let Ok(value) = std::env::var(env_name) {
            if !value.is_empty() {
                env_vars.insert(env_name.to_string(), value);
            }
        }
    }

    if env_vars.is_empty() {
        None
    } else {
        Some(env_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_structure() {
        // Create test metadata with tool sections
        let metadata = BeakerMetadata {
            head: Some(HeadSections {
                core: Some(
                    toml::toml! {
                        model_version = "test-v1.0.0"
                        confidence_threshold = 0.25
                        processing_time_ms = 150.5
                    }
                    .into(),
                ),
                config: Some(
                    toml::toml! {
                        confidence = 0.25
                        iou_threshold = 0.45
                        crop = true
                    }
                    .into(),
                ),
                execution: Some(ExecutionContext {
                    timestamp: Some(chrono::Utc::now()),
                    beaker_version: Some("0.1.1".to_string()),
                    command_line: Some(vec!["head".to_string(), "test.jpg".to_string()]),
                    exit_code: Some(0),
                    model_processing_time_ms: Some(150.5),
                    file_io: Some(IoTiming {
                        read_time_ms: Some(5.2),
                        write_time_ms: Some(8.1),
                    }),
                    beaker_env_vars: None,
                }),
                system: Some(SystemInfo {
                    device_requested: Some("auto".to_string()),
                    device_selected: Some("cpu".to_string()),
                    device_selection_reason: Some("Auto-selected CPU".to_string()),
                    execution_providers: vec!["CPUExecutionProvider".to_string()],
                    model_source: Some("embedded".to_string()),
                    model_path: None,
                    model_size_bytes: Some(12345678),
                    model_load_time_ms: Some(25.3),
                    model_checksum: Some("abc123def456".to_string()),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Serialize to TOML and print to see structure
        let toml_output = toml::to_string_pretty(&metadata).unwrap();
        println!("Generated TOML structure:\n{toml_output}");

        // Verify it can be parsed back
        let parsed: BeakerMetadata = toml::from_str(&toml_output).unwrap();
        assert!(parsed.head.is_some());
        assert!(parsed.head.as_ref().unwrap().config.is_some());
        assert!(parsed.head.as_ref().unwrap().execution.is_some());
        assert!(parsed.head.as_ref().unwrap().system.is_some());
    }

    #[test]
    fn test_get_metadata_path() {
        let input_path = Path::new("/path/to/image.jpg");
        let metadata_path = get_metadata_path(input_path, None).unwrap();
        assert_eq!(metadata_path.file_name().unwrap(), "image.beaker.toml");
        assert_eq!(metadata_path.parent().unwrap(), Path::new("/path/to"));

        let metadata_path_with_output = get_metadata_path(input_path, Some("/output")).unwrap();
        assert_eq!(
            metadata_path_with_output,
            Path::new("/output/image.beaker.toml")
        );
    }

    #[test]
    fn test_collect_beaker_env_vars_basic() {
        // Test that the function can be called - actual env var testing is in integration tests
        let _result = collect_beaker_env_vars();
        // Just verify the function doesn't panic
    }

    // Environment variable tests moved to integration tests to avoid race conditions
}
