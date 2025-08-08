use anyhow::Result;
use chrono::{DateTime, Utc};
use log::warn;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
}
