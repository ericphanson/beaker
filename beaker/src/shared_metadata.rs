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
    pub beaker_env_vars: Option<std::collections::HashMap<String, String>>,
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

/// Collect all BEAKER_* environment variables that are present and non-empty
pub fn collect_beaker_env_vars() -> Option<std::collections::HashMap<String, String>> {
    let mut beaker_vars = std::collections::HashMap::new();
    
    for (key, value) in std::env::vars() {
        if key.starts_with("BEAKER_") && !value.is_empty() {
            beaker_vars.insert(key, value);
        }
    }
    
    if beaker_vars.is_empty() {
        None
    } else {
        Some(beaker_vars)
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
    fn test_collect_beaker_env_vars() {
        // Test with no BEAKER_ variables
        std::env::remove_var("BEAKER_TEST_VAR1");
        std::env::remove_var("BEAKER_TEST_VAR2");
        let _result = collect_beaker_env_vars();
        // There might be other BEAKER_ variables in the system, so we don't test for None directly

        // Set some test BEAKER_ variables
        std::env::set_var("BEAKER_TEST_VAR1", "value1");
        std::env::set_var("BEAKER_TEST_VAR2", "value2");
        std::env::set_var("BEAKER_EMPTY_VAR", ""); // This should not be included

        let result = collect_beaker_env_vars();
        assert!(result.is_some());
        let vars = result.unwrap();
        assert_eq!(vars.get("BEAKER_TEST_VAR1"), Some(&"value1".to_string()));
        assert_eq!(vars.get("BEAKER_TEST_VAR2"), Some(&"value2".to_string()));
        assert!(!vars.contains_key("BEAKER_EMPTY_VAR")); // Empty vars should not be included

        // Clean up
        std::env::remove_var("BEAKER_TEST_VAR1");
        std::env::remove_var("BEAKER_TEST_VAR2");
        std::env::remove_var("BEAKER_EMPTY_VAR");
    }

    #[test]
    fn test_enhanced_metadata_with_env_vars_and_cutout() {
        // Set some test environment variables
        std::env::set_var("BEAKER_TEST_ENV", "test_value");
        std::env::set_var("BEAKER_CUSTOM_MODEL", "/custom/model.onnx");

        // Create test metadata with cutout and environment variables
        use crate::mask_encoding::MaskEntry;
        
        let mask_entry = MaskEntry {
            width: 4,
            height: 2,
            format: "rle-binary-v1 | gzip | base64".to_string(),
            start_value: 0,
            order: "row-major".to_string(),
            data: "H4sIAAAAAAAA/ytJLS4BAG0+lf4EAAAA".to_string(), // Example base64 data
        };

        let beaker_env_vars = collect_beaker_env_vars();
        let has_env_vars = beaker_env_vars.is_some();

        let metadata = BeakerMetadata {
            cutout: Some(CutoutSections {
                core: Some(
                    toml::toml! {
                        model_version = "isnet-general-use-v1"
                        processing_time_ms = 2500.0
                        output_path = "/path/to/output.png"
                    }
                    .into(),
                ),
                config: Some(
                    toml::toml! {
                        alpha_matting = false
                        save_mask = true
                        post_process_mask = true
                    }
                    .into(),
                ),
                execution: Some(ExecutionContext {
                    timestamp: Some(chrono::Utc::now()),
                    beaker_version: Some("0.1.1".to_string()),
                    command_line: Some(vec!["cutout".to_string(), "test.jpg".to_string()]),
                    exit_code: Some(0),
                    model_processing_time_ms: Some(2500.0),
                    beaker_env_vars,
                }),
                system: Some(SystemInfo {
                    device_requested: Some("auto".to_string()),
                    device_selected: Some("cpu".to_string()),
                    device_selection_reason: Some("Auto-selected CPU".to_string()),
                    execution_providers: vec!["CPUExecutionProvider".to_string()],
                    model_source: Some("downloaded".to_string()),
                    model_path: Some("/cache/isnet-general-use.onnx".to_string()),
                    model_size_bytes: Some(45678901),
                    model_load_time_ms: Some(1200.5),
                    model_checksum: Some("fc16ebd8b0c10d971d3513d564d01e29".to_string()),
                }),
                input: Some(InputProcessing {
                    image_path: "/path/to/test.jpg".to_string(),
                    source: "/path/to/test.jpg".to_string(),
                    source_type: "file".to_string(),
                    strict_mode: false,
                }),
                mask: Some(mask_entry),
            }),
            ..Default::default()
        };

        // Serialize to TOML and print to see structure
        let toml_output = toml::to_string_pretty(&metadata).unwrap();
        println!("Enhanced TOML structure with env vars and mask:\n{toml_output}");

        // Verify the structure includes the new fields
        assert!(toml_output.contains("model_checksum"));
        assert!(toml_output.contains("[cutout.mask]"));
        assert!(toml_output.contains("beaker_env_vars") || !has_env_vars);

        // Verify it can be parsed back
        let parsed: BeakerMetadata = toml::from_str(&toml_output).unwrap();
        assert!(parsed.cutout.is_some());
        let cutout = parsed.cutout.unwrap();
        assert!(cutout.system.is_some());
        assert!(cutout.system.unwrap().model_checksum.is_some());
        assert!(cutout.mask.is_some());

        // Clean up test environment variables
        std::env::remove_var("BEAKER_TEST_ENV");
        std::env::remove_var("BEAKER_CUSTOM_MODEL");
    }
}
