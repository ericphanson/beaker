use anyhow::Result;
use chrono::{DateTime, Utc};
use image::DynamicImage;
use log::warn;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;
use toml_edit::{Array, DocumentMut, Item, Value};

/// Create a formatted TOML array with each row on a single line
fn create_formatted_array() -> Array {
    let mut outer = Array::new();
    // Optional; TOML allows trailing commas in arrays (v1.0).
    outer.set_trailing_comma(false);
    outer.set_trailing("\n"); // newline before closing ]
    outer
}

/// Add a row to the formatted array with proper indentation
fn add_formatted_row(outer: &mut Array, inner: Array) {
    let mut row_val = Value::from(inner);
    // Start each row on its own line (comma stays after the row).
    row_val.decor_mut().set_prefix("\n    ");
    outer.push_formatted(row_val);
}

/// Put each 20-elem row on a single line; rows separated by newlines.
fn build_inline_rows_u8(mat: &[[u8; 20]; 20]) -> Value {
    let mut outer = create_formatted_array();

    for row in mat {
        let mut inner = Array::new();
        for &x in row {
            inner.push(x as i64); // TOML integers are i64
        }
        add_formatted_row(&mut outer, inner);
    }

    Value::from(outer)
}

/// Put each 20-elem row on a single line; rows separated by newlines.
/// Rounds f32 values to 3 decimal places.
fn build_inline_rows_f32(mat: &[[f32; 20]; 20]) -> Value {
    let mut outer = create_formatted_array();

    for row in mat {
        let mut inner = Array::new();
        for &x in row {
            // Round to 3 decimal places and store as f64 (TOML's float type)
            let rounded = format!("{x:.2}").parse::<f64>().unwrap_or(f64::NAN);
            inner.push(rounded);
        }
        add_formatted_row(&mut outer, inner);
    }

    Value::from(outer)
}

/// Extract a 20x20 u8 array from TOML value
fn extract_u8_grid(grid_value: &toml::Value) -> Option<[[u8; 20]; 20]> {
    if let Some(grid_array) = grid_value.as_array() {
        if grid_array.len() == 20 {
            let mut grid: [[u8; 20]; 20] = [[0; 20]; 20];

            for (i, row) in grid_array.iter().enumerate() {
                if let Some(row_array) = row.as_array() {
                    if row_array.len() == 20 {
                        for (j, val) in row_array.iter().enumerate() {
                            if let Some(num) = val.as_integer() {
                                grid[i][j] = num as u8;
                            } else {
                                return None;
                            }
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }

            Some(grid)
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract a 20x20 f32 array from TOML value
fn extract_f32_grid(grid_value: &toml::Value) -> Option<[[f32; 20]; 20]> {
    if let Some(grid_array) = grid_value.as_array() {
        if grid_array.len() == 20 {
            let mut grid: [[f32; 20]; 20] = [[0.0; 20]; 20];

            for (i, row) in grid_array.iter().enumerate() {
                if let Some(row_array) = row.as_array() {
                    if row_array.len() == 20 {
                        for (j, val) in row_array.iter().enumerate() {
                            if let Some(num) = val.as_float() {
                                grid[i][j] = num as f32;
                            } else if let Some(num) = val.as_integer() {
                                grid[i][j] = num as f32;
                            } else {
                                return None;
                            }
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }

            Some(grid)
        } else {
            None
        }
    } else {
        None
    }
}

/// ONNX download cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OnnxCacheStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_models_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_models_size_mb: Option<f64>,
}

/// CoreML cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CoremlCacheStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_size_mb: Option<f64>,
}

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
    "BEAKER_DETECT_MODEL_PATH",
    "BEAKER_DETECT_MODEL_URL",
    "BEAKER_DETECT_MODEL_CHECKSUM",
    "BEAKER_CUTOUT_MODEL_PATH",
    "BEAKER_CUTOUT_MODEL_URL",
    "BEAKER_CUTOUT_MODEL_CHECKSUM",
    "BEAKER_QUALITY_MODEL_PATH",
    "BEAKER_QUALITY_MODEL_URL",
    "BEAKER_QUALITY_MODEL_CHECKSUM",
    "BEAKER_NO_COLOR",
    "BEAKER_DEBUG",
    "NO_COLOR",
    "RUST_LOG",
];

/// Shared metadata structure that can contain detect and cutout results
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct BeakerMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect: Option<DetectSections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cutout: Option<CutoutSections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualitySections>,
}

/// All sections for detect command tool
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct DetectSections {
    // Core results (flattened detection results)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub core: Option<toml::Value>,
    // Subsections
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

/// All sections for cutout processing tool
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct QualitySections {
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

    // Cache Statistics (only present when respective caches are accessed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_cache: Option<OnnxCacheStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coreml_cache: Option<CoremlCacheStats>,
}

impl SystemInfo {}

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

/// Save metadata to a file using toml_edit for better formatting control
pub fn save_metadata(metadata: &BeakerMetadata, path: &Path) -> Result<()> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let toml_content = toml_edit::ser::to_string_pretty(metadata)?;

    // Now parse into a document
    let mut doc = match toml_content.parse::<DocumentMut>() {
        Ok(document) => document,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to parse generated TOML into DocumentMut: {}",
                e
            ));
        }
    };

    // Customize the quality grid formatting if present
    if let Some(quality_table) = doc.get_mut("quality") {
        if let Some(quality_table) = quality_table.as_table_mut() {
            // Format local_paq2piq_grid (u8 values)
            if let Some(local_quality_grid_item) = quality_table.get_mut("local_paq2piq_grid") {
                if let Some(ref quality_sections) = metadata.quality {
                    if let Some(ref core) = quality_sections.core {
                        if let Some(grid_value) = core.get("local_paq2piq_grid") {
                            if let Some(grid) = extract_u8_grid(grid_value) {
                                let formatted_value = build_inline_rows_u8(&grid);
                                *local_quality_grid_item = Item::Value(formatted_value);
                            }
                        }
                    }
                }
            }

            // Format local_blur_weights (f32 values)
            if let Some(local_blur_weights_item) = quality_table.get_mut("local_blur_weights") {
                if let Some(ref quality_sections) = metadata.quality {
                    if let Some(ref core) = quality_sections.core {
                        if let Some(weights_value) = core.get("local_blur_weights") {
                            if let Some(weights) = extract_f32_grid(weights_value) {
                                let formatted_value = build_inline_rows_f32(&weights);
                                *local_blur_weights_item = Item::Value(formatted_value);
                            }
                        }
                    }
                }
            }
        }
    }
    let final_content = doc.to_string();
    fs::write(path, final_content)?;
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

/// Get cache information from a directory (single traversal)
/// Returns (count, total_size_mb) for the cache directory
pub fn get_cache_info(cache_dir: &Path) -> Result<(u32, f64)> {
    if !cache_dir.exists() {
        log::debug!("Cache directory does not exist: {}", cache_dir.display());
        return Ok((0, 0.0));
    }

    let mut count = 0u32;
    let mut total_size = 0u64;

    log::debug!("Collecting cache info from: {}", cache_dir.display());

    // Read directory entries
    let entries = fs::read_dir(cache_dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Skip directories, lock files, and hidden files
        if path.is_dir()
            || path.extension().and_then(|s| s.to_str()) == Some("lock")
            || path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with('.'))
        {
            continue;
        }

        // Count regular files and accumulate their sizes
        if let Ok(metadata) = fs::metadata(&path) {
            if metadata.is_file() {
                count += 1;
                total_size += metadata.len();
                log::debug!(
                    "  Found cached file: {} ({} bytes)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    metadata.len()
                );
            }
        }
    }

    let total_size_mb = total_size as f64 / (1024.0 * 1024.0);

    log::debug!("Cache summary: {count} files, {total_size_mb:.2} MB total");

    Ok((count, total_size_mb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_structure() {
        // Create test metadata with tool sections
        let metadata = BeakerMetadata {
            detect: Some(DetectSections {
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
                        crop = ["head"]
                    }
                    .into(),
                ),
                execution: Some(ExecutionContext {
                    timestamp: Some(chrono::Utc::now()),
                    beaker_version: Some("0.1.1".to_string()),
                    command_line: Some(vec!["detect".to_string(), "test.jpg".to_string()]),
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
                    // Cache statistics should be None for embedded models in this test
                    onnx_cache: None,
                    coreml_cache: None,
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
        assert!(parsed.detect.is_some());
        assert!(parsed.detect.as_ref().unwrap().config.is_some());
        assert!(parsed.detect.as_ref().unwrap().execution.is_some());
        assert!(parsed.detect.as_ref().unwrap().system.is_some());
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
