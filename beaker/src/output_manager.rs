//! Output path management providing unified logic for all models.
//!
//! This module handles the complexities of output path generation across different
//! models, ensuring consistent behavior for:
//! - Single vs multiple outputs
//! - Output directory vs same directory as input
//! - Suffix handling for auxiliary files
//! - Numbered outputs with appropriate zero-padding
//! - Metadata path utilities

use anyhow::Result;
use log::debug;
use std::path::{Path, PathBuf};

use crate::model_processing::ModelConfig;
use crate::shared_metadata::{
    get_metadata_path, load_or_create_metadata, save_metadata, CutoutSections, HeadSections,
};

/// Unified output path management for all models
pub struct OutputManager<'a> {
    config: &'a dyn ModelConfig,
    input_path: &'a Path,
}

impl<'a> OutputManager<'a> {
    /// Create a new OutputManager for the given config and input path
    pub fn new(config: &'a dyn ModelConfig, input_path: &'a Path) -> Self {
        Self { config, input_path }
    }

    /// Get the input file stem (filename without extension)
    fn input_stem(&self) -> &str {
        self.input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
    }

    /// Generate primary output path without suffix when using output_dir
    ///
    /// This is for main outputs that should be clean when placed in a dedicated output directory
    pub fn generate_main_output_path(
        &self,
        default_suffix: &str,
        extension: &str,
    ) -> Result<PathBuf> {
        let input_stem = self.input_stem();

        let output_filename = if self.config.base().output_dir.is_some() {
            // Clean filename when using output directory
            format!("{input_stem}.{extension}")
        } else {
            // Add suffix when placing next to input
            format!("{input_stem}_{default_suffix}.{extension}")
        };

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Generate numbered output path for multiple similar outputs
    ///
    /// Examples:
    /// - Single item: "image_crop.jpg" or "image.jpg" (with output_dir)
    /// - Multiple items < 10: "image_crop-1.jpg", "image_crop-2.jpg"
    /// - Multiple items >= 10: "image_crop-01.jpg", "image_crop-02.jpg"
    pub fn generate_numbered_output(
        &self,
        base_suffix: &str,
        index: usize,
        total: usize,
        extension: &str,
    ) -> Result<PathBuf> {
        let input_stem = self.input_stem();

        let output_filename = if total == 1 {
            // Single output - use main output behavior
            if self.config.base().output_dir.is_some() {
                format!("{input_stem}.{extension}")
            } else {
                format!("{input_stem}_{base_suffix}.{extension}")
            }
        } else {
            // Multiple outputs - always numbered
            let number_format = if total >= 10 {
                format!("{index:02}") // Zero-padded for 10+
            } else {
                format!("{index}") // No padding for < 10
            };

            if self.config.base().output_dir.is_some() {
                format!("{input_stem}-{number_format}.{extension}")
            } else {
                format!("{input_stem}_{base_suffix}-{number_format}.{extension}")
            }
        };

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Generate auxiliary output path (always includes suffix)
    pub fn generate_auxiliary_output(&self, suffix: &str, extension: &str) -> Result<PathBuf> {
        let input_stem = self.input_stem();
        let output_filename = format!("{input_stem}_{suffix}.{extension}");

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Make a file path relative to the metadata file location
    pub fn make_relative_to_metadata(&self, path: &Path) -> Result<String> {
        if self.config.base().skip_metadata {
            return Ok(path.to_string_lossy().to_string());
        }

        let metadata_path =
            get_metadata_path(self.input_path, self.config.base().output_dir.as_deref())?;

        make_path_relative_to_toml(path, &metadata_path)
    }

    /// Save complete metadata sections (core + enhanced sections)
    pub fn save_complete_metadata(
        &self,
        head_sections: Option<HeadSections>,
        cutout_sections: Option<CutoutSections>,
    ) -> Result<()> {
        if self.config.base().skip_metadata {
            return Ok(());
        }

        let metadata_path =
            get_metadata_path(self.input_path, self.config.base().output_dir.as_deref())?;

        let mut metadata = load_or_create_metadata(&metadata_path)?;

        // Update the sections that were provided
        if let Some(head) = head_sections {
            metadata.head = Some(head);
        }
        if let Some(cutout) = cutout_sections {
            metadata.cutout = Some(cutout);
        }

        save_metadata(&metadata, &metadata_path)?;

        debug!("📋 Saved complete metadata to: {}", metadata_path.display());

        Ok(())
    }
}

/// Make a file path relative to a TOML file (used for metadata)
pub fn make_path_relative_to_toml(file_path: &Path, toml_path: &Path) -> Result<String> {
    if let Some(toml_dir) = toml_path.parent() {
        if let Ok(rel_path) = file_path.strip_prefix(toml_dir) {
            // Convert to forward slashes for cross-platform compatibility
            Ok(rel_path.to_string_lossy().replace('\\', "/"))
        } else {
            // If we can't make it relative, use absolute path
            Ok(file_path.to_string_lossy().to_string())
        }
    } else {
        Ok(file_path.to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BaseModelConfig, HeadDetectionConfig};
    use tempfile::TempDir;

    fn create_test_config(output_dir: Option<String>) -> HeadDetectionConfig {
        HeadDetectionConfig {
            base: BaseModelConfig {
                sources: vec!["test.jpg".to_string()],
                device: "cpu".to_string(),
                output_dir,
                skip_metadata: false,
                strict: true,
            },
            confidence: 0.25,
            iou_threshold: 0.45,
            crop: true,
            bounding_box: false,
        }
    }

    #[test]
    fn test_main_output_path_same_directory() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_main_output_path("cutout", "png").unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_cutout.png"));
    }

    #[test]
    fn test_main_output_path_with_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_main_output_path("cutout", "png").unwrap();

        assert_eq!(output_path, output_dir.join("test.png"));
    }

    #[test]
    fn test_auxiliary_output_always_has_suffix() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_auxiliary_output("mask", "png").unwrap();

        assert_eq!(output_path, output_dir.join("test_mask.png"));
    }

    #[test]
    fn test_numbered_output_single_item() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 1, 1, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop.jpg"));
    }

    #[test]
    fn test_numbered_output_multiple_items_less_than_10() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 2, 5, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop-2.jpg"));
    }

    #[test]
    fn test_numbered_output_multiple_items_10_or_more() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 5, 15, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop-05.jpg"));
    }

    #[test]
    fn test_numbered_output_with_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 3, 12, "jpg")
            .unwrap();

        assert_eq!(output_path, output_dir.join("test-03.jpg"));
    }

    #[test]
    fn test_make_path_relative_to_toml() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("test.beaker.toml");
        let file_path = temp_dir.path().join("test_crop.jpg");

        let relative = make_path_relative_to_toml(&file_path, &toml_path).unwrap();
        assert_eq!(relative, "test_crop.jpg");
    }

    #[test]
    fn test_make_path_relative_with_subdirectory() {
        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("subdir");
        let toml_path = temp_dir.path().join("test.beaker.toml");
        let file_path = subdir.join("test_crop.jpg");

        let relative = make_path_relative_to_toml(&file_path, &toml_path).unwrap();
        assert_eq!(relative, "subdir/test_crop.jpg");
    }
}
