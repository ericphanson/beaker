//! Stamp file management for Make-compatible incremental builds
//!
//! This module handles the creation and management of stamp files that enable
//! Make to perform accurate incremental builds. Each stamp file contains a
//! deterministic hash based only on inputs that affect the byte-level output
//! of that stage.

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::cache_common::calculate_md5;
use beaker_stamp::write_cfg_stamp;

#[cfg(test)]
use std::fs;

/// Generate tool version hash
pub fn generate_tool_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("beaker:{}", env!("CARGO_PKG_VERSION")));
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Generate model hash from file path
pub fn generate_model_hash_from_path(model_path: &Path) -> Result<String> {
    if model_path.exists() {
        calculate_md5(model_path).map(|hash| hash[..16].to_string())
    } else {
        Err(anyhow!("Model file not found: {}", model_path.display()))
    }
}

/// Create stamp file if content has changed
///
/// This is a legacy function that duplicates beaker-stamp functionality.
/// New code should use beaker_stamp::write_cfg_stamp directly.
pub fn create_or_update_stamp(stamp_type: &str, hash: &str, content: &str) -> Result<PathBuf> {
    let stamps_dir = beaker_stamp::paths::stamp_dir();

    let stamp_filename = format!("{stamp_type}-{hash}.stamp");
    let stamp_path = stamps_dir.join(stamp_filename);

    beaker_stamp::write::write_atomic_if_changed(&stamp_path, content.as_bytes())
        .map_err(|e| anyhow!("Failed to write stamp file: {}", e))?;

    Ok(stamp_path)
}

/// Generate all stamps for any model run using beaker-stamp
pub fn generate_stamps_for_model<T: beaker_stamp::Stamp>(
    model_name: &str,
    config: &T,
    model_path: Option<&Path>,
) -> Result<StampInfo> {
    // Generate config stamp using the new proc macro approach
    let config_stamp = write_cfg_stamp(model_name, config)
        .map_err(|e| anyhow!("Failed to write {} config stamp: {}", model_name, e))?;

    // Generate tool stamp
    let tool_hash = generate_tool_hash();
    let tool_content = format!("tool:{tool_hash}");
    let tool_stamp = create_or_update_stamp("tool", &tool_hash, &tool_content)?;

    // Generate model stamp if model file exists
    let model_stamp = if let Some(model_path) = model_path {
        let model_hash = generate_model_hash_from_path(model_path)?;
        let model_content = format!("model-{model_name}:{model_hash}");
        Some(create_or_update_stamp(
            &format!("model-{model_name}"),
            &model_hash,
            &model_content,
        )?)
    } else {
        None
    };

    Ok(StampInfo {
        config_stamp,
        tool_stamp,
        model_stamp,
    })
}

/// Information about generated stamp files
#[derive(Debug)]
pub struct StampInfo {
    pub config_stamp: PathBuf,
    pub tool_stamp: PathBuf,
    pub model_stamp: Option<PathBuf>,
}

impl StampInfo {
    /// Get all stamp paths as a vector
    pub fn all_stamps(&self) -> Vec<&Path> {
        let mut stamps = vec![self.config_stamp.as_path(), self.tool_stamp.as_path()];
        if let Some(ref model_stamp) = self.model_stamp {
            stamps.push(model_stamp.as_path());
        }
        stamps
    }
}

#[cfg(test)]
mod tests {
    //! Tests for stamp generation and config hashing
    //!
    //! **MAINTENANCE REMINDER**: When adding new fields to DetectionConfig or CutoutConfig
    //! that affect output file bytes, you MUST:
    //! 1. Update the corresponding hash function (generate_*_config_hash)
    //! 2. Add test cases here to verify the new field affects the hash
    //! 3. Verify that changes to the field cause different stamp files to be generated
    //!
    //! This helps prevent dependency tracking issues where changes don't trigger rebuilds.

    use super::*;
    use crate::config::{DetectCommand, DetectionConfig, GlobalArgs};
    use clap_verbosity_flag::Verbosity;

    fn create_test_detection_config() -> DetectionConfig {
        let global = GlobalArgs {
            device: "cpu".to_string(),
            output_dir: None,
            metadata: false,
            depfile: None,
            verbosity: Verbosity::new(0, 0),
            permissive: false,
            no_color: false,
        };

        let cmd = DetectCommand {
            sources: vec!["test.jpg".to_string()],
            confidence: 0.25,
            iou_threshold: 0.45,
            crop: Some("head".to_string()),
            bounding_box: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        DetectionConfig::from_args(global, cmd).unwrap()
    }

    #[test]
    fn test_detection_stamp_deterministic() {
        use beaker_stamp::Stamp;
        let config = create_test_detection_config();
        let hash1 = config.stamp_hash();
        let hash2 = config.stamp_hash();
        assert_eq!(hash1, hash2);
        assert!(hash1.starts_with("sha256:"));
    }

    #[test]
    fn test_detection_stamp_changes_with_params() {
        use beaker_stamp::Stamp;
        let mut config = create_test_detection_config();
        let hash1 = config.stamp_hash();

        config.confidence = 0.5; // Change confidence
        let hash2 = config.stamp_hash();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_tool_hash_generation() {
        let hash1 = generate_tool_hash();
        let hash2 = generate_tool_hash();
        assert_eq!(hash1, hash2); // Should be deterministic
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_stamp_file_creation() {
        // Use a unique stamp type to avoid conflicts
        let stamp_type = format!("test-{}", std::process::id());
        let hash = "abcd1234";
        let content = "test content";

        let stamp_path = create_or_update_stamp(&stamp_type, hash, content).unwrap();

        assert!(stamp_path.exists());
        assert_eq!(fs::read_to_string(&stamp_path).unwrap(), content);

        // Cleanup
        let _ = fs::remove_file(&stamp_path);
    }

    #[test]
    fn test_stamp_file_preserves_mtime_when_unchanged() {
        let stamp_type = format!("test-preserve-{}", std::process::id());
        let hash = "efgh5678";
        let content = "preserve test content";

        // Create initial stamp
        let stamp_path = create_or_update_stamp(&stamp_type, hash, content).unwrap();
        let initial_metadata = fs::metadata(&stamp_path).unwrap();
        let initial_modified = initial_metadata.modified().unwrap();

        // Wait a bit to ensure time difference would be detectable
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Create stamp again with same content
        let stamp_path2 = create_or_update_stamp(&stamp_type, hash, content).unwrap();
        assert_eq!(stamp_path, stamp_path2);

        let final_metadata = fs::metadata(&stamp_path).unwrap();
        let final_modified = final_metadata.modified().unwrap();

        // Modification time should be preserved
        assert_eq!(initial_modified, final_modified);

        // Cleanup
        let _ = fs::remove_file(&stamp_path);
    }
}
