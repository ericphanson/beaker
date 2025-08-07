//! Internal model access interface for runtime model management.
//!
//! This module provides a unified interface for accessing models that can come from
//! either embedded bytes (default) or runtime paths (via environment variables).
//! It integrates with the existing model cache system for downloaded models.

use crate::model_cache::{get_or_download_model, ModelInfo};
use crate::onnx_session::ModelSource;
use anyhow::Result;
use std::path::PathBuf;

/// Runtime model information that can be created from environment variables.
/// This allows overriding URLs and checksums at runtime.
#[derive(Debug, Clone)]
pub struct RuntimeModelInfo {
    pub name: String,
    pub url: String,
    pub md5_checksum: String,
    pub filename: String,
}

impl RuntimeModelInfo {
    /// Create RuntimeModelInfo from static ModelInfo with potential env var overrides.
    pub fn from_model_info_with_overrides(
        base_info: &ModelInfo,
        url_env_var: Option<&str>,
        checksum_env_var: Option<&str>,
    ) -> Self {
        let url = if let Some(env_var) = url_env_var {
            std::env::var(env_var).unwrap_or_else(|_| base_info.url.to_string())
        } else {
            base_info.url.to_string()
        };

        let checksum = if let Some(env_var) = checksum_env_var {
            std::env::var(env_var).unwrap_or_else(|_| base_info.md5_checksum.to_string())
        } else {
            base_info.md5_checksum.to_string()
        };

        RuntimeModelInfo {
            name: base_info.name.to_string(),
            url,
            md5_checksum: checksum,
            filename: base_info.filename.to_string(),
        }
    }
}

/// Get or download a model using RuntimeModelInfo (supports env var overrides).
///
/// Note: This function uses Box::leak to work around the lifetime requirements of ModelInfo.
/// This creates a small memory leak but is acceptable for model configuration that typically
/// happens once at startup. In a future refactor, model_cache should be updated to work
/// with owned strings or explicit lifetimes.
pub fn get_or_download_runtime_model(model_info: RuntimeModelInfo) -> Result<PathBuf> {
    // Convert owned strings to &'static str using Box::leak
    // This is a pragmatic solution for the model configuration use case
    let static_name: &'static str = Box::leak(model_info.name.into_boxed_str());
    let static_url: &'static str = Box::leak(model_info.url.into_boxed_str());
    let static_checksum: &'static str = Box::leak(model_info.md5_checksum.into_boxed_str());
    let static_filename: &'static str = Box::leak(model_info.filename.into_boxed_str());

    let temp_model_info = ModelInfo {
        name: static_name,
        url: static_url,
        md5_checksum: static_checksum,
        filename: static_filename,
    };

    get_or_download_model(&temp_model_info)
}

/// Internal interface for accessing models at runtime.
/// Provides unified access to embedded models and runtime model paths.
pub trait ModelAccess {
    /// Get the model source for this model type.
    /// Returns either embedded bytes (default) or a file path (if overridden via env vars).
    fn get_model_source<'a>() -> Result<ModelSource<'a>>;

    /// Get the embedded model bytes (fallback), if any.
    /// Returns None for models that don't have embedded versions.
    fn get_embedded_bytes() -> Option<&'static [u8]> {
        None
    }

    /// Get the environment variable name for runtime model path override.
    fn get_env_var_name() -> &'static str;

    /// Get the environment variable name for runtime model URL override, if supported.
    fn get_url_env_var_name() -> Option<&'static str> {
        None
    }

    /// Get the environment variable name for runtime model checksum override, if supported.
    fn get_checksum_env_var_name() -> Option<&'static str> {
        None
    }

    /// Get base model info for downloading from remote sources.
    /// Returns None for models that don't support remote download.
    fn get_base_model_info() -> Option<&'static ModelInfo> {
        None
    }
}

/// Generic implementation of model access that handles env var checking and fallbacks.
pub fn get_model_source_with_env_override<T: ModelAccess>() -> Result<ModelSource<'static>> {
    // Check if user has specified a runtime model path via environment variable
    if let Ok(model_path) = std::env::var(T::get_env_var_name()) {
        log::debug!(
            "🔄 Using runtime model path from {}: {}",
            T::get_env_var_name(),
            model_path
        );

        // Validate that the path exists
        let path = PathBuf::from(&model_path);
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "Model path specified in {} does not exist: {}",
                T::get_env_var_name(),
                model_path
            ));
        }

        return Ok(ModelSource::FilePath(model_path));
    }

    // Check if we should download from a remote source
    if let Some(base_model_info) = T::get_base_model_info() {
        // Create RuntimeModelInfo with potential URL/checksum overrides
        let runtime_model_info = RuntimeModelInfo::from_model_info_with_overrides(
            base_model_info,
            T::get_url_env_var_name(),
            T::get_checksum_env_var_name(),
        );

        // Log if any overrides are being used
        if let Some(url_env) = T::get_url_env_var_name() {
            if std::env::var(url_env).is_ok() {
                log::debug!(
                    "🔄 Using custom URL from {}: {}",
                    url_env,
                    runtime_model_info.url
                );
            }
        }
        if let Some(checksum_env) = T::get_checksum_env_var_name() {
            if std::env::var(checksum_env).is_ok() {
                log::debug!(
                    "🔄 Using custom checksum from {}: {}",
                    checksum_env,
                    runtime_model_info.md5_checksum
                );
            }
        }

        if let Ok(download_path) = get_or_download_runtime_model(runtime_model_info) {
            let path_str = download_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Downloaded model path is not valid UTF-8"))?;

            log::debug!("📥 Using downloaded model: {path_str}");
            return Ok(ModelSource::FilePath(path_str.to_string()));
        }
    }

    // Default: use embedded bytes if available
    if let Some(embedded_bytes) = T::get_embedded_bytes() {
        log::debug!("📦 Using embedded model bytes");
        return Ok(ModelSource::EmbeddedBytes(embedded_bytes));
    }

    // If no embedded bytes and no download info, we can't provide a model
    Err(anyhow::anyhow!(
        "No model source available: no embedded bytes and no download info for {}",
        T::get_env_var_name()
    ))
}

/// Head detection model access implementation.
pub struct HeadModelAccess;

impl ModelAccess for HeadModelAccess {
    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        get_model_source_with_env_override::<Self>()
    }

    fn get_embedded_bytes() -> Option<&'static [u8]> {
        // Reference to the embedded model bytes from head_detection.rs
        Some(crate::head_detection::MODEL_BYTES)
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_HEAD_MODEL_PATH"
    }

    // Currently, head models don't support remote download (embedded only)
    // But this could be added in the future by uncommenting the following:
    // fn get_base_model_info() -> Option<&'static ModelInfo> {
    //     Some(&HEAD_MODEL_INFO)
    // }
}

/// Cutout model access implementation.
pub struct CutoutModelAccess;

impl ModelAccess for CutoutModelAccess {
    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        get_model_source_with_env_override::<Self>()
    }

    fn get_embedded_bytes() -> Option<&'static [u8]> {
        // Cutout models are not embedded, they are downloaded
        None
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_CUTOUT_MODEL_PATH"
    }

    fn get_url_env_var_name() -> Option<&'static str> {
        Some("BEAKER_CUTOUT_MODEL_URL")
    }

    fn get_checksum_env_var_name() -> Option<&'static str> {
        Some("BEAKER_CUTOUT_MODEL_CHECKSUM")
    }

    fn get_base_model_info() -> Option<&'static ModelInfo> {
        Some(&crate::cutout_processing::CUTOUT_MODEL_INFO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::NamedTempFile;

    #[test]
    fn test_head_model_access_embedded_default() {
        // Ensure env var is not set
        env::remove_var("BEAKER_HEAD_MODEL_PATH");

        let source = HeadModelAccess::get_model_source().unwrap();

        match source {
            ModelSource::EmbeddedBytes(bytes) => {
                assert!(
                    !bytes.is_empty(),
                    "Embedded model bytes should not be empty"
                );
            }
            _ => panic!("Expected embedded bytes when no env var is set"),
        }
    }

    #[test]
    fn test_head_model_access_env_override() {
        // Create a temporary file to act as a model
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        // Set environment variable
        env::set_var("BEAKER_HEAD_MODEL_PATH", temp_path);

        let source = HeadModelAccess::get_model_source().unwrap();

        match source {
            ModelSource::FilePath(path) => {
                assert_eq!(path, temp_path);
            }
            _ => panic!("Expected file path when env var is set"),
        }

        // Clean up
        env::remove_var("BEAKER_HEAD_MODEL_PATH");
    }

    #[test]
    fn test_head_model_access_invalid_path() {
        // Set environment variable to non-existent path
        env::set_var("BEAKER_HEAD_MODEL_PATH", "/non/existent/path.onnx");

        let result = HeadModelAccess::get_model_source();
        assert!(result.is_err(), "Should fail with non-existent path");

        let error_msg = result.err().unwrap().to_string();
        assert!(
            error_msg.contains("does not exist"),
            "Error should mention non-existent path"
        );

        // Clean up
        env::remove_var("BEAKER_HEAD_MODEL_PATH");
    }

    #[test]
    fn test_env_var_name() {
        assert_eq!(
            HeadModelAccess::get_env_var_name(),
            "BEAKER_HEAD_MODEL_PATH"
        );
    }

    #[test]
    fn test_embedded_bytes_available() {
        let bytes = HeadModelAccess::get_embedded_bytes();
        assert!(bytes.is_some(), "Head model should have embedded bytes");
        assert!(
            !bytes.unwrap().is_empty(),
            "Embedded model bytes should not be empty"
        );
    }

    #[test]
    fn test_cutout_model_access_env_vars() {
        // Test environment variable names
        assert_eq!(
            CutoutModelAccess::get_env_var_name(),
            "BEAKER_CUTOUT_MODEL_PATH"
        );
        assert_eq!(
            CutoutModelAccess::get_url_env_var_name(),
            Some("BEAKER_CUTOUT_MODEL_URL")
        );
        assert_eq!(
            CutoutModelAccess::get_checksum_env_var_name(),
            Some("BEAKER_CUTOUT_MODEL_CHECKSUM")
        );
    }

    #[test]
    fn test_cutout_model_access_no_embedded_bytes() {
        let bytes = CutoutModelAccess::get_embedded_bytes();
        assert!(
            bytes.is_none(),
            "Cutout model should not have embedded bytes"
        );
    }

    #[test]
    fn test_cutout_model_access_has_base_info() {
        let model_info = CutoutModelAccess::get_base_model_info();
        assert!(
            model_info.is_some(),
            "Cutout model should have base model info"
        );

        let info = model_info.unwrap();
        assert_eq!(info.name, "isnet-general-use-v1");
        assert!(info.url.contains("isnet-general-use.onnx"));
        assert!(!info.md5_checksum.is_empty());
        assert_eq!(info.filename, "isnet-general-use.onnx");
    }

    #[test]
    fn test_cutout_model_access_path_override() {
        // Clean up any existing env vars
        env::remove_var("BEAKER_CUTOUT_MODEL_PATH");
        env::remove_var("BEAKER_CUTOUT_MODEL_URL");
        env::remove_var("BEAKER_CUTOUT_MODEL_CHECKSUM");

        // Create a temporary file to act as a model
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        // Set environment variable for path override
        env::set_var("BEAKER_CUTOUT_MODEL_PATH", temp_path);

        let source = CutoutModelAccess::get_model_source().unwrap();

        match source {
            ModelSource::FilePath(path) => {
                assert_eq!(path, temp_path);
            }
            _ => panic!("Expected file path when env var is set"),
        }

        // Clean up
        env::remove_var("BEAKER_CUTOUT_MODEL_PATH");
    }

    #[test]
    fn test_cutout_model_access_invalid_path() {
        // Clean up any existing env vars
        env::remove_var("BEAKER_CUTOUT_MODEL_PATH");
        env::remove_var("BEAKER_CUTOUT_MODEL_URL");
        env::remove_var("BEAKER_CUTOUT_MODEL_CHECKSUM");

        // Set environment variable to non-existent path
        env::set_var("BEAKER_CUTOUT_MODEL_PATH", "/non/existent/path.onnx");

        let result = CutoutModelAccess::get_model_source();
        assert!(result.is_err(), "Should fail with non-existent path");

        let error_msg = result.err().unwrap().to_string();
        assert!(
            error_msg.contains("does not exist"),
            "Error should mention non-existent path"
        );

        // Clean up
        env::remove_var("BEAKER_CUTOUT_MODEL_PATH");
    }

    #[test]
    fn test_runtime_model_info_with_overrides() {
        // Test RuntimeModelInfo creation with env var overrides
        let base_info = &crate::cutout_processing::CUTOUT_MODEL_INFO;

        // Test without any env vars (should use base info)
        env::remove_var("TEST_URL");
        env::remove_var("TEST_CHECKSUM");

        let runtime_info = RuntimeModelInfo::from_model_info_with_overrides(
            base_info,
            Some("TEST_URL"),
            Some("TEST_CHECKSUM"),
        );

        assert_eq!(runtime_info.name, base_info.name);
        assert_eq!(runtime_info.url, base_info.url);
        assert_eq!(runtime_info.md5_checksum, base_info.md5_checksum);
        assert_eq!(runtime_info.filename, base_info.filename);

        // Test with env var overrides
        env::set_var("TEST_URL", "https://example.com/custom.onnx");
        env::set_var("TEST_CHECKSUM", "abcd1234");

        let runtime_info = RuntimeModelInfo::from_model_info_with_overrides(
            base_info,
            Some("TEST_URL"),
            Some("TEST_CHECKSUM"),
        );

        assert_eq!(runtime_info.name, base_info.name);
        assert_eq!(runtime_info.url, "https://example.com/custom.onnx");
        assert_eq!(runtime_info.md5_checksum, "abcd1234");
        assert_eq!(runtime_info.filename, base_info.filename);

        // Clean up
        env::remove_var("TEST_URL");
        env::remove_var("TEST_CHECKSUM");
    }
}
