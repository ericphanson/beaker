//! Internal model access interface for runtime model management.
//!
//! This module provides a unified interface for accessing models that can come from
//! either embedded bytes (default) or runtime paths (via environment variables).
//! It integrates with the existing model cache system for downloaded models.

use crate::model_cache::{get_or_download_model, ModelInfo};
use crate::onnx_session::ModelSource;
use anyhow::Result;
use std::path::PathBuf;

/// Internal interface for accessing models at runtime.
/// Provides unified access to embedded models and runtime model paths.
pub trait ModelAccess {
    /// Get the model source for this model type.
    /// Returns either embedded bytes (default) or a file path (if overridden via env vars).
    fn get_model_source<'a>() -> Result<ModelSource<'a>>;

    /// Get the embedded model bytes (fallback).
    fn get_embedded_bytes() -> &'static [u8];

    /// Get the environment variable name for runtime model path override.
    fn get_env_var_name() -> &'static str;

    /// Get optional model info for downloading from remote sources.
    /// Returns None for models that don't support remote download.
    fn get_model_info() -> Option<&'static ModelInfo> {
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
    if let Some(model_info) = T::get_model_info() {
        if let Ok(download_path) = get_or_download_model(model_info) {
            let path_str = download_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Downloaded model path is not valid UTF-8"))?;

            log::debug!("📥 Using downloaded model: {path_str}");
            return Ok(ModelSource::FilePath(path_str.to_string()));
        }
    }

    // Default: use embedded bytes
    log::debug!("📦 Using embedded model bytes");
    Ok(ModelSource::EmbeddedBytes(T::get_embedded_bytes()))
}

/// Head detection model access implementation.
pub struct HeadModelAccess;

impl ModelAccess for HeadModelAccess {
    fn get_model_source<'a>() -> Result<ModelSource<'a>> {
        get_model_source_with_env_override::<Self>()
    }

    fn get_embedded_bytes() -> &'static [u8] {
        // Reference to the embedded model bytes from head_detection.rs
        crate::head_detection::MODEL_BYTES
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_HEAD_MODEL_PATH"
    }

    // Currently, head models don't support remote download (embedded only)
    // But this could be added in the future by uncommenting the following:
    // fn get_model_info() -> Option<&'static ModelInfo> {
    //     Some(&HEAD_MODEL_INFO)
    // }
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
        assert!(
            !bytes.is_empty(),
            "Embedded model bytes should be available"
        );
    }
}
