//! Model access interface with integrated caching for runtime model management.
//!
//! This module provides a unified interface for accessing models that can come from
//! either embedded bytes (default) or runtime paths (via environment variables).
//! It includes integrated model caching and download functionality.

use crate::cache_common;
use crate::color_utils::symbols;
use crate::onnx_session::ModelSource;
use crate::progress::{add_progress_bar, remove_progress_bar};
use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Model information for caching and verification (with owned strings)
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub url: String,
    pub md5_checksum: String,
    pub filename: String,
}

/// Get the cache directory for storing downloaded models
pub fn get_cache_dir() -> Result<PathBuf> {
    cache_common::get_cache_dir_with_env_override("ONNX_MODEL_CACHE_DIR", "onnx-models")
}

/// Get the CoreML cache directory for storing compiled CoreML models
pub fn get_coreml_cache_dir() -> Result<PathBuf> {
    cache_common::get_cache_base_dir().map(|dir| dir.join("beaker").join("coreml"))
}

/// Get detailed file information for debugging
fn get_file_info(path: &Path) -> Result<String> {
    cache_common::get_file_info(path)
}

/// Download a model from URL to the specified path with progress bar
fn download_model(url: &str, output_path: &Path) -> Result<()> {
    log::info!("📥 Downloading model from: {url}");

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
        log::debug!("📁 Created parent directory: {}", parent.display());
    }

    // Download the file with better error handling and redirect support
    log::debug!("🌐 Starting HTTP request...");
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| anyhow!("Failed to send HTTP request: {}", e))?;

    let status = response.status();
    log::debug!("📡 HTTP response status: {status}");

    if !status.is_success() {
        return Err(anyhow!("HTTP request failed with status: {}", status));
    }

    let content_length = response.content_length();

    // Create progress bar if we know the content length
    let progress_bar = if let Some(length) = content_length {
        let length_mb = length as f64 / (1024.0 * 1024.0);
        log::info!("📏 Download size: {length_mb:.1} MB");

        let pb = ProgressBar::new(length);
        add_progress_bar(pb.clone());

        // Simple no-color progress bar style
        let style = ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )
            .map_err(|e| anyhow!("Failed to create progress style: {}", e))?
            .progress_chars("#> ");

        pb.set_style(style);
        pb.set_message("Downloading model");

        Some(pb)
    } else {
        log::warn!(
            "{}Content-Length header missing, showing spinner instead of progress bar",
            symbols::warning()
        );

        let pb = ProgressBar::new_spinner();
        add_progress_bar(pb.clone());
        pb.set_message("Downloading model (unknown size)...");
        pb.enable_steady_tick(Duration::from_millis(100));

        Some(pb)
    };

    // Create output file
    log::debug!("{} Creating output file...", symbols::save_file());
    let mut file = fs::File::create(output_path).map_err(|e| {
        anyhow!(
            "Failed to create output file {}: {}",
            output_path.display(),
            e
        )
    })?;

    // Stream download with progress updates
    let mut downloaded = 0u64;
    let mut buffer = [0; 8192]; // 8KB buffer for streaming

    loop {
        let bytes_read = response
            .read(&mut buffer)
            .map_err(|e| anyhow!("Failed to read response data: {}", e))?;

        if bytes_read == 0 {
            break; // End of stream
        }

        file.write_all(&buffer[..bytes_read])
            .map_err(|e| anyhow!("Failed to write to file {}: {}", output_path.display(), e))?;

        downloaded += bytes_read as u64;

        if let Some(ref pb) = progress_bar {
            pb.set_position(downloaded);
        }
    }

    // Explicitly flush and sync to ensure data is written to disk
    file.flush()
        .map_err(|e| anyhow!("Failed to flush file {}: {}", output_path.display(), e))?;

    file.sync_all()
        .map_err(|e| anyhow!("Failed to sync file {}: {}", output_path.display(), e))?;

    // Drop the file handle to ensure it's closed
    drop(file);

    // Clean up progress bar
    if let Some(pb) = progress_bar {
        pb.finish_and_clear();
        remove_progress_bar(&pb);
    }

    let actual_size_mb = downloaded as f64 / (1024.0 * 1024.0);
    log::debug!("📦 Downloaded {downloaded} bytes ({actual_size_mb:.2} MB)");

    if downloaded == 0 {
        return Err(anyhow!(
            "Downloaded file is empty (0 bytes). This usually indicates a network or server issue."
        ));
    }

    // Validate content length if provided
    if let Some(expected_length) = content_length {
        if downloaded != expected_length {
            log::warn!(
                "{}Size mismatch: expected {expected_length} bytes, got {downloaded} bytes",
                symbols::warning()
            );
        }
    }

    // Verify file was written correctly
    let written_metadata = fs::metadata(output_path)?;
    log::debug!(
        "{} File written successfully: {} bytes",
        symbols::save_file(),
        written_metadata.len()
    );

    if written_metadata.len() != downloaded {
        return Err(anyhow!(
            "File size mismatch after writing: expected {} bytes, file has {} bytes",
            downloaded,
            written_metadata.len()
        ));
    }

    log::info!(
        "{} Model downloaded to: {}",
        crate::color_utils::symbols::completed_successfully(),
        output_path.display()
    );

    Ok(())
}

/// Download model with concurrency protection using lock files
fn download_with_concurrency_protection(
    model_path: &Path,
    lock_path: &Path,
    model_info: &ModelInfo,
) -> Result<()> {
    const MAX_WAIT_TIME: Duration = Duration::from_secs(300); // 5 minutes max wait
    const INITIAL_WAIT: Duration = Duration::from_millis(50);
    const MAX_WAIT_INTERVAL: Duration = Duration::from_millis(1000);

    let start_time = std::time::Instant::now();
    let mut wait_duration = INITIAL_WAIT;

    loop {
        // Try to create lock file atomically
        // First make sure the directory exists
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(mut lock_file) => {
                // We got the lock, proceed with download
                log::debug!(
                    "{} Acquired download lock for {}",
                    symbols::lock_acquired(),
                    model_info.filename
                );

                // Write process info to lock file
                let lock_info = format!(
                    "locked by process {} at {}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                );
                let _ = write!(lock_file, "{lock_info}");
                let _ = lock_file.flush();

                // Download the model
                let result = download_model(&model_info.url, model_path);

                // Clean up lock file
                let _ = fs::remove_file(lock_path);
                log::debug!(
                    "{} Released download lock for {}",
                    symbols::lock_released(),
                    model_info.filename
                );

                return result;
            }
            Err(file_error) => {
                log::debug!(
                    "{} Failed to acquire lock: {}",
                    symbols::operation_failed(),
                    file_error
                );
                // Lock file exists, check if another process completed the download
                if model_path.exists() {
                    log::debug!(
                        "{} Checking if concurrent download completed...",
                        symbols::checking()
                    );
                    match cache_common::verify_checksum(model_path, &model_info.md5_checksum) {
                        Ok(true) => {
                            log::debug!(
                                "{} Model downloaded by another process, using cached version",
                                symbols::resources_found()
                            );
                            return Ok(());
                        }
                        Ok(false) => {
                            log::debug!(
                                "{}Concurrent download produced invalid checksum, will retry",
                                symbols::warning()
                            );
                            // Remove invalid file so we can retry
                            let _ = fs::remove_file(model_path);
                        }
                        Err(e) => {
                            log::debug!(
                                "{}Error checking concurrent download: {e}, will retry",
                                symbols::warning()
                            );
                        }
                    }
                }

                // Check timeout
                if start_time.elapsed() > MAX_WAIT_TIME {
                    // Force download after timeout, remove potentially stale lock
                    log::warn!(
                        "{} Download lock timeout ({}s), forcing download (removing stale lock)",
                        symbols::timeout(),
                        MAX_WAIT_TIME.as_secs()
                    );

                    // Try to get info about the lock file
                    if let Ok(lock_metadata) = fs::metadata(lock_path) {
                        let lock_age = lock_metadata
                            .modified()
                            .ok()
                            .and_then(|t| t.elapsed().ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        log::warn!("   Stale lock file age: {lock_age}s");
                    }

                    let _ = fs::remove_file(lock_path);
                    return download_model(&model_info.url, model_path);
                }

                // Wait with exponential backoff before retry
                let lock_info = fs::read_to_string(lock_path).ok().and_then(|content| {
                    // Extract PID from "locked by process {pid} at {timestamp}"
                    content
                        .split_whitespace()
                        .nth(3) // "locked", "by", "process", "{pid}", ...
                        .and_then(|pid_str| pid_str.parse::<u32>().ok())
                });

                let lock_msg = if let Some(pid) = lock_info {
                    format!("(lock held by PID {pid})")
                } else {
                    // We couldn't parse/read the lock file, so it is unlikely
                    // there is a valid process holding it
                    wait_duration /= 2;
                    "(lock details unavailable)".to_string()
                };

                log::debug!(
                    "{} Waiting for concurrent download to complete... ({}s elapsed, next check in {}ms) {}",
                    symbols::waiting(),
                    start_time.elapsed().as_secs(),
                    wait_duration.as_millis(),
                    lock_msg
                );
                std::thread::sleep(wait_duration);

                // Exponential backoff up to maximum
                wait_duration = std::cmp::min(wait_duration * 2, MAX_WAIT_INTERVAL);
            }
        }
    }
}

/// Get the cached model path, downloading if necessary
pub fn get_or_download_model(model_info: &ModelInfo) -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_path = cache_dir.join(&model_info.filename);
    let lock_path = cache_dir.join(format!("{}.lock", model_info.filename));

    log::debug!("🗂️  Cache directory: {}", cache_dir.display());
    log::debug!("📄 Model path: {}", model_path.display());

    // Check if model already exists and has correct checksum
    if model_path.exists() {
        log::debug!(
            "{} Checking cached model: {}",
            crate::color_utils::symbols::checking(),
            model_path.display()
        );

        // Quick sanity check: verify file is not empty before expensive MD5
        if let Ok(metadata) = fs::metadata(&model_path) {
            let file_size = metadata.len();
            if file_size == 0 {
                log::warn!(
                    "{}Cached model file is empty, re-downloading",
                    symbols::warning()
                );
                fs::remove_file(&model_path)?;
            } else {
                // File exists and has content, now verify checksum
                match cache_common::verify_checksum(&model_path, &model_info.md5_checksum) {
                    Ok(true) => {
                        // Fast path: model is cached and valid
                        log::info!(
                            "{} Using cached model: {}",
                            crate::color_utils::symbols::completed_successfully(),
                            model_info.filename
                        );
                        return Ok(model_path);
                    }
                    Ok(false) => {
                        // Get detailed info about the file for debugging
                        let file_info = get_file_info(&model_path)
                            .unwrap_or_else(|e| format!("Error getting file info: {e}"));
                        let actual_checksum = cache_common::calculate_md5(&model_path)
                            .unwrap_or_else(|e| format!("Error calculating checksum: {e}"));

                        log::warn!("{}Cached model has invalid checksum, re-downloading\n   Expected: {}\n   Actual:   {}\n   File info: {}",
                        symbols::warning(),
                        model_info.md5_checksum, actual_checksum, file_info);

                        fs::remove_file(&model_path)?;
                    }
                    Err(e) => {
                        let colored_error: String =
                            crate::color_utils::colors::error_level(&e.to_string());
                        let file_info = get_file_info(&model_path)
                            .unwrap_or_else(|e| format!("Error getting file info: {e}"));

                        log::warn!(
                            "{}Error verifying checksum: {colored_error}, re-downloading",
                            symbols::warning()
                        );
                        log::warn!("   File info: {file_info}");

                        fs::remove_file(&model_path)?;
                    }
                }
            }
        } else {
            log::debug!(
                "{}Cannot read file metadata, treating as missing",
                symbols::warning()
            );
        }
    }
    // Handle concurrent downloads with lock file
    download_with_concurrency_protection(&model_path, &lock_path, model_info)?;

    // Verify the downloaded model
    log::debug!(
        "{} Verifying downloaded model checksum...",
        symbols::checking()
    );
    let actual_checksum = cache_common::calculate_md5(&model_path)?;
    let file_info = get_file_info(&model_path)?;

    log::debug!("   Expected checksum: {}", model_info.md5_checksum);
    log::debug!("   Actual checksum:   {actual_checksum}");
    log::debug!("   File info: {file_info}");

    if actual_checksum != model_info.md5_checksum {
        fs::remove_file(&model_path)?;

        // Provide comprehensive error information for debugging
        return Err(anyhow!(
            "Downloaded model failed checksum verification.\n\
             Expected checksum: {}\n\
             Actual checksum:   {}\n\
             File details: {}\n\
             Model URL: {}\n\
             Cache directory: {}\n\
             \n\
             Possible causes:\n\
             - Network corruption during download\n\
             - Disk space or permission issues\n\
             - Cache corruption\n\
             - Model file was updated but checksum wasn't\n\
             \n\
             Try clearing the cache or setting FORCE_DOWNLOAD=1",
            model_info.md5_checksum,
            actual_checksum,
            file_info,
            model_info.url,
            model_path
                .parent()
                .unwrap_or_else(|| Path::new("unknown"))
                .display()
        ));
    }

    log::info!(
        "{} Model downloaded and verified successfully",
        crate::color_utils::symbols::completed_successfully()
    );

    Ok(model_path)
}

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
            std::env::var(env_var).unwrap_or_else(|_| base_info.url.clone())
        } else {
            base_info.url.clone()
        };

        let checksum = if let Some(env_var) = checksum_env_var {
            std::env::var(env_var).unwrap_or_else(|_| base_info.md5_checksum.clone())
        } else {
            base_info.md5_checksum.clone()
        };

        RuntimeModelInfo {
            name: base_info.name.clone(),
            url,
            md5_checksum: checksum,
            filename: base_info.filename.clone(),
        }
    }

    /// Convert to ModelInfo
    pub fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            url: self.url.clone(),
            md5_checksum: self.md5_checksum.clone(),
            filename: self.filename.clone(),
        }
    }
}

/// Get or download a model using RuntimeModelInfo (supports env var overrides).
/// This now properly uses owned strings without memory leaks.
pub fn get_or_download_runtime_model(model_info: RuntimeModelInfo) -> Result<PathBuf> {
    let model_info = model_info.to_model_info();
    get_or_download_model(&model_info)
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

    /// Get default model info for downloading from remote sources.
    /// Returns None for models that don't support remote download.
    fn get_default_model_info() -> Option<ModelInfo> {
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
    if let Some(default_model_info) = T::get_default_model_info() {
        // Create RuntimeModelInfo with potential URL/checksum overrides
        let runtime_model_info = RuntimeModelInfo::from_model_info_with_overrides(
            &default_model_info,
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

#[cfg(test)]
mod tests {
    use super::*;

    // Model cache tests - testing core functionality without env var modification
    #[test]
    fn test_cache_dir_basic() {
        // Test basic cache dir functionality (without modifying env vars)
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("onnx-models"));
    }

    #[test]
    fn test_coreml_cache_dir() {
        let cache_dir = get_coreml_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("beaker"));
        assert!(cache_dir.to_string_lossy().contains("coreml"));
    }

    #[test]
    fn test_runtime_model_info_creation() {
        let base_info = ModelInfo {
            name: "test-model".to_string(),
            url: "https://example.com/model.onnx".to_string(),
            md5_checksum: "abcd1234".to_string(),
            filename: "model.onnx".to_string(),
        };

        // Test without any env vars (should use base info)
        let runtime_info = RuntimeModelInfo::from_model_info_with_overrides(
            &base_info,
            Some("NONEXISTENT_URL"),
            Some("NONEXISTENT_CHECKSUM"),
        );

        assert_eq!(runtime_info.name, base_info.name);
        assert_eq!(runtime_info.url, base_info.url);
        assert_eq!(runtime_info.md5_checksum, base_info.md5_checksum);
        assert_eq!(runtime_info.filename, base_info.filename);
    }

    #[test]
    fn test_runtime_to_model_info_conversion() {
        let runtime_info = RuntimeModelInfo {
            name: "test-model".to_string(),
            url: "https://example.com/model.onnx".to_string(),
            md5_checksum: "abcd1234".to_string(),
            filename: "model.onnx".to_string(),
        };

        let model_info = runtime_info.to_model_info();

        assert_eq!(model_info.name, "test-model");
        assert_eq!(model_info.url, "https://example.com/model.onnx");
        assert_eq!(model_info.md5_checksum, "abcd1234");
        assert_eq!(model_info.filename, "model.onnx");
    }
}
