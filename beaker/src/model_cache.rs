use anyhow::{anyhow, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Model information for caching and verification
#[derive(Debug)]
pub struct ModelInfo {
    pub name: &'static str,
    pub url: &'static str,
    pub md5_checksum: &'static str,
    pub filename: &'static str,
}

/// Get the cache directory for storing downloaded models
pub fn get_cache_dir() -> Result<PathBuf> {
    // Check if ONNX_MODEL_CACHE_DIR environment variable is set (same as build.rs)
    if let Ok(cache_dir) = std::env::var("ONNX_MODEL_CACHE_DIR") {
        // Handle tilde expansion for home directory
        if let Some(stripped) = cache_dir.strip_prefix("~/") {
            if let Some(home_dir) = dirs::home_dir() {
                return Ok(home_dir.join(stripped));
            }
        }
        return Ok(PathBuf::from(cache_dir));
    }

    // Fallback to default cache directory
    dirs::cache_dir()
        .map(|dir| dir.join("onnx-models"))
        .ok_or_else(|| anyhow!("Unable to determine cache directory"))
}

/// Get the CoreML cache directory for storing compiled CoreML models
pub fn get_coreml_cache_dir() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|dir| dir.join("beaker").join("coreml"))
        .ok_or_else(|| anyhow!("Unable to determine CoreML cache directory"))
}

/// Calculate MD5 hash of a file
fn calculate_md5(path: &Path) -> Result<String> {
    let contents = fs::read(path)?;
    let mut hasher = md5::Context::new();
    hasher.consume(&contents);
    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Verify the checksum of a downloaded model
fn verify_checksum(path: &Path, expected_md5: &str) -> Result<bool> {
    let actual_md5 = calculate_md5(path)?;
    Ok(actual_md5 == expected_md5)
}

/// Get detailed file information for debugging
fn get_file_info(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)?;
    let size = metadata.len();
    let size_mb = size as f64 / (1024.0 * 1024.0);

    Ok(format!(
        "size: {} bytes ({:.2} MB), permissions: {:?}, modified: {:?}",
        size,
        size_mb,
        metadata.permissions(),
        metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    ))
}

/// Download a model from URL to the specified path
fn download_model(url: &str, output_path: &Path) -> Result<()> {
    log::info!("📥 Downloading model from: {url}");

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
        log::debug!("📁 Created parent directory: {}", parent.display());
    }

    // Check available disk space before download
    if let Ok(parent) = output_path
        .parent()
        .ok_or_else(|| anyhow!("No parent directory"))
    {
        if let Ok(_metadata) = fs::metadata(parent) {
            log::debug!("📊 Parent directory exists");
        }
    }

    // Download the file with better error handling and redirect support
    log::debug!("🌐 Starting HTTP request...");
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| anyhow!("Failed to send HTTP request: {}", e))?;

    let status = response.status();
    log::debug!("📡 HTTP response status: {status}");

    if !status.is_success() {
        return Err(anyhow!("HTTP request failed with status: {}", status));
    }

    let content_length = response.content_length();

    if let Some(length) = content_length {
        let length_mb = length as f64 / (1024.0 * 1024.0);
        log::debug!("📏 Expected download size: {length} bytes ({length_mb:.2} MB)");
    } else {
        log::warn!("⚠️  Content-Length header missing, unable to determine file size");
    }

    log::debug!("📦 Reading response bytes...");
    let content = response
        .bytes()
        .map_err(|e| anyhow!("Failed to read response bytes: {}", e))?;

    let actual_size = content.len();
    let actual_size_mb = actual_size as f64 / (1024.0 * 1024.0);

    log::info!("📦 Downloaded {actual_size} bytes ({actual_size_mb:.2} MB)");

    if actual_size == 0 {
        return Err(anyhow!(
            "Downloaded file is empty (0 bytes). This usually indicates a network or server issue."
        ));
    }

    // Validate content length if provided
    if let Some(expected_length) = content_length {
        if actual_size != expected_length as usize {
            log::warn!(
                "⚠️  Size mismatch: expected {expected_length} bytes, got {actual_size} bytes"
            );
        }
    }

    // Write to file with explicit flushing
    log::debug!("💾 Writing to file...");
    let mut file = fs::File::create(output_path).map_err(|e| {
        anyhow!(
            "Failed to create output file {}: {}",
            output_path.display(),
            e
        )
    })?;

    file.write_all(&content)
        .map_err(|e| anyhow!("Failed to write to file {}: {}", output_path.display(), e))?;

    // Explicitly flush and sync to ensure data is written to disk
    file.flush()
        .map_err(|e| anyhow!("Failed to flush file {}: {}", output_path.display(), e))?;

    file.sync_all()
        .map_err(|e| anyhow!("Failed to sync file {}: {}", output_path.display(), e))?;

    // Drop the file handle to ensure it's closed
    drop(file);

    // Verify file was written correctly
    let written_metadata = fs::metadata(output_path)?;
    log::debug!(
        "💾 File written successfully: {} bytes",
        written_metadata.len()
    );

    if written_metadata.len() != actual_size as u64 {
        return Err(anyhow!(
            "File size mismatch after writing: expected {} bytes, file has {} bytes",
            actual_size,
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

/// Get the cached model path, downloading if necessary
pub fn get_or_download_model(model_info: &ModelInfo) -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_path = cache_dir.join(model_info.filename);
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

        match verify_checksum(&model_path, model_info.md5_checksum) {
            Ok(true) => {
                log::debug!(
                    "{} Using cached model with valid checksum",
                    crate::color_utils::symbols::completed_successfully()
                );
                return Ok(model_path);
            }
            Ok(false) => {
                // Get detailed info about the file for debugging
                let file_info = get_file_info(&model_path)
                    .unwrap_or_else(|e| format!("Error getting file info: {e}"));
                let actual_checksum = calculate_md5(&model_path)
                    .unwrap_or_else(|e| format!("Error calculating checksum: {e}"));

                log::warn!("⚠️  Cached model has invalid checksum, re-downloading\n   Expected: {}\n   Actual:   {}\n   File info: {}",
                          model_info.md5_checksum, actual_checksum, file_info);

                fs::remove_file(&model_path)?;
            }
            Err(e) => {
                let colored_error: String = crate::color_utils::colors::error_level(&e.to_string());
                let file_info = get_file_info(&model_path)
                    .unwrap_or_else(|e| format!("Error getting file info: {e}"));

                log::warn!("⚠️  Error verifying checksum: {colored_error}, re-downloading");
                log::warn!("   File info: {file_info}");

                fs::remove_file(&model_path)?;
            }
        }
    }

    // Handle concurrent downloads with lock file
    download_with_concurrency_protection(&model_path, &lock_path, model_info)?;

    // Verify the downloaded model
    log::debug!("🔍 Verifying downloaded model checksum...");
    let actual_checksum = calculate_md5(&model_path)?;
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

/// Download model with concurrency protection using lock files
fn download_with_concurrency_protection(
    model_path: &Path,
    lock_path: &Path,
    model_info: &ModelInfo,
) -> Result<()> {
    const MAX_WAIT_TIME: Duration = Duration::from_secs(300); // 5 minutes max wait
    const POLL_INTERVAL: Duration = Duration::from_millis(500);

    let start_time = std::time::Instant::now();

    loop {
        // Try to create lock file atomically
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(mut lock_file) => {
                // We got the lock, proceed with download
                log::debug!("🔒 Acquired download lock for {}", model_info.filename);

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
                let result = download_model(model_info.url, model_path);

                // Clean up lock file
                let _ = fs::remove_file(lock_path);
                log::debug!("🔓 Released download lock for {}", model_info.filename);

                return result;
            }
            Err(_) => {
                // Lock file exists, check if another process completed the download
                if model_path.exists() {
                    log::debug!("🔍 Checking if concurrent download completed...");
                    match verify_checksum(model_path, model_info.md5_checksum) {
                        Ok(true) => {
                            log::debug!(
                                "🎯 Model downloaded by another process, using cached version"
                            );
                            return Ok(());
                        }
                        Ok(false) => {
                            log::debug!(
                                "⚠️  Concurrent download produced invalid checksum, will retry"
                            );
                            // Remove invalid file so we can retry
                            let _ = fs::remove_file(model_path);
                        }
                        Err(e) => {
                            log::debug!("⚠️  Error checking concurrent download: {e}, will retry");
                        }
                    }
                }

                // Check timeout
                if start_time.elapsed() > MAX_WAIT_TIME {
                    // Force download after timeout, remove potentially stale lock
                    log::warn!(
                        "⏰ Download lock timeout ({}s), forcing download (removing stale lock)",
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
                    return download_model(model_info.url, model_path);
                }

                // Wait and retry
                log::debug!(
                    "⏳ Waiting for concurrent download to complete... ({}s elapsed)",
                    start_time.elapsed().as_secs()
                );
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_dir() {
        // Save original environment variable state
        let original_var = std::env::var("ONNX_MODEL_CACHE_DIR");

        // Test without ONNX_MODEL_CACHE_DIR set
        std::env::remove_var("ONNX_MODEL_CACHE_DIR");
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("onnx-models"));

        // Test with ONNX_MODEL_CACHE_DIR set
        let custom_cache = "/tmp/custom-onnx-cache";
        std::env::set_var("ONNX_MODEL_CACHE_DIR", custom_cache);
        let cache_dir_custom = get_cache_dir().unwrap();
        assert_eq!(cache_dir_custom.to_string_lossy(), custom_cache);

        // Test with tilde expansion
        std::env::set_var("ONNX_MODEL_CACHE_DIR", "~/.cache/test-onnx");
        let cache_dir_tilde = get_cache_dir().unwrap();
        // Should not contain literal tilde
        assert!(!cache_dir_tilde.to_string_lossy().contains("~"));
        // Should contain the expanded path
        assert!(cache_dir_tilde
            .to_string_lossy()
            .contains(".cache/test-onnx"));

        // Restore original environment variable state
        match original_var {
            Ok(val) => std::env::set_var("ONNX_MODEL_CACHE_DIR", val),
            Err(_) => std::env::remove_var("ONNX_MODEL_CACHE_DIR"),
        }
    }

    #[test]
    fn test_md5_calculation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "hello world").unwrap();

        let md5 = calculate_md5(&file_path).unwrap();
        assert_eq!(md5, "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    fn test_coreml_cache_dir() {
        let cache_dir = get_coreml_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("beaker"));
        assert!(cache_dir.to_string_lossy().contains("coreml"));
    }
}
