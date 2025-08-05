use anyhow::{anyhow, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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
    dirs::cache_dir()
        .map(|dir| dir.join("beaker").join("models"))
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

/// Download a model from URL to the specified path
fn download_model(url: &str, output_path: &Path) -> Result<()> {
    log::info!("📥 Downloading model from: {url}");

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download the file
    let response = reqwest::blocking::get(url)?;
    let content = response.bytes()?;

    // Write to file
    let mut file = fs::File::create(output_path)?;
    file.write_all(&content)?;

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
                log::warn!("⚠️  Cached model has invalid checksum, re-downloading");
                fs::remove_file(&model_path)?;
            }
            Err(e) => {
                let colored_error: String = crate::color_utils::colors::error_level(&e.to_string());
                log::warn!("⚠️  Error verifying checksum: {colored_error}, re-downloading");
                fs::remove_file(&model_path)?;
            }
        }
    }

    // Download the model
    download_model(model_info.url, &model_path)?;

    // Verify the downloaded model
    if !verify_checksum(&model_path, model_info.md5_checksum)? {
        fs::remove_file(&model_path)?;
        return Err(anyhow!(
            "Downloaded model failed checksum verification. Expected: {}, got different hash.",
            model_info.md5_checksum
        ));
    }

    log::info!(
        "{} Model downloaded and verified successfully",
        crate::color_utils::symbols::completed_successfully()
    );

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("beaker"));
        assert!(cache_dir.to_string_lossy().contains("models"));
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
