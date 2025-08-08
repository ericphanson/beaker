use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Common cache directory resolution with environment variable support
pub fn get_cache_base_dir() -> Result<PathBuf> {
    dirs::cache_dir().ok_or_else(|| anyhow!("Unable to determine cache directory"))
}

/// Get cache directory with environment variable override and tilde expansion
pub fn get_cache_dir_with_env_override(env_var: &str, default_subdir: &str) -> Result<PathBuf> {
    // Check if environment variable is set
    if let Ok(cache_dir) = std::env::var(env_var) {
        // Handle tilde expansion for home directory
        if let Some(stripped) = cache_dir.strip_prefix("~/") {
            if let Some(home_dir) = dirs::home_dir() {
                return Ok(home_dir.join(stripped));
            }
        }
        // Make sure the directory exists
        let path = PathBuf::from(cache_dir);
        if !path.exists() {
            fs::create_dir_all(&path)?;
        }
        return Ok(path);
    }

    // Fallback to default cache directory
    get_cache_base_dir().map(|dir| dir.join(default_subdir))
}

/// Calculate MD5 hash of a file
pub fn calculate_md5(path: &Path) -> Result<String> {
    let contents = fs::read(path)?;
    Ok(calculate_md5_bytes(&contents))
}

/// Calculate MD5 hash of bytes
pub fn calculate_md5_bytes(bytes: &[u8]) -> String {
    let mut hasher = md5::Context::new();
    hasher.consume(bytes);
    let result = hasher.finalize();
    format!("{result:x}")
}

/// Verify the checksum of a file
pub fn verify_checksum(path: &Path, expected_md5: &str) -> Result<bool> {
    let actual_md5 = calculate_md5(path)?;
    Ok(actual_md5 == expected_md5)
}

/// Get detailed file information for debugging
pub fn get_file_info(path: &Path) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    #[test]
    fn test_md5_calculation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "hello world").unwrap();

        let md5 = calculate_md5(&file_path).unwrap();
        assert_eq!(md5, "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    fn test_md5_bytes() {
        let md5 = calculate_md5_bytes(b"hello world");
        assert_eq!(md5, "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    #[serial]
    fn test_cache_dir_with_env() {
        // Save original environment variable state
        let original_var = std::env::var("TEST_CACHE_DIR");

        // Test without environment variable set
        std::env::remove_var("TEST_CACHE_DIR");
        let cache_dir = get_cache_dir_with_env_override("TEST_CACHE_DIR", "test-subdir").unwrap();
        assert!(cache_dir.to_string_lossy().contains("test-subdir"));

        // Test with environment variable set
        let custom_cache = "/tmp/custom-cache";
        std::env::set_var("TEST_CACHE_DIR", custom_cache);
        let cache_dir_custom =
            get_cache_dir_with_env_override("TEST_CACHE_DIR", "test-subdir").unwrap();
        assert_eq!(cache_dir_custom.to_string_lossy(), custom_cache);

        // Test with tilde expansion
        std::env::set_var("TEST_CACHE_DIR", "~/.cache/test");
        let cache_dir_tilde =
            get_cache_dir_with_env_override("TEST_CACHE_DIR", "test-subdir").unwrap();
        // Should not contain literal tilde
        assert!(!cache_dir_tilde.to_string_lossy().contains("~"));
        // Should contain the expanded path
        assert!(cache_dir_tilde.to_string_lossy().contains(".cache/test"));

        // Restore original environment variable state
        match original_var {
            Ok(val) => std::env::set_var("TEST_CACHE_DIR", val),
            Err(_) => std::env::remove_var("TEST_CACHE_DIR"),
        }
    }
}
