use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Shared metadata structure that can contain both head and cutout results
#[derive(Serialize, Deserialize, Default)]
pub struct BeakerMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cutout: Option<serde_json::Value>,
}

/// Load existing metadata from a file, or create new empty metadata
pub fn load_or_create_metadata(path: &Path) -> Result<BeakerMetadata> {
    if path.exists() {
        let content = fs::read_to_string(path)?;
        let metadata: BeakerMetadata = toml::from_str(&content)?;
        Ok(metadata)
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

    let toml_content = toml::to_string_pretty(metadata)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_metadata_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let metadata_path = temp_dir.path().join("test.beaker.toml");

        // Create and save metadata
        let metadata = BeakerMetadata::default();
        save_metadata(&metadata, &metadata_path).unwrap();

        // Load and verify
        let loaded = load_or_create_metadata(&metadata_path).unwrap();
        assert!(loaded.head.is_none());
        assert!(loaded.cutout.is_none());
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
}
