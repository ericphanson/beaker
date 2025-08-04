use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for image collection behavior
#[derive(Debug, Clone)]
pub struct ImageInputConfig {
    pub require_glob_matches: bool,
    pub strict_mode: bool,
}

impl Default for ImageInputConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            require_glob_matches: true,
        }
    }
}

impl ImageInputConfig {
    /// Create a configuration for strict mode (like head_detection.rs)
    pub fn strict() -> Self {
        Self {
            strict_mode: true,
            require_glob_matches: true,
        }
    }

    /// Create a configuration for permissive mode (like cutout_processing.rs)
    pub fn permissive() -> Self {
        Self {
            strict_mode: false,
            require_glob_matches: false,
        }
    }

    /// Create a configuration based on strict flag
    /// If strict=true, uses strict mode; if strict=false, uses permissive mode
    pub fn from_strict_flag(strict: bool) -> Self {
        if strict {
            Self::strict()
        } else {
            Self::permissive()
        }
    }
}

/// Check if a file is a supported image format
/// Supports: jpg, jpeg, png, webp, bmp, tiff, tif
pub fn is_supported_image_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        matches!(
            ext_lower.as_str(),
            "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff" | "tif"
        )
    } else {
        false
    }
}

/// Find all image files in a directory (non-recursive)
pub fn find_images_in_directory(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut image_files = Vec::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && is_supported_image_file(&path) {
            image_files.push(path);
        }
    }

    // Sort for consistent ordering
    image_files.sort();
    Ok(image_files)
}

/// Collect all image files from multiple sources (files, directories, or glob patterns)
pub fn collect_images_from_sources(
    sources: &[String],
    config: &ImageInputConfig,
) -> Result<Vec<PathBuf>> {
    let mut all_image_files = Vec::new();

    for source in sources {
        let source_path = Path::new(source);

        if source_path.is_file() {
            // Single file exists
            if is_supported_image_file(source_path) {
                all_image_files.push(source_path.to_path_buf());
            } else if config.strict_mode {
                return Err(anyhow::anyhow!(
                    "File is not a supported image format: {}",
                    source_path.display()
                ));
            }
            // In permissive mode, silently skip unsupported files
        } else if source_path.is_dir() {
            // Directory - find all images inside
            let dir_images = find_images_in_directory(source_path)?;
            all_image_files.extend(dir_images);
        } else if !source.contains('*') && !source.contains('?') && !source.contains('[') {
            // Looks like a simple file path (not a glob pattern) but doesn't exist
            if config.strict_mode {
                return Err(anyhow::anyhow!("File does not exist: {}", source));
            } else {
                // In permissive mode, report the missing file as a warning
                log::warn!(
                    "{}File does not exist: {}",
                    crate::color_utils::symbols::warning(),
                    source
                );
            }
        } else {
            // Could be a glob pattern
            match glob::glob(source) {
                Ok(paths) => {
                    let mut found_any = false;
                    for path_result in paths {
                        match path_result {
                            Ok(path) => {
                                if path.is_file() && is_supported_image_file(&path) {
                                    all_image_files.push(path);
                                    found_any = true;
                                }
                            }
                            Err(e) => {
                                log::warn!(
                                    "{}Warning: Error reading path in glob {source}: {e}",
                                    crate::color_utils::symbols::warning()
                                );
                            }
                        }
                    }
                    if !found_any && config.require_glob_matches {
                        return Err(anyhow::anyhow!(
                            "No image files found matching pattern: {}",
                            source
                        ));
                    }
                }
                Err(_) => {
                    // Not a valid glob pattern, treat as non-existent path
                    if config.strict_mode {
                        return Err(anyhow::anyhow!(
                            "Source path does not exist and is not a valid glob pattern: {}",
                            source
                        ));
                    } else {
                        // In permissive mode, report the missing file as a warning
                        log::warn!(
                            "{}Source path does not exist: {}",
                            crate::color_utils::symbols::warning(),
                            source
                        );
                    }
                }
            }
        }
    }

    // Sort all collected files for consistent ordering
    all_image_files.sort();

    // Remove duplicates (in case same file is specified multiple ways)
    all_image_files.dedup();

    // Check if we found any images - fail in strict mode, succeed in permissive mode
    if all_image_files.is_empty() && config.strict_mode {
        return Err(anyhow::anyhow!(
            "No image files found in the specified sources"
        ));
    }

    Ok(all_image_files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_is_supported_image_file() {
        assert!(is_supported_image_file(Path::new("test.jpg")));
        assert!(is_supported_image_file(Path::new("test.jpeg")));
        assert!(is_supported_image_file(Path::new("test.png")));
        assert!(is_supported_image_file(Path::new("test.webp")));
        assert!(is_supported_image_file(Path::new("test.bmp")));
        assert!(is_supported_image_file(Path::new("test.tiff")));
        assert!(is_supported_image_file(Path::new("test.tif")));

        assert!(is_supported_image_file(Path::new("TEST.JPG"))); // Case insensitive

        assert!(!is_supported_image_file(Path::new("test.txt")));
        assert!(!is_supported_image_file(Path::new("test.gif")));
        assert!(!is_supported_image_file(Path::new("test")));
    }

    #[test]
    fn test_find_images_in_directory() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Create test files
        fs::write(dir_path.join("image1.jpg"), b"fake image").unwrap();
        fs::write(dir_path.join("image2.png"), b"fake image").unwrap();
        fs::write(dir_path.join("not_image.txt"), b"text file").unwrap();

        let images = find_images_in_directory(dir_path).unwrap();
        assert_eq!(images.len(), 2);
        assert!(images
            .iter()
            .any(|p| p.file_name().unwrap() == "image1.jpg"));
        assert!(images
            .iter()
            .any(|p| p.file_name().unwrap() == "image2.png"));
    }

    #[test]
    fn test_collect_images_strict_mode() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Create test files
        let image_path = dir_path.join("test.jpg");
        let text_path = dir_path.join("test.txt");
        fs::write(&image_path, b"fake image").unwrap();
        fs::write(&text_path, b"text file").unwrap();

        let config = ImageInputConfig::strict();

        // Should succeed with image file
        let sources = vec![image_path.to_string_lossy().to_string()];
        let result = collect_images_from_sources(&sources, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);

        // Should fail with non-image file
        let sources = vec![text_path.to_string_lossy().to_string()];
        let result = collect_images_from_sources(&sources, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_images_permissive_mode() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Create test files
        let image_path = dir_path.join("test.jpg");
        let text_path = dir_path.join("test.txt");
        fs::write(&image_path, b"fake image").unwrap();
        fs::write(&text_path, b"text file").unwrap();

        let config = ImageInputConfig::permissive();

        // Should succeed with mixed files, but only return image
        let sources = vec![
            image_path.to_string_lossy().to_string(),
            text_path.to_string_lossy().to_string(),
        ];
        let result = collect_images_from_sources(&sources, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_collect_images_directory() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Create test files
        fs::write(dir_path.join("image1.jpg"), b"fake image").unwrap();
        fs::write(dir_path.join("image2.png"), b"fake image").unwrap();
        fs::write(dir_path.join("not_image.txt"), b"text file").unwrap();

        let config = ImageInputConfig::default();
        let sources = vec![dir_path.to_string_lossy().to_string()];
        let result = collect_images_from_sources(&sources, &config).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result
            .iter()
            .any(|p| p.file_name().unwrap() == "image1.jpg"));
        assert!(result
            .iter()
            .any(|p| p.file_name().unwrap() == "image2.png"));
    }

    #[test]
    fn test_from_strict_flag() {
        // Test strict=true creates strict config
        let strict_config = ImageInputConfig::from_strict_flag(true);
        assert!(strict_config.strict_mode);
        assert!(strict_config.require_glob_matches);

        // Test strict=false creates permissive config
        let permissive_config = ImageInputConfig::from_strict_flag(false);
        assert!(!permissive_config.strict_mode);
        assert!(!permissive_config.require_glob_matches);
    }
}
