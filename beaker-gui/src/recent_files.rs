use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_RECENT_FILES: usize = 10;
const RECENT_FILES_FILENAME: &str = "recent.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecentItemType {
    Image,
    Folder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentItem {
    pub path: PathBuf,
    pub item_type: RecentItemType,
    pub timestamp: String, // ISO 8601 format
}

#[derive(Debug, Serialize, Deserialize)]
struct RecentFilesData {
    items: Vec<RecentItem>,
}

pub struct RecentFiles {
    config_path: PathBuf,
    items: Vec<RecentItem>,
}

impl RecentFiles {
    /// Create a new RecentFiles manager
    pub fn new() -> Result<Self> {
        let config_path = Self::get_config_path()?;
        let items = Self::load_from_disk(&config_path).unwrap_or_default();

        Ok(Self { config_path, items })
    }

    /// Get the config directory path
    fn get_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;

        let beaker_config_dir = config_dir.join("beaker-gui");
        std::fs::create_dir_all(&beaker_config_dir)?;

        Ok(beaker_config_dir.join(RECENT_FILES_FILENAME))
    }

    /// Load recent files from disk
    fn load_from_disk(path: &Path) -> Result<Vec<RecentItem>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let data = std::fs::read_to_string(path)?;
        let recent_data: RecentFilesData = serde_json::from_str(&data)?;
        Ok(recent_data.items)
    }

    /// Save recent files to disk
    fn save_to_disk(&self) -> Result<()> {
        let data = RecentFilesData {
            items: self.items.clone(),
        };
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(&self.config_path, json)?;
        Ok(())
    }

    /// Add a new item to recent files (most recent first)
    pub fn add(&mut self, path: PathBuf, item_type: RecentItemType) -> Result<()> {
        // Get current timestamp in ISO 8601 format
        let timestamp = chrono::Utc::now().to_rfc3339();

        // Remove existing entry for this path if present
        self.items.retain(|item| item.path != path);

        // Add to front
        self.items.insert(
            0,
            RecentItem {
                path,
                item_type,
                timestamp,
            },
        );

        // Trim to max size
        self.items.truncate(MAX_RECENT_FILES);

        // Save to disk
        self.save_to_disk()?;

        Ok(())
    }

    /// Get all recent items
    pub fn items(&self) -> &[RecentItem] {
        &self.items
    }

    /// Clear all recent files
    #[allow(dead_code)]
    pub fn clear(&mut self) -> Result<()> {
        self.items.clear();
        self.save_to_disk()?;
        Ok(())
    }

    /// Remove a specific item by path
    #[allow(dead_code)]
    pub fn remove(&mut self, path: &Path) -> Result<()> {
        self.items.retain(|item| item.path != path);
        self.save_to_disk()?;
        Ok(())
    }
}

impl Default for RecentFiles {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            config_path: PathBuf::new(),
            items: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_recent_files() -> (RecentFiles, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(RECENT_FILES_FILENAME);

        let recent_files = RecentFiles {
            config_path,
            items: Vec::new(),
        };

        (recent_files, temp_dir)
    }

    #[test]
    fn test_add_recent_file() {
        let (mut recent_files, _temp_dir) = create_test_recent_files();

        recent_files
            .add(PathBuf::from("/test/image.jpg"), RecentItemType::Image)
            .unwrap();

        assert_eq!(recent_files.items().len(), 1);
        assert_eq!(
            recent_files.items()[0].path,
            PathBuf::from("/test/image.jpg")
        );
        assert_eq!(recent_files.items()[0].item_type, RecentItemType::Image);
    }

    #[test]
    fn test_add_duplicate_moves_to_front() {
        let (mut recent_files, _temp_dir) = create_test_recent_files();

        recent_files
            .add(PathBuf::from("/test/image1.jpg"), RecentItemType::Image)
            .unwrap();
        recent_files
            .add(PathBuf::from("/test/image2.jpg"), RecentItemType::Image)
            .unwrap();
        recent_files
            .add(PathBuf::from("/test/image1.jpg"), RecentItemType::Image)
            .unwrap();

        assert_eq!(recent_files.items().len(), 2);
        assert_eq!(
            recent_files.items()[0].path,
            PathBuf::from("/test/image1.jpg")
        );
        assert_eq!(
            recent_files.items()[1].path,
            PathBuf::from("/test/image2.jpg")
        );
    }

    #[test]
    fn test_max_recent_files() {
        let (mut recent_files, _temp_dir) = create_test_recent_files();

        // Add more than MAX_RECENT_FILES
        for i in 0..15 {
            recent_files
                .add(
                    PathBuf::from(format!("/test/image{}.jpg", i)),
                    RecentItemType::Image,
                )
                .unwrap();
        }

        assert_eq!(recent_files.items().len(), MAX_RECENT_FILES);
        assert_eq!(
            recent_files.items()[0].path,
            PathBuf::from("/test/image14.jpg")
        );
    }

    #[test]
    fn test_clear() {
        let (mut recent_files, _temp_dir) = create_test_recent_files();

        recent_files
            .add(PathBuf::from("/test/image.jpg"), RecentItemType::Image)
            .unwrap();
        assert_eq!(recent_files.items().len(), 1);

        recent_files.clear().unwrap();
        assert_eq!(recent_files.items().len(), 0);
    }

    #[test]
    fn test_remove() {
        let (mut recent_files, _temp_dir) = create_test_recent_files();

        recent_files
            .add(PathBuf::from("/test/image1.jpg"), RecentItemType::Image)
            .unwrap();
        recent_files
            .add(PathBuf::from("/test/image2.jpg"), RecentItemType::Image)
            .unwrap();

        recent_files
            .remove(&PathBuf::from("/test/image1.jpg"))
            .unwrap();

        assert_eq!(recent_files.items().len(), 1);
        assert_eq!(
            recent_files.items()[0].path,
            PathBuf::from("/test/image2.jpg")
        );
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(RECENT_FILES_FILENAME);

        // Create and add items
        {
            let mut recent_files = RecentFiles {
                config_path: config_path.clone(),
                items: Vec::new(),
            };

            recent_files
                .add(PathBuf::from("/test/image.jpg"), RecentItemType::Image)
                .unwrap();
            recent_files
                .add(PathBuf::from("/test/folder"), RecentItemType::Folder)
                .unwrap();
        }

        // Load from disk in new instance
        {
            let items = RecentFiles::load_from_disk(&config_path).unwrap();
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].path, PathBuf::from("/test/folder"));
            assert_eq!(items[0].item_type, RecentItemType::Folder);
            assert_eq!(items[1].path, PathBuf::from("/test/image.jpg"));
            assert_eq!(items[1].item_type, RecentItemType::Image);
        }
    }
}
