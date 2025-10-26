pub mod detection;
pub mod directory;
pub mod welcome;

pub use detection::DetectionView;
pub use directory::DirectoryView;
#[allow(unused_imports)] // Re-exported for tests
pub use directory::ProcessingStatus;
pub use welcome::{WelcomeAction, WelcomeView};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directory_view_module_accessible() {
        use std::path::PathBuf;
        let _view = DirectoryView::new(PathBuf::from("/tmp"), vec![]);
        // If this compiles, the module is properly exported
    }
}
