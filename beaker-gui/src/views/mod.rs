pub mod detection;
pub mod welcome;
mod directory;

pub use detection::DetectionView;
pub use welcome::{WelcomeAction, WelcomeView};
pub use directory::{DirectoryView, ProcessingStatus, ImageState};

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
