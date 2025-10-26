// Library exports for testing

pub mod app;
pub mod recent_files;
pub mod style;
pub mod views;

pub use app::{BeakerApp, View};
pub use recent_files::RecentFiles;
pub use views::{DetectionView, WelcomeView};
