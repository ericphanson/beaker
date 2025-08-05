//! Global progress bar management
//!
//! This module provides a global `MultiProgress` instance that can be shared
//! across the entire application without needing to thread it through function
//! parameters.

use indicatif::MultiProgress;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Global multi-progress bar instance
static MULTI: Lazy<Arc<MultiProgress>> = Lazy::new(|| Arc::new(MultiProgress::new()));

/// Get access to the global multi-progress bar.
///
/// This function returns a cheap clone (just another `Arc` pointer) of the
/// global `MultiProgress` instance, allowing multiple parts of the application
/// to add progress bars without needing to pass the instance around.
pub fn global_mp() -> Arc<MultiProgress> {
    MULTI.clone()
}

// Add progress bar to the global multi-progress instance
pub fn add_progress_bar(pb: indicatif::ProgressBar) {
    global_mp().add(pb);
}

// Remove a progress bar from the global multi-progress instance
pub fn remove_progress_bar(pb: &indicatif::ProgressBar) {
    global_mp().remove(pb);
}
