//! Color and styling utilities with support for NO_COLOR and TERM environment variables.
//!
//! This module provides utilities for conditional colored output based on:
//! - `--no-color` CLI flag
//! - `NO_COLOR` environment variable (https://no-color.org/)
//! - `BEAKER_NO_COLOR` environment variable (application-specific)
//! - `TERM=dumb` environment variable
//! - TTY detection for stderr

use colored::ColoredString;
use std::io::{stderr, IsTerminal};
use std::sync::OnceLock;

/// Global color configuration state
static COLOR_CONFIG: OnceLock<ColorConfig> = OnceLock::new();

/// Check environment variables and TTY state for color support
fn should_disable_colors_from_env() -> bool {
    // Check NO_COLOR standard (https://no-color.org/)
    !std::env::var("NO_COLOR").unwrap_or_default().is_empty()
        // Check application-specific override
        || !std::env::var("BEAKER_NO_COLOR").unwrap_or_default().is_empty()
        // Check for dumb terminal
        || std::env::var("TERM").unwrap_or_default() == "dumb"
        // Check if stderr is not a TTY (log messages go to stderr)
        || !stderr().is_terminal()
}

#[derive(Debug, Clone)]
struct ColorConfig {
    colors_enabled: bool,
}

impl ColorConfig {
    fn new(no_color_flag: bool) -> Self {
        let colors_enabled = !no_color_flag && !should_disable_colors_from_env();
        Self { colors_enabled }
    }

    fn is_enabled(&self) -> bool {
        self.colors_enabled
    }
}

/// Initialize the color configuration with the CLI flag state.
/// This should be called once at application startup after parsing CLI arguments.
pub fn init_color_config(no_color_flag: bool) {
    let config = ColorConfig::new(no_color_flag);
    COLOR_CONFIG.set(config).unwrap_or_else(|_| {
        eprintln!("Warning: Color configuration already initialized");
    });
}

/// Check if colors are enabled based on configuration
fn colors_enabled() -> bool {
    COLOR_CONFIG
        .get()
        .map(|config| config.is_enabled())
        .unwrap_or_else(|| {
            // Fallback if not initialized - check env vars and TTY only
            !should_disable_colors_from_env()
        })
}

/// Apply color to a string only if colors are enabled for stderr output
pub fn maybe_color_stderr<F>(text: &str, color_fn: F) -> String
where
    F: FnOnce(&str) -> ColoredString,
{
    if colors_enabled() {
        color_fn(text).to_string()
    } else {
        text.to_string()
    }
}

pub fn maybe_dim_stderr(text: &str) -> String {
    use colored::Colorize;
    maybe_color_stderr(text, |s| s.bright_black())
}

/// Semantic color functions for different message types
pub mod colors {
    use super::maybe_color_stderr;
    use colored::Colorize;

    /// Color for error-level messages (critical failures)
    pub fn error_level(text: &str) -> String {
        maybe_color_stderr(text, |s| s.red().bold())
    }

    /// Color for warning-level messages
    pub fn warning_level(text: &str) -> String {
        maybe_color_stderr(text, |s| s.yellow())
    }

    /// Color for info-level messages (general information)
    pub fn info_level(text: &str) -> String {
        maybe_color_stderr(text, |s| s.green())
    }

    /// Color for debug-level messages
    pub fn debug_level(text: &str) -> String {
        maybe_color_stderr(text, |s| s.blue())
    }

    /// Color for trace-level messages (detailed tracing)
    pub fn trace_level(text: &str) -> String {
        maybe_color_stderr(text, |s| s.magenta())
    }
}
/// Semantic symbols for different operation types and states
pub mod symbols {
    use super::colors_enabled;

    pub fn model_loaded() -> &'static str {
        if colors_enabled() {
            "✅"
        } else {
            "  "
        }
    }
    /// Symbol for starting a head detection operation
    pub fn head_detection_start() -> &'static str {
        if colors_enabled() {
            "🔍"
        } else {
            ""
        }
    }

    /// Symbol for starting a background removal operation
    pub fn background_removal_start() -> &'static str {
        if colors_enabled() {
            "✂️ "
        } else {
            "[CUTOUT]"
        }
    }

    /// Symbol for operation failures
    pub fn operation_failed() -> &'static str {
        if colors_enabled() {
            "❌"
        } else {
            "[FAILED]"
        }
    }

    /// Symbol for technical setup and configuration
    pub fn system_setup() -> &'static str {
        if colors_enabled() {
            "⚙️ "
        } else {
            ""
        }
    }

    /// Symbol for finding/targeting resources
    pub fn resources_found() -> &'static str {
        if colors_enabled() {
            "🎯"
        } else {
            ""
        }
    }

    /// Symbol for search/checking operations
    pub fn checking() -> &'static str {
        if colors_enabled() {
            "🔍"
        } else {
            ""
        }
    }

    /// Symbol for successful completion
    pub fn completed_successfully() -> &'static str {
        if colors_enabled() {
            "✅"
        } else {
            "[SUCCESS]"
        }
    }

    /// Symbol for partial success (some successes, some failures)
    pub fn completed_partially_successfully() -> &'static str {
        if colors_enabled() {
            "⚠️ "
        } else {
            "[PARTIAL-SUCCESS]"
        }
    }

    /// Symbol for warnings
    pub fn warning() -> &'static str {
        if colors_enabled() {
            "⚠️ "
        } else {
            ""
        }
    }
}

/// Progress bar utilities that respect TTY state
pub mod progress {
    use crate::progress::add_progress_bar;

    use super::colors_enabled;
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::{stderr, IsTerminal};

    /// Create a progress bar for batch processing, only if stderr is interactive
    pub fn create_batch_progress_bar(total: usize) -> Option<ProgressBar> {
        // Only show progress bar if:
        // 1. Processing more than 1 item
        // 2. stderr is a TTY (interactive)
        // 3. Colors are enabled (respects all our color settings)
        if total > 1 && stderr().is_terminal() {
            let pb = ProgressBar::new(total as u64);
            add_progress_bar(pb.clone());
            let style = if colors_enabled() {
                ProgressStyle::default_bar()
                    .template(
                        "[{elapsed_precise}] [{bar:30.green/black}] ({percent}%) {msg}\n{prefix}",
                    )
                    .unwrap()
                    .progress_chars("█▓▒░")
            } else {
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:30}] ({percent}%) {msg}\n{prefix}")
                    .unwrap()
                    .progress_chars("#> ")
            };

            pb.set_style(style);
            pb.enable_steady_tick(std::time::Duration::from_millis(100));

            Some(pb)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_config_respects_no_color_flag() {
        // Test with no_color_flag = true - should always disable colors
        let config = ColorConfig::new(true);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_color_config_respects_no_color_env() {
        std::env::set_var("NO_COLOR", "1");
        let config = ColorConfig::new(false);
        assert!(!config.is_enabled());
        std::env::remove_var("NO_COLOR");
    }

    #[test]
    fn test_color_config_respects_term_dumb() {
        std::env::set_var("TERM", "dumb");
        let config = ColorConfig::new(false);
        assert!(!config.is_enabled());
        std::env::remove_var("TERM");
    }

    #[test]
    fn test_color_config_respects_beaker_no_color() {
        std::env::set_var("BEAKER_NO_COLOR", "1");
        let config = ColorConfig::new(false);
        assert!(!config.is_enabled());
        std::env::remove_var("BEAKER_NO_COLOR");
    }

    #[test]
    fn test_maybe_color_with_colors_disabled() {
        use colored::Colorize;

        // Simulate colors disabled
        COLOR_CONFIG
            .set(ColorConfig {
                colors_enabled: false,
            })
            .ok();

        let result = maybe_color_stderr("test", |s| s.red());
        assert_eq!(result, "test");
    }
}
