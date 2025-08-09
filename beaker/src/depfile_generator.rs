//! Make-compatible dependency file generation
//!
//! This module generates dependency files (.d files) that can be included
//! in Makefiles to enable accurate incremental builds. The depfiles list
//! all inputs that affect the byte-level output as prerequisites.
//!
//! ## New Architecture (Recommended)
//!
//! The new approach uses OutputManager to track actual outputs produced:
//! - `generate_depfile_from_output_manager()` uses OutputManager's tracked outputs
//! - This eliminates synchronization issues between depfile generation and actual outputs
//! - Single source of truth for what files are actually produced
//!
//! ## Legacy Functions (Deprecated)
//!
//! The functions `get_detection_output_files()` and `get_cutout_output_files()`
//! manually duplicate OutputManager logic and should not be used for new code.
//! They are kept for backwards compatibility and testing only.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

use crate::output_manager::OutputManager;
use crate::stamp_manager::StampInfo;

/// Generate a Make-compatible dependency file using OutputManager's tracked outputs
pub fn generate_depfile_from_output_manager(
    depfile_path: &Path,
    output_manager: &OutputManager,
    input_files: &[PathBuf],
    stamp_info: &StampInfo,
) -> Result<()> {
    let tracked_outputs = output_manager.get_produced_outputs();
    generate_depfile(depfile_path, &tracked_outputs, input_files, stamp_info)
}

/// Generate a Make-compatible dependency file
pub fn generate_depfile(
    depfile_path: &Path,
    targets: &[PathBuf],
    input_files: &[PathBuf],
    stamp_info: &StampInfo,
) -> Result<()> {
    // Collect all prerequisites: input files + stamp files
    let mut prerequisites = Vec::new();

    // Add input files
    for input_file in input_files {
        prerequisites.push(input_file.clone());
    }

    // Add stamp files
    for stamp_path in stamp_info.all_stamps() {
        prerequisites.push(stamp_path.to_path_buf());
    }

    // Generate depfile content
    let content = generate_depfile_content(targets, &prerequisites)?;

    // Write depfile atomically
    write_depfile_atomically(depfile_path, &content)?;

    Ok(())
}

/// Generate the content string for a depfile
fn generate_depfile_content(targets: &[PathBuf], prerequisites: &[PathBuf]) -> Result<String> {
    if targets.is_empty() {
        return Ok(String::new());
    }

    // Format targets
    let target_strs: Vec<String> = targets.iter().map(|p| escape_path_for_make(p)).collect();
    let targets_line = target_strs.join(" ");

    // Format prerequisites
    let prereq_strs: Vec<String> = prerequisites
        .iter()
        .map(|p| escape_path_for_make(p))
        .collect();
    let prereqs_line = prereq_strs.join(" ");

    // Generate Make rule
    let content = if prereqs_line.is_empty() {
        format!("{targets_line}:\n")
    } else {
        format!("{targets_line}: {prereqs_line}\n")
    };

    Ok(content)
}

/// Escape a path for use in Makefiles
///
/// Make requires spaces and some special characters to be escaped
fn escape_path_for_make(path: &Path) -> String {
    let path_str = path.to_string_lossy();

    // Escape spaces, dollar signs, and backslashes
    path_str
        .replace('\\', "\\\\") // Escape backslashes first
        .replace(' ', "\\ ") // Escape spaces
        .replace('$', "$$") // Escape dollar signs
}

/// Write depfile content atomically using temp file + rename
fn write_depfile_atomically(depfile_path: &Path, content: &String) -> Result<()> {
    // Create parent directory if needed
    if let Some(parent) = depfile_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write to temporary file first
    let temp_path = depfile_path.with_extension("tmp");
    fs::write(&temp_path, content)?;

    // Atomic rename
    fs::rename(temp_path, depfile_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_path_escaping() {
        // Test normal path
        let normal_path = Path::new("/path/to/file.txt");
        assert_eq!(escape_path_for_make(normal_path), "/path/to/file.txt");

        // Test path with spaces
        let space_path = Path::new("/path/with spaces/file.txt");
        assert_eq!(
            escape_path_for_make(space_path),
            "/path/with\\ spaces/file.txt"
        );

        // Test path with dollar signs
        let dollar_path = Path::new("/path/$VAR/file.txt");
        assert_eq!(escape_path_for_make(dollar_path), "/path/$$VAR/file.txt");
    }

    #[test]
    fn test_depfile_content_generation() {
        let targets = vec![PathBuf::from("output.png"), PathBuf::from("output.toml")];
        let prerequisites = vec![
            PathBuf::from("input.jpg"),
            PathBuf::from("/cache/stamp1.stamp"),
        ];

        let content = generate_depfile_content(&targets, &prerequisites).unwrap();
        let expected = "output.png output.toml: input.jpg /cache/stamp1.stamp\n";
        assert_eq!(content, expected);
    }

    #[test]
    fn test_depfile_content_with_spaces() {
        let targets = vec![PathBuf::from("output with spaces.png")];
        let prerequisites = vec![PathBuf::from("input with spaces.jpg")];

        let content = generate_depfile_content(&targets, &prerequisites).unwrap();
        let expected = "output\\ with\\ spaces.png: input\\ with\\ spaces.jpg\n";
        assert_eq!(content, expected);
    }

    #[test]
    fn test_atomic_depfile_write() {
        let temp_dir = tempdir().unwrap();
        let depfile_path = temp_dir.path().join("test.d");
        let content = "target: prereq1 prereq2\n".to_string();

        write_depfile_atomically(&depfile_path, &content).unwrap();

        assert!(depfile_path.exists());
        assert_eq!(fs::read_to_string(&depfile_path).unwrap(), content);
    }
}
