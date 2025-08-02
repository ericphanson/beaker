use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Run a command and return (exit_code, stdout, stderr)
fn run_beaker_command(args: &[&str]) -> (i32, String, String) {
    let output = Command::new("cargo")
        .args(["run", "--"])
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (exit_code, stdout, stderr)
}

/// Get the path to test images relative to the beaker-rs directory
fn test_image_path(filename: &str) -> String {
    format!("../{filename}")
}

/// Check if a file exists and has content
fn assert_file_exists_with_content(path: &Path) {
    assert!(path.exists(), "File should exist: {}", path.display());
    let metadata = fs::metadata(path).expect("Failed to get file metadata");
    assert!(
        metadata.len() > 0,
        "File should not be empty: {}",
        path.display()
    );
}

#[test]
fn test_help_command() {
    let (exit_code, stdout, _stderr) = run_beaker_command(&["--help"]);

    assert_eq!(exit_code, 0, "Help command should exit successfully");
    assert!(
        stdout.contains("Bird detection and analysis toolkit"),
        "Help should contain description"
    );
    assert!(
        stdout.contains("head"),
        "Help should mention head subcommand"
    );
}

#[test]
fn test_basic_head_detection_single_bird() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(stdout.contains("Found"), "Should report found detections");
    assert!(stdout.contains("detections"), "Should mention detections");

    // Check that TOML output was created
    let toml_path = temp_dir.path().join("example-beaker.toml");
    assert_file_exists_with_content(&toml_path);

    // Verify TOML content structure
    let toml_content = fs::read_to_string(&toml_path).expect("Failed to read TOML file");
    assert!(
        toml_content.contains("[head]"),
        "TOML should contain head section"
    );
    assert!(
        toml_content.contains("confidence_threshold"),
        "TOML should contain confidence threshold"
    );
}

#[test]
fn test_basic_head_detection_two_birds() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example-2-birds.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(stdout.contains("Found"), "Should report found detections");

    // Check that TOML output was created
    let toml_path = temp_dir.path().join("example-2-birds-beaker.toml");
    assert_file_exists_with_content(&toml_path);
}

#[test]
fn test_head_detection_with_crops() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--crop",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        stdout.contains("Will create head crops"),
        "Should indicate crop creation"
    );
    assert!(stdout.contains("Found"), "Should report found detections");

    // Check for TOML output
    let toml_path = temp_dir.path().join("example-beaker.toml");
    assert_file_exists_with_content(&toml_path);

    // Check for crop images (pattern: example-crop-1.jpg, example-crop-2.jpg, etc.)
    let crop_files: Vec<_> = fs::read_dir(temp_dir.path())
        .expect("Failed to read temp directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_name().to_string_lossy().contains("crop")
                && entry.file_name().to_string_lossy().ends_with(".jpg")
        })
        .collect();

    assert!(
        !crop_files.is_empty(),
        "Should create at least one crop file"
    );

    // Verify crop files have content
    for crop_file in crop_files {
        assert_file_exists_with_content(&crop_file.path());
    }
}

#[test]
fn test_head_detection_with_bounding_boxes() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--bounding-box",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        stdout.contains("Will save bounding box images"),
        "Should indicate bounding box creation"
    );
    assert!(stdout.contains("Found"), "Should report found detections");

    // Check for TOML output
    let toml_path = temp_dir.path().join("example-beaker.toml");
    assert_file_exists_with_content(&toml_path);

    // Check for bounding box image
    let bbox_path = temp_dir.path().join("example-bounding-box.jpg");
    assert_file_exists_with_content(&bbox_path);
}

#[test]
fn test_head_detection_with_all_outputs() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example-2-birds.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.3",
        "--crop",
        "--bounding-box",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        stdout.contains("Will create head crops"),
        "Should indicate crop creation"
    );
    assert!(
        stdout.contains("Will save bounding box images"),
        "Should indicate bounding box creation"
    );
    assert!(stdout.contains("Found"), "Should report found detections");

    // Check for TOML output
    let toml_path = temp_dir.path().join("example-2-birds-beaker.toml");
    assert_file_exists_with_content(&toml_path);

    // Check for bounding box image
    let bbox_path = temp_dir.path().join("example-2-birds-bounding-box.jpg");
    assert_file_exists_with_content(&bbox_path);

    // Check for crop images
    let crop_files: Vec<_> = fs::read_dir(temp_dir.path())
        .expect("Failed to read temp directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_name().to_string_lossy().contains("crop")
                && entry.file_name().to_string_lossy().ends_with(".jpg")
        })
        .collect();

    assert!(
        !crop_files.is_empty(),
        "Should create at least one crop file"
    );
}

#[test]
fn test_no_metadata_option() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--no-metadata",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        !stdout.contains("Will create metadata output"),
        "Should not mention metadata creation"
    );
    assert!(stdout.contains("Found"), "Should report found detections");

    // Check that metadata output was NOT created
    let toml_path = temp_dir.path().join("example-beaker.toml");
    assert!(
        !toml_path.exists(),
        "Metadata file should not exist when --no-metadata is used"
    );
}

#[test]
fn test_different_confidence_thresholds() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    // Test with high confidence (should find fewer detections)
    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.9",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        stdout.contains("Confidence threshold: 0.9"),
        "Should show correct confidence threshold"
    );
    assert!(stdout.contains("Found"), "Should report found detections");
}

#[test]
fn test_different_devices() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    // Test CPU device
    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--device",
        "cpu",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "CPU device should work. Stderr: {stderr}");
    assert!(stdout.contains("Device: cpu"), "Should show CPU device");
    assert!(stdout.contains("Found"), "Should report found detections");

    // Test auto device (default)
    let temp_dir2 = TempDir::new().expect("Failed to create temp directory");
    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--device",
        "auto",
        "--output-dir",
        temp_dir2.path().to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Auto device should work. Stderr: {stderr}");
    assert!(stdout.contains("Device: auto"), "Should show auto device");
    assert!(stdout.contains("Found"), "Should report found detections");
}

#[test]
fn test_nonexistent_image() {
    let (exit_code, _stdout, stderr) =
        run_beaker_command(&["head", "nonexistent-image.jpg", "--confidence", "0.5"]);

    assert_ne!(exit_code, 0, "Command should fail for nonexistent image");
    assert!(!stderr.is_empty(), "Should produce error message");
}

#[test]
fn test_invalid_confidence() {
    let image_path = test_image_path("example.jpg");

    // Test confidence > 1.0
    let (exit_code, _stdout, stderr) =
        run_beaker_command(&["head", &image_path, "--confidence", "1.5"]);

    // Should either fail with exit code or handle gracefully
    if exit_code == 0 {
        // If it doesn't fail, it should at least clamp the value
        println!("Note: Confidence 1.5 was accepted, might be clamped internally");
    } else {
        assert!(
            !stderr.is_empty(),
            "Should produce error message for invalid confidence"
        );
    }
}

#[test]
fn test_custom_iou_threshold() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--confidence",
        "0.5",
        "--iou-threshold",
        "0.3",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Command should exit successfully. Stderr: {stderr}"
    );
    assert!(
        stdout.contains("IoU threshold: 0.3"),
        "Should show correct IoU threshold"
    );
    assert!(stdout.contains("Found"), "Should report found detections");
}

#[test]
fn test_output_file_naming() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Test that different input files create different output files
    let tests = vec![
        ("example.jpg", "example-beaker.toml"),
        ("example-2-birds.jpg", "example-2-birds-beaker.toml"),
    ];

    for (input_file, expected_toml) in tests {
        let image_path = test_image_path(input_file);

        let (exit_code, stdout, stderr) = run_beaker_command(&[
            "head",
            &image_path,
            "--confidence",
            "0.5",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);

        assert_eq!(
            exit_code, 0,
            "Command should exit successfully for {input_file}. Stderr: {stderr}"
        );
        assert!(
            stdout.contains("Found"),
            "Should report found detections for {input_file}"
        );

        // Check that the correct TOML file was created
        let toml_path = temp_dir.path().join(expected_toml);
        assert_file_exists_with_content(&toml_path);
    }
}
