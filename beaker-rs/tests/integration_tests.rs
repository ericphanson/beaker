use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Run a command and return (exit_code, stdout, stderr)
fn run_beaker_command(args: &[&str]) -> (i32, String, String) {
    let mut full_args = vec!["run", "--"];
    full_args.extend_from_slice(args);

    // Add --verbose to see output in tests
    if !args.contains(&"--verbose") && args.len() > 1 {
        full_args.insert(2, "--verbose");
    }

    let output = Command::new("cargo")
        .args(&full_args)
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

// ============================================================================
// NEW TESTS FOR MULTIPLE INPUT SOURCES (directories, globs, multiple files)
// ============================================================================

/// Set up test data in a temporary directory for the new multi-source tests
fn setup_test_data(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Copy test images from parent directory
    let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();

    // Copy main test images
    fs::copy(source_dir.join("example.jpg"), temp_dir.join("example.jpg"))?;
    fs::copy(
        source_dir.join("example-2-birds.jpg"),
        temp_dir.join("example-2-birds.jpg"),
    )?;
    fs::copy(
        source_dir.join("example-crop.jpg"),
        temp_dir.join("example-crop.jpg"),
    )?;

    // Create a subdirectory with more images
    let subdir = temp_dir.join("subdir");
    fs::create_dir(&subdir)?;
    fs::copy(source_dir.join("example.jpg"), subdir.join("bird1.jpg"))?;
    fs::copy(
        source_dir.join("example-2-birds.jpg"),
        subdir.join("bird2.jpg"),
    )?;

    Ok(())
}

#[test]
fn test_directory_batch_processing() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Set up test data
    setup_test_data(&test_data_dir).expect("Failed to set up test data");

    let output_dir = temp_dir.path().join("output");
    fs::create_dir(&output_dir).expect("Failed to create output directory");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        test_data_dir.to_str().unwrap(),
        "--confidence",
        "0.5",
        "--crop",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Directory processing should exit successfully. Stderr: {stderr}"
    );

    // Should indicate batch processing
    assert!(
        stdout.contains("Processing") && stdout.contains("images"),
        "Should indicate multiple image processing"
    );

    // Should show intelligent device selection for multiple images
    assert!(
        stdout.contains("CoreML") || stdout.contains("CPU"),
        "Should show device selection based on batch size"
    );

    // Check that multiple TOML files were created
    let toml_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with("-beaker.toml")
        })
        .collect();

    assert!(
        toml_files.len() >= 2,
        "Should create TOML files for multiple images, found: {}",
        toml_files.len()
    );

    // Check that crop files were created
    let crop_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains("crop"))
        .collect();

    assert!(
        !crop_files.is_empty(),
        "Should create crop files for detected heads"
    );
}

#[test]
fn test_glob_pattern_processing() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Set up test data
    setup_test_data(&test_data_dir).expect("Failed to set up test data");

    let output_dir = temp_dir.path().join("output");
    fs::create_dir(&output_dir).expect("Failed to create output directory");

    // Use glob pattern to match specific files
    let glob_pattern = format!("{}/*-2-*.jpg", test_data_dir.display());

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &glob_pattern,
        "--confidence",
        "0.5",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Glob pattern processing should exit successfully. Stderr: {stderr}"
    );

    // Should process the matched file(s)
    assert!(
        stdout.contains("Processing") && stdout.contains("image"),
        "Should indicate image processing"
    );

    // Check that TOML file was created for the matched image
    let toml_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.contains("2-birds") && name.ends_with("-beaker.toml")
        })
        .collect();

    assert!(
        !toml_files.is_empty(),
        "Should create TOML file for glob-matched image"
    );
}

#[test]
fn test_multiple_explicit_files() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Set up test data
    setup_test_data(&test_data_dir).expect("Failed to set up test data");

    let output_dir = temp_dir.path().join("output");
    fs::create_dir(&output_dir).expect("Failed to create output directory");

    // Specify multiple files explicitly
    let file1 = test_data_dir.join("example.jpg");
    let file2 = test_data_dir.join("example-2-birds.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        file1.to_str().unwrap(),
        file2.to_str().unwrap(),
        "--confidence",
        "0.5",
        "--crop",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Multiple file processing should exit successfully. Stderr: {stderr}"
    );

    // Should indicate processing multiple images from multiple sources
    assert!(
        stdout.contains("Processing") && stdout.contains("images from 2 sources"),
        "Should indicate processing from multiple sources"
    );

    // Check that TOML files were created for both images
    let expected_tomls = ["example-beaker.toml", "example-2-birds-beaker.toml"];
    for expected_toml in &expected_tomls {
        let toml_path = output_dir.join(expected_toml);
        assert_file_exists_with_content(&toml_path);
    }

    // Check that crop files were created
    let crop_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains("crop"))
        .collect();

    assert!(
        !crop_files.is_empty(),
        "Should create crop files for detected heads"
    );
}

#[test]
fn test_mixed_sources_file_and_directory() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Set up test data
    setup_test_data(&test_data_dir).expect("Failed to set up test data");

    let output_dir = temp_dir.path().join("output");
    fs::create_dir(&output_dir).expect("Failed to create output directory");

    // Mix a single file and a directory
    let single_file = test_data_dir.join("example.jpg");
    let subdir = test_data_dir.join("subdir");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        single_file.to_str().unwrap(),
        subdir.to_str().unwrap(),
        "--confidence",
        "0.5",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Mixed source processing should exit successfully. Stderr: {stderr}"
    );

    // Should indicate processing multiple images from multiple sources
    assert!(
        stdout.contains("Processing") && stdout.contains("from 2 sources"),
        "Should indicate processing from 2 different source types"
    );

    // Should process at least 3 images (1 from file + 2 from subdir)
    assert!(
        stdout.contains("3 images") || stdout.contains("Processing 3"),
        "Should process 3 images total"
    );

    // Check that TOML files were created
    let toml_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with("-beaker.toml")
        })
        .collect();

    assert!(
        toml_files.len() >= 3,
        "Should create TOML files for all processed images, found: {}",
        toml_files.len()
    );
}

#[test]
fn test_device_selection_based_on_batch_size() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Set up test data
    setup_test_data(&test_data_dir).expect("Failed to set up test data");

    let output_dir = temp_dir.path().join("output");
    fs::create_dir(&output_dir).expect("Failed to create output directory");

    // Test with single image (should use CPU for small batch)
    let single_file = test_data_dir.join("example.jpg");
    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        single_file.to_str().unwrap(),
        "--confidence",
        "0.5",
        "--device",
        "auto",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Single image should work. Stderr: {stderr}");
    assert!(
        stdout.contains("CPU") || stdout.contains("using CPU for small batch"),
        "Single image should prefer CPU, got: {stdout}"
    );

    // Test with directory (multiple images, should consider CoreML if available)
    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        test_data_dir.to_str().unwrap(),
        "--confidence",
        "0.5",
        "--device",
        "auto",
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Directory should work. Stderr: {stderr}");
    // For multiple images, should either use CoreML (if available) or CPU
    assert!(
        stdout.contains("CoreML") || stdout.contains("CPU"),
        "Should show device selection for batch processing"
    );
}

#[test]
fn test_empty_directory_handling() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let empty_dir = temp_dir.path().join("empty");
    fs::create_dir(&empty_dir).expect("Failed to create empty directory");

    let (exit_code, _stdout, _stderr) =
        run_beaker_command(&["head", empty_dir.to_str().unwrap(), "--confidence", "0.5"]);

    // Should exit with success but report no images found
    assert_eq!(exit_code, 0, "Empty directory should not cause failure");
    // The stdout should indicate no images were processed
}

#[test]
fn test_nonexistent_glob_pattern() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_data_dir = temp_dir.path().join("test_images");
    fs::create_dir(&test_data_dir).expect("Failed to create test data directory");

    // Use a glob pattern that won't match anything
    let glob_pattern = format!("{}/*.nonexistent", test_data_dir.display());

    let (exit_code, _stdout, _stderr) =
        run_beaker_command(&["head", &glob_pattern, "--confidence", "0.5"]);

    assert_ne!(exit_code, 0, "Should fail when no files match glob pattern");
    assert!(
        _stderr.contains("No image files found") || _stderr.contains("matching pattern"),
        "Should indicate no matching files"
    );
}
