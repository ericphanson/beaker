use std::process::Command;
use tempfile::TempDir;

/// Test that quality debug images directory is only created when --debug-dump-images flag is passed
#[test]
fn test_quality_debug_images_not_created_without_flag() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Run quality command without --debug-dump-images flag
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "quality",
            test_image.to_str().unwrap(),
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
            "--metadata",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed
    assert!(
        output.status.success(),
        "Quality command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that NO debug directory was created
    let debug_dir_pattern = temp_dir.path().join("quality_debug_images_test");
    assert!(
        !debug_dir_pattern.exists(),
        "Debug directory should not exist without --debug-dump-images flag: {:?}",
        debug_dir_pattern
    );
}

/// Test that quality debug images directory IS created when --debug-dump-images flag is passed
#[test]
fn test_quality_debug_images_created_with_flag() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Run quality command with --debug-dump-images flag
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "quality",
            test_image.to_str().unwrap(),
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
            "--metadata",
            "--debug-dump-images",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed
    assert!(
        output.status.success(),
        "Quality command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that debug directory WAS created
    let debug_dir_pattern = temp_dir.path().join("quality_debug_images_test");
    assert!(
        debug_dir_pattern.exists(),
        "Debug directory should exist with --debug-dump-images flag: {:?}",
        debug_dir_pattern
    );

    // Verify that debug images were actually saved
    let expected_files = [
        "t224_heat.png",
        "p224_heat.png",
        "pfused_heat.png",
        "weights_heat.png",
        "image.png",
    ];

    for file in &expected_files {
        let file_path = debug_dir_pattern.join(file);
        assert!(
            file_path.exists(),
            "Expected debug file should exist: {:?}",
            file_path
        );
    }
}
