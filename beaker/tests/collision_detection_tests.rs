use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Test collision detection when processing files with same basename to single output directory

#[test]
fn test_collision_detection_errors_without_force() {
    let temp_dir = TempDir::new().unwrap();

    // Create two subdirectories with files having the same basename
    let dir1 = temp_dir.path().join("dir1");
    let dir2 = temp_dir.path().join("dir2");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    let image1 = dir1.join("bird.jpg");
    let image2 = dir2.join("bird.jpg");
    fs::copy("../example.jpg", &image1).unwrap();
    fs::copy("../example.jpg", &image2).unwrap();

    let output_dir = temp_dir.path().join("output");

    // Run detect command with both images to single output directory
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            image1.to_str().unwrap(),
            image2.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--crop",
            "head",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with collision error
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Output path collision detected"),
        "Should detect collision, got: {stderr}"
    );
    assert!(
        stderr.contains("--force"),
        "Should suggest --force flag, got: {stderr}"
    );
}

#[test]
fn test_collision_allowed_with_force() {
    let temp_dir = TempDir::new().unwrap();

    // Create two subdirectories with files having the same basename
    let dir1 = temp_dir.path().join("dir1");
    let dir2 = temp_dir.path().join("dir2");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    let image1 = dir1.join("bird.jpg");
    let image2 = dir2.join("bird.jpg");
    fs::copy("../example.jpg", &image1).unwrap();
    fs::copy("../example.jpg", &image2).unwrap();

    let output_dir = temp_dir.path().join("output");

    // Run detect command with --force flag
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            image1.to_str().unwrap(),
            image2.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--crop",
            "head",
            "--force",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed with --force
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Should succeed with --force, stderr: {stderr}"
    );
}

#[test]
fn test_no_collision_without_output_dir() {
    let temp_dir = TempDir::new().unwrap();

    // Create two subdirectories with files having the same basename
    let dir1 = temp_dir.path().join("dir1");
    let dir2 = temp_dir.path().join("dir2");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    let image1 = dir1.join("bird.jpg");
    let image2 = dir2.join("bird.jpg");
    fs::copy("../example.jpg", &image1).unwrap();
    fs::copy("../example.jpg", &image2).unwrap();

    // Run detect command WITHOUT --output-dir
    // Files should go to their respective source directories, no collision
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            image1.to_str().unwrap(),
            image2.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--crop",
            "head",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed without collision check
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Should succeed without output-dir, stderr: {stderr}"
    );

    // Verify both outputs exist in their respective directories
    assert!(
        dir1.join("bird_crop_head.jpg").exists(),
        "Output should exist in dir1"
    );
    assert!(
        dir2.join("bird_crop_head.jpg").exists(),
        "Output should exist in dir2"
    );
}

#[test]
fn test_no_collision_with_different_basenames() {
    let temp_dir = TempDir::new().unwrap();

    // Create directory with files having different basenames
    let dir = temp_dir.path().join("images");
    fs::create_dir_all(&dir).unwrap();

    let image1 = dir.join("bird1.jpg");
    let image2 = dir.join("bird2.jpg");
    fs::copy("../example.jpg", &image1).unwrap();
    fs::copy("../example.jpg", &image2).unwrap();

    let output_dir = temp_dir.path().join("output");

    // Run detect command with different basenames
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            image1.to_str().unwrap(),
            image2.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--crop",
            "head",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed since basenames are different
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Should succeed with different basenames, stderr: {stderr}"
    );
}

#[test]
fn test_collision_error_message_quality() {
    let temp_dir = TempDir::new().unwrap();

    // Create two subdirectories with files having the same basename
    let dir1 = temp_dir.path().join("photos/vacation");
    let dir2 = temp_dir.path().join("photos/work");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    let image1 = dir1.join("bird.jpg");
    let image2 = dir2.join("bird.jpg");
    fs::copy("../example.jpg", &image1).unwrap();
    fs::copy("../example.jpg", &image2).unwrap();

    let output_dir = temp_dir.path().join("output");

    // Run detect command
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            image1.to_str().unwrap(),
            image2.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--crop",
            "head",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with collision error
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check error message contains helpful information
    assert!(
        stderr.contains("Output path collision detected"),
        "Should have clear error title"
    );
    assert!(
        stderr.contains("vacation") || stderr.contains("work"),
        "Should show paths to both files"
    );
    assert!(
        stderr.contains("bird_crop_head.jpg") || stderr.contains("bird"),
        "Should show the conflicting output filename"
    );
    assert!(stderr.contains("Solutions:"), "Should provide solutions");
    assert!(
        stderr.contains("--force"),
        "Should suggest --force as a solution"
    );
    assert!(
        stderr.contains("Process files separately") || stderr.contains("Rename input files"),
        "Should suggest alternative solutions"
    );
}
