use std::process::Command;
use tempfile::TempDir;

/// Test environment variable functionality using subprocess execution
/// This avoids race conditions by setting environment variables only on subprocesses

#[test]
fn test_detect_model_path_override() {
    let temp_dir = TempDir::new().unwrap();

    // Create a fake model file
    let temp_model = temp_dir.path().join("fake_model.onnx");
    std::fs::write(&temp_model, b"fake model data").unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Run beaker detect with BEAKER_DETECT_MODEL_PATH set to non-existent file
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--metadata",
        ])
        .env("BEAKER_DETECT_MODEL_PATH", "/non/existent/path.onnx")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with error about non-existent path
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist"),
        "Error should mention non-existent path, got: {stderr}"
    );
}

#[test]
fn test_cutout_model_path_override() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Run beaker cutout with BEAKER_CUTOUT_MODEL_PATH set to non-existent file
    let output = Command::new("cargo")
        .args(["run", "--", "cutout", test_image.to_str().unwrap()])
        .env(
            "BEAKER_CUTOUT_MODEL_PATH",
            "/non/existent/cutout_model.onnx",
        )
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with error about non-existent path
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist"),
        "Error should mention non-existent path, got: {stderr}"
    );
}

#[test]
fn test_version_command_with_env_vars() {
    // Run beaker version with environment variables set
    let output = Command::new("cargo")
        .args(["run", "--", "version"])
        .env("BEAKER_DEBUG", "true")
        .env("NO_COLOR", "1")
        .env("RUST_LOG", "info")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed
    assert!(output.status.success(), "Version command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain version info
    assert!(stdout.contains("beaker v"), "Should contain version info");

    // Should contain environment variables section
    assert!(
        stdout.contains("Environment Variables:"),
        "Should show environment variables"
    );
    assert!(
        stdout.contains("BEAKER_DEBUG: true"),
        "Should show BEAKER_DEBUG"
    );
    assert!(stdout.contains("NO_COLOR: 1"), "Should show NO_COLOR");
    assert!(stdout.contains("RUST_LOG: info"), "Should show RUST_LOG");
}

#[test]
fn test_metadata_captures_env_vars() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Run beaker detect with environment variables and metadata
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--confidence",
            "0.5",
            "--metadata",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ])
        .env("BEAKER_DEBUG", "test_value")
        .env("RUST_LOG", "debug")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should succeed
    assert!(output.status.success(), "Detect command should succeed");

    // Check metadata file was created
    let metadata_path = temp_dir.path().join("test.beaker.toml");
    assert!(metadata_path.exists(), "Metadata file should exist");

    // Read and verify metadata contains environment variables
    let metadata_content = std::fs::read_to_string(&metadata_path).unwrap();
    assert!(
        metadata_content.contains("[detect.execution.beaker_env_vars]"),
        "Should contain environment variables section"
    );
    assert!(
        metadata_content.contains("BEAKER_DEBUG = \"test_value\""),
        "Should contain BEAKER_DEBUG variable"
    );
    assert!(
        metadata_content.contains("RUST_LOG = \"debug\""),
        "Should contain RUST_LOG variable"
    );
}
