use std::process::Command;
use tempfile::TempDir;

/// Test CLI model functionality using subprocess execution
/// This tests the new --model-path, --model-url, and --model-checksum arguments

#[test]
fn test_cli_model_path_validation() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test with non-existent model path
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--model-path",
            "/non/existent/model.onnx",
            "--metadata",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with error about non-existent path
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist") || stderr.contains("No such file"),
        "Error should mention non-existent path, got: {stderr}"
    );
}

#[test]
fn test_cli_model_validation_conflicts() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test with both --model-path and --model-url (should fail)
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--model-path",
            "/some/path.onnx",
            "--model-url",
            "https://example.com/model.onnx",
            "--metadata",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with validation error
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Cannot specify both") || stderr.contains("--model-path and --model-url"),
        "Error should mention conflict, got: {stderr}"
    );
}

#[test]
fn test_cli_model_checksum_without_url() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test with --model-checksum but no --model-url (should fail)
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--model-checksum",
            "abcd1234",
            "--metadata",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with validation error
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("can only be used with --model-url"),
        "Error should mention checksum requires URL, got: {stderr}"
    );
}

#[test]
fn test_cli_empty_model_file_rejection() {
    let temp_dir = TempDir::new().unwrap();

    // Create an empty "model" file
    let empty_model = temp_dir.path().join("empty_model.onnx");
    std::fs::write(&empty_model, b"").unwrap(); // 0 bytes

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test with empty model file (should fail)
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--model-path",
            empty_model.to_str().unwrap(),
            "--metadata",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with error about empty file
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty") || stderr.contains("0 bytes"),
        "Error should mention empty file, got: {stderr}"
    );
}

#[test]
fn test_cli_model_url_without_checksum() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test with valid model URL from the issue (without checksum)
    // This should proceed but show a warning about skipped verification
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "detect",
            test_image.to_str().unwrap(),
            "--model-url",
            "https://github.com/ericphanson/beaker/releases/download/bird-head-detector-v0.1.1/best.onnx",
            "--metadata",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Print output for debugging
    println!("STDOUT: {stdout}");
    println!("STDERR: {stderr}");

    // Should succeed (or fail gracefully with network/model issues)
    // We expect a warning about checksum verification being skipped
    if !output.status.success() {
        // If it fails, it should be due to network issues or model incompatibility, not CLI validation
        assert!(
            stderr.contains("download")
                || stderr.contains("network")
                || stderr.contains("functionality test")
                || stderr.contains("Model failed")
                || stderr.contains("HTTP")
                || stderr.contains("Connection"),
            "Should fail due to network/model issues, not CLI validation. Got: {stderr}"
        );
    } else {
        // If it succeeds, we should see the warning about checksum verification
        assert!(
            stderr.contains("checksum") && stderr.contains("skipped"),
            "Should show checksum verification warning. Got: {stderr}"
        );
    }
}

#[test]
fn test_cutout_cli_model_options() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test image
    let test_image = temp_dir.path().join("test.jpg");
    std::fs::copy("../example.jpg", &test_image).unwrap();

    // Test that cutout command also has the CLI model options
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "cutout",
            test_image.to_str().unwrap(),
            "--model-path",
            "/non/existent/cutout.onnx",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    // Should fail with error about non-existent path
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist") || stderr.contains("No such file"),
        "Cutout should also support CLI model options, got: {stderr}"
    );
}
