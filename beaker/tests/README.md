# Beaker Test Suite

End-to-end integration tests for the beaker CLI tool.

## Running Tests

```bash
# Run all tests
cargo test --test integration_tests

# Run with output visible
cargo test --test integration_tests -- --nocapture

# Run specific test
cargo test --test integration_tests test_basic_head_detection_single_bird

# Run in release mode (faster)
cargo test --test integration_tests --release
```

## Test Images

Tests use `example.jpg` and `example-2-birds.jpg` from the repository root.

## Adding Tests

Add new test functions to `tests/integration_tests.rs`:

```rust
#[test]
fn test_new_functionality() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let image_path = test_image_path("example.jpg");

    let (exit_code, stdout, stderr) = run_beaker_command(&[
        "head",
        &image_path,
        "--new-option",
        "--output-dir", temp_dir.path().to_str().unwrap()
    ]);

    assert_eq!(exit_code, 0, "Command should succeed. Stderr: {}", stderr);
    // Add assertions...
}
```
