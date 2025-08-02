# Bird Head Detector Tests

This directory contains comprehensive end-to-end tests for the bird-head-detector tool.

## What the tests cover

The test suite validates the complete workflow:

1. **Tool Installation**: Builds and installs the `bird-head-detector` tool using `uv`
2. **Default Behavior**: Tests that cropping is enabled by default
3. **Command Options**: Tests all command-line options and their combinations:
   - `--skip-crop`: Disables default cropping behavior
   - `--save-bounding-box`: Creates visualization with detection boxes
   - `--output-dir`: Saves outputs to specified directory
   - `--padding`: Configurable bounding box expansion
   - `--conf`: Confidence threshold for detections
4. **File Format Handling**: Tests PNG/JPG format preservation
5. **Directory Processing**: Tests batch processing of multiple images
6. **Output Naming**: Validates correct output file naming conventions

## Running the tests

### Quick run
```bash
uv run python test/run_tests.py
```

### Direct execution
```bash
uv run python test/test_e2e.py
```

### Using unittest
```bash
uv run python -m unittest test.test_e2e -v
```

## Test data

The tests use the `example.jpg` image from the repository root, creating multiple copies with different names and formats to test various scenarios:

- Single image processing (JPG and PNG)
- Batch directory processing
- Output directory functionality
- Format preservation

## Test environment

The tests:
- Create a temporary directory for all test operations
- Build and install the tool fresh for each test run
- Clean up all temporary files and uninstall the tool when complete
- Use actual `uv tool install` to test the real installation process

## Expected behavior

All tests should pass if:
- The tool builds and installs correctly
- Default cropping behavior works
- All command-line options function as expected
- File outputs are created with correct names and formats
- Directory processing handles multiple images correctly

The test suite captures the end-to-end functionality that was manually verified during development.
