# Make Integration and Dependency Tracking

This document describes how Beaker integrates with Make for incremental builds through dependency files (depfiles) and stamp files.

## Overview

Beaker generates Make-compatible dependency files that enable precise incremental builds. The system tracks changes to inputs, model files, tool versions, and configuration parameters that affect the byte-level output of processing operations.

### Core Components

- **Dependency files (.d)**: Make-compatible files listing targets and their prerequisites
- **Stamp files**: Cache files containing hashes of configuration, tool version, and models
- **Output tracking**: Single-source-of-truth tracking via OutputManager

## Dependency File Format

Dependency files follow standard Make syntax:

```make
target1 target2: prereq1 prereq2 prereq3
```

Example for detection with cropping:
```make
example_crop.jpg: example.jpg ~/.cache/beaker/stamps/cfg-detect-f5fc808f3ba2145d.stamp ~/.cache/beaker/stamps/tool-7e3a5fb4e114981e.stamp ~/.cache/beaker/stamps/model-detect-fc16ebd8b0c10d97.stamp
```

### Escaping Rules

Paths are escaped for Make compatibility:
- Spaces become `\ `
- Dollar signs become `$$`
- Backslashes become `\\`

## Stamp Files

Stamp files provide fine-grained change detection for different input categories:

### Configuration Stamps (`cfg-<tool>-<hash>.stamp`)

Track parameters that affect byte-level output. Generated using the `#[stamp]` attribute on config fields:

- Detection: confidence, IoU threshold, crop classes, bounding box flag, model overrides, output directory
- Cutout: post-processing, alpha matting parameters, background color, mask saving, model overrides, output directory

### Tool Stamps (`tool-<hash>.stamp`)

Track the Beaker version to ensure rebuilds when the tool itself changes.

### Model Stamps (`model-<tool>-<hash>.stamp`)

Track model file contents using MD5 hashes. Only generated when custom model paths are provided.

## What Beaker Tracks

### Essential Dependencies

1. **Source images**: Primary input files
2. **Model files**: When custom model paths are specified
3. **Tool version**: Beaker executable version
4. **Output-affecting configuration**: Parameters marked with `#[stamp]`

### What is NOT Tracked

1. **Metadata files**: TOML/JSON files are not prerequisites (per design invariant)
2. **Performance settings**: Device selection, verbosity levels
3. **Output directory**: Tracked only when it affects output file naming
4. **Timestamps or cache status**: Only content changes matter

## Design Invariants

1. **Real byte-affecting inputs only**: Depfiles list only inputs that change actual output bytes
2. **Shared metadata never a prerequisite**: Later stages may update shared TOML without affecting earlier stages
3. **Idempotent writes**: Unchanged content preserves mtime for stable incremental behavior
4. **Atomic operations**: Temp file + rename pattern prevents partial writes
5. **One depfile per target**: Each processing operation generates its own dependency file

## Adding New Configuration Options

When adding new configuration fields:

1. **Determine impact**: Does this field affect byte-level output?
2. **Add stamp attribute**: If yes, add `#[stamp]` to the field in the config struct
3. **Test verification**: The proc macro automatically includes stamped fields
4. **Document intent**: Update field documentation to clarify byte-level impact

Example:
```rust
#[derive(Stamp)]
pub struct DetectionConfig {
    #[stamp] pub confidence: f32,           // Affects detection results
    #[stamp] pub crop_classes: HashSet<DetectionClass>, // Affects cropping output
    pub device: String,                     // Performance only - not stamped
}
```

## Adding New Output Types

When adding new output file types:

1. **Use OutputManager**: Call `generate_*_output_with_tracking()` methods with `track=true` (default)
2. **Avoid duplicate tracking**: Do not track outputs in both result structs and OutputManager
3. **Consider metadata**: TOML files should use `track=false` per design invariant

## Adding New Processing Modes

When adding new processing modes (new tools):

1. **Implement ModelProcessor trait**: Define config, result, and processing logic
2. **Add tool name**: Implement `tool_name()` method in ModelConfig trait
3. **Update stamp generation**: Add new case to `generate_stamps_for_tool()` function
4. **Create config struct**: Use `#[derive(Stamp)]` with appropriate `#[stamp]` attributes

## Testing Make Integration

The `tests/make_integration/` directory contains comprehensive tests:

- **Basic incrementality**: Verify no rebuild when nothing changes
- **Input sensitivity**: Verify rebuild when source files change
- **Configuration sensitivity**: Verify rebuild when stamped parameters change
- **Model sensitivity**: Verify rebuild when custom models change
- **Metadata preservation**: Verify shared metadata doesn't trigger rebuilds

Run tests with:
```bash
cd beaker/tests/make_integration
./test_make_integration.sh
```

Set `BEAKER_STAMP_DIR` to use custom stamp directories for isolated testing.

## Limitations and Failure Modes

### Known Limitations

1. **Model URL changes**: Changes to `--model-url` require manual cache clearing
2. **Environment dependencies**: Changes to system ONNX runtime not detected
3. **Concurrent builds**: Stamp file updates are not atomic across multiple targets
4. **Clock skew**: Incorrect system time can cause false rebuilds or missed changes

### Common Failure Scenarios

1. **Corrupted stamp files**: Malformed JSON causes stamp generation to fail
2. **Missing stamp directories**: Automatic creation may fail with permission issues
3. **Path encoding issues**: Non-UTF8 paths may cause Make escaping problems
4. **Large dependency lists**: Very long prerequisite lists may hit Make line limits

### Debugging Tips

1. **Check stamp contents**: Stamp files contain human-readable configuration hashes
2. **Verify prerequisites**: Use `make -n` to see what Make thinks needs rebuilding
3. **Clean stamp cache**: Delete stamp directory to force full rebuild
4. **Enable debug logging**: Use `RUST_LOG=debug` for detailed stamp generation logs

## CI Integration

Make integration tests run automatically in CI through the workflow defined in `.github/workflows/make-integration.yml`. Tests use isolated stamp directories to prevent interference with other builds.
