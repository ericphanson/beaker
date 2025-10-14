# CRITICAL-04: Quality Command Produces Zero Output Without --metadata

## Status
🔴 **CRITICAL** - Broken user experience

## Priority
**P0** - Must fix before next release

## Category
User Experience / Feature Completeness

## Affected Components
- `beaker/src/main.rs` (lines 237-267)
- `beaker/src/quality_processing.rs` (entire pipeline)

## Problem Description

### What's Broken
Running `beaker quality image.jpg` succeeds (exit code 0) but creates no output files unless `--metadata` flag is explicitly passed. This is inconsistent with detect/cutout commands which always produce visual outputs.

### Root Cause
The quality command implementation never populates the outputs list, unlike detect and cutout commands.

**Quality command** (`main.rs:237-247`):
```rust
Some(Commands::Quality(quality_cmd)) => {
    // Build outputs list (none for quality assessment)
    let outputs = Vec::<&str>::new();  // ← Always empty!

    let feature_str = ""; // No special features for quality assessment

    let output_str = if outputs.is_empty() {
        "".to_string()
    } else {
        format!(" | outputs: {}", outputs.join(", "))
    };
    // ... proceeds without checking if outputs is empty
```

**Compare to detect** (`main.rs:165-168`):
```rust
if outputs.is_empty() {
    error!("No outputs requested! Pass at least one of `--metadata`, `--crop`, or `--bounding-box`.");
    std::process::exit(1);  // ← Fails if no outputs
}
```

**Compare to cutout** (`main.rs:202-203`):
```rust
let mut outputs = Vec::new();
outputs.push("cutout"); // ← Always produces at least cutout image
```

### Evidence (User Experience)

**Confusing user interaction**:
```bash
$ beaker quality bird.jpg
🔍 Quality assessment | device: auto
✓ Processed bird.jpg (1/1) in 523 ms | quality: 0.847

$ ls
bird.jpg    # No outputs created!

$ echo $?
0           # Exit code 0 (success)
```

User expectations vs reality:
- **Expected**: Some quality visualization (heatmap, score overlay, report)
- **Actual**: Nothing (only log output)
- **With --metadata**: Creates `bird.beaker.toml` with quality scores
- **Problem**: User doesn't know they need `--metadata` flag

## Impact Assessment

### User Impact
- **Severity**: HIGH - Wastes user time and computational resources
- **Frequency**: HIGH - Affects all users trying quality assessment for first time
- **Symptoms**:
  - "Silent success" - command completes but produces nothing
  - Users think command failed or didn't work
  - No indication that `--metadata` is required
  - Different behavior from detect/cutout (inconsistent UX)

### System Impact
- Wasted computation (runs inference, downloads model, produces no output)
- Confusing documentation (what does quality command actually DO?)
- Support burden (users asking "why doesn't it work?")

### Data Loss Risk
- None (no files created, so none lost)
- But: time and bandwidth wasted

## Reproduction Steps

### Minimal Reproduction
```bash
# Run quality command without --metadata
beaker quality example.jpg

# Check outputs
ls -la

# Result: Only example.jpg exists (no new files)
# Exit code: 0 (success)
```

### Comparison with Other Commands
```bash
# Detect: ALWAYS produces output
beaker detect example.jpg --crop head
ls  # Shows: example_crop_head.jpg ✓

# Cutout: ALWAYS produces output
beaker cutout example.jpg
ls  # Shows: example_cutout.png ✓

# Quality: NO output (unless --metadata)
beaker quality example.jpg
ls  # Shows: nothing new ✗

# Quality with --metadata: produces metadata only
beaker quality example.jpg --metadata
ls  # Shows: example.beaker.toml (TOML file, not image)
```

## Proposed Solution

### Approach
Quality command should ALWAYS produce meaningful output. Options:
1. Make `--metadata` implicit (always save)
2. Generate quality visualization image by default
3. Require `--metadata` and fail if not provided (like detect's validation)

### Implementation Plan

**Option A: Always Save Metadata (Recommended)**
```rust
// In main.rs:237-267
Some(Commands::Quality(quality_cmd)) => {
    let mut outputs = Vec::new();

    // Quality always produces metadata (no flag needed)
    outputs.push("metadata");

    let output_str = format!(" | outputs: {}", outputs.join(", "));

    info!(
        "{} Quality assessment | device: {}{}",
        symbols::detection_start(),
        cli.global.device,
        output_str
    );

    // Force metadata to true for quality command
    let mut global_with_metadata = cli.global.clone();
    global_with_metadata.metadata = true;

    let internal_config = QualityConfig::from_args(global_with_metadata, quality_cmd.clone());
    // ... rest unchanged
}
```

**Option B: Generate Visualization Image**
```rust
// In quality_processing.rs - add new output
pub fn save_quality_visualization(
    result: &QualityResult,
    output_path: &Path,
) -> Result<()> {
    // Create heatmap overlay on original image
    // Show quality scores as color-coded grid
    // Add legend with score interpretation

    let viz_path = output_path.with_file_name(
        format!("{}_quality_viz.jpg", output_path.file_stem().unwrap().to_str().unwrap())
    );

    // ... generate visualization ...

    log::info!("Saved quality visualization: {}", viz_path.display());
    Ok(())
}
```

**Option C: Require --metadata Flag**
```rust
// In main.rs:237-267
Some(Commands::Quality(quality_cmd)) => {
    // Validate that metadata is requested
    if !cli.global.metadata {
        error!(
            "{} Quality assessment requires --metadata flag to save results\n\
             Hint: Run with --metadata to save quality scores to .beaker.toml",
            symbols::operation_failed()
        );
        std::process::exit(1);
    }

    let mut outputs = Vec::new();
    outputs.push("metadata");
    // ... rest unchanged
}
```

### Files to Modify

**Option A** (always save metadata):
1. `beaker/src/main.rs` (lines 237-267) - Force metadata flag
2. Update help text to clarify metadata is always saved

**Option B** (add visualization):
1. `beaker/src/quality_processing.rs` - Add visualization generation
2. `beaker/src/blur_detection.rs` - Reuse/refactor heatmap generation
3. `beaker/src/main.rs` - Update outputs list

**Option C** (require flag):
1. `beaker/src/main.rs` (lines 237-267) - Add validation
2. Update help text to clarify `--metadata` is required

### Testing Requirements
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_quality_produces_output() {
        // Run quality command
        let output = Command::new("beaker")
            .args(&["quality", "test.jpg"])
            .output()
            .expect("failed to execute");

        // Verify exit code 0
        assert!(output.status.success());

        // Verify output file exists (metadata or visualization)
        let expected_output = if option_a {
            "test.beaker.toml"
        } else if option_b {
            "test_quality_viz.jpg"
        } else {
            panic!("Should have failed");
        };

        assert!(Path::new(expected_output).exists());
    }

    #[test]
    fn test_quality_without_metadata_behavior() {
        // Option A: Succeeds and creates metadata
        // Option B: Succeeds and creates visualization
        // Option C: Fails with clear error message
    }
}
```

### Validation Criteria
- [ ] `beaker quality image.jpg` produces at least one output file
- [ ] Exit code 0 means output was created successfully
- [ ] Help text clearly documents what quality command produces
- [ ] Consistent with detect/cutout behavior (always produces something)
- [ ] Error message (if Option C) is clear and actionable

## Dependencies
- **Blockers**: None
- **Related**: UX-01 (quality metadata not shown in summary) - same root cause
- **Complements**: FEATURE-03 (quality debug images) - could reuse visualization code

## Decisions Required

### DECISION-1: Which approach?
**Options**:
- **A**: Always save metadata (make --metadata implicit for quality)
- **B**: Generate quality visualization image by default
- **C**: Require --metadata flag and error if not provided

**Recommendation**: **Option A** (always save metadata)

**Pros/Cons**:

| Option | Pros | Cons |
|--------|------|------|
| A | Simple, consistent with "quality data is metadata", no breaking changes | Metadata-only output might not be obvious to users |
| B | Visual output is intuitive, users can SEE quality | Requires designing/implementing visualization UI |
| C | Explicit is better than implicit, forces user choice | Breaking change (existing scripts fail) |

**Decision needed from**: @maintainer

### DECISION-2: If Option B, what visualization?
**Options**:
- **A**: Heatmap overlay showing quality scores per region
- **B**: Simple image with overall score + histogram
- **C**: Side-by-side: original + heatmap
- **D**: Reuse existing debug images from blur_detection.rs

**Recommendation**: Option D (reuse debug heatmaps)
- Already implemented and tested
- Line 472-489 in blur_detection.rs generates comprehensive visualizations
- Just need to move generation logic outside `log::log_enabled!(Debug)` guard

**Decision needed from**: @maintainer

## Estimated Effort
- **Investigation**: ✅ Complete
- **Implementation**:
  - **Option A**: 1-2 hours (simple flag forcing)
  - **Option B**: 6-8 hours (visualization design + implementation)
  - **Option C**: 1 hour (validation only)
- **Testing**: 1-2 hours (all options)
- **Documentation**: 30 minutes
- **Total**: 0.5 day (Option A/C) or 1.5 days (Option B)

## Success Metrics
- Zero user complaints about "quality command doesn't work"
- `beaker quality image.jpg` always creates visible output
- Help text clearly documents output behavior
- Exit code 0 correlates with successful output creation

## Rollback Plan
If change causes issues:
1. Add `--no-metadata` flag to restore old behavior (Option A)
2. Add `--no-visualization` flag (Option B)
3. Revert to current behavior with deprecation warning

## Breaking Changes
- **Option A**: None (only adds output where none existed)
- **Option B**: None (only adds output where none existed)
- **Option C**: YES - existing commands without `--metadata` will fail

## Future Enhancements
After this fix:
- **ENHANCEMENT**: Quality thresholds (--min-quality flag to filter)
- **ENHANCEMENT**: Batch quality report (CSV/JSON output)
- **ENHANCEMENT**: Quality-based image sorting

## References
- **Agent Report**: ModelProcessor Integration Analysis, Bug #1
- **Related Code**:
  - `blur_detection.rs:471-489` - Debug heatmap generation
  - `quality_processing.rs:145-151` - Debug image directory creation
- **Related Issues**:
  - UX-01: Quality metadata not in output summary
  - FEATURE-03: Quality debug images always created

## Notes for Implementer
- If Option A: Ensure `cli.global.metadata` is set BEFORE creating QualityConfig
- If Option B: Consider making visualization optional with `--no-viz` flag
- If Option C: Provide helpful error message with example command
- Update README.md examples to reflect new behavior
- Consider adding quality visualization to README (if Option B)
