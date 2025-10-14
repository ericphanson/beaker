# CRITICAL-03: Multi-File Basename Collision Causes Silent Data Loss

## Status
🔴 **CRITICAL** - Silent data loss

## Priority
**P0** - Must fix before next release

## Category
Data Integrity / Output Management

## Affected Components
- `beaker/src/output_manager.rs` (lines 44-48, 77)
- All processing commands (detect, cutout, quality)

## Problem Description

### What's Broken
When processing multiple files with the same basename from different directories to a single output directory, the second file silently overwrites the first file's outputs. No warning or error is issued.

### Root Cause
The `input_stem()` method only extracts the filename, losing directory context. When multiple files have the same name, they generate identical output paths.

**Current Code** (`output_manager.rs:44-48`):
```rust
pub fn input_stem(&self) -> &str {
    self.input_path
        .file_stem()  // ← Only extracts filename, loses directory
        .and_then(|s| s.to_str())
        .unwrap_or("output")
}
```

**Output path generation** (`output_manager.rs:77`):
```rust
let output_filename = format!("{input_stem}_{default_suffix}.{extension}");
// Both /photos/A/bird.jpg and /photos/B/bird.jpg generate:
// → bird_crop_head.jpg (identical!)
```

### Evidence (Scenario Walkthrough)

**Batch processing example**:
```bash
$ tree
photos/
  ├── A/
  │   └── bird.jpg
  └── B/
      └── bird.jpg

$ beaker detect photos/A/bird.jpg photos/B/bird.jpg --crop head --output-dir /output

# Output:
# [1/2] Processing photos/A/bird.jpg → /output/bird_crop_head.jpg ✓
# [2/2] Processing photos/B/bird.jpg → /output/bird_crop_head.jpg ✓
#                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                       Same path! Second overwrites first!

$ ls /output
bird_crop_head.jpg    # Only ONE file! A's crop is GONE
```

## Impact Assessment

### User Impact
- **Severity**: CRITICAL - Silent data loss
- **Frequency**: Common when processing organized photo libraries (Date/Location/etc folders)
- **Symptoms**:
  - Fewer output files than input files
  - No error messages
  - Last processed file "wins"
  - User doesn't know which files were overwritten

### System Impact
- Wasted computation (processes all files but keeps only last)
- Confusing batch processing results
- Difficult to debug (no logs indicating overwrite)

### Data Loss Risk
- **High**: Overwrites completed processing results silently
- User cannot recover overwritten files (no backup)
- Especially bad for long-running batch jobs

## Reproduction Steps

### Minimal Reproduction
```bash
# Create test files
mkdir -p test/{dir1,dir2}
cp example.jpg test/dir1/bird.jpg
cp example.jpg test/dir2/bird.jpg

# Process both
beaker detect test/dir1/bird.jpg test/dir2/bird.jpg --crop head --output-dir output/

# Check output
ls output/
# Expected: 2 files
# Actual: 1 file (bird_crop_head.jpg)

# Verify which one survived
md5sum output/bird_crop_head.jpg test/dir1/bird.jpg test/dir2/bird.jpg
# Matches dir2 (last processed)
```

### Verification
```bash
# Batch with glob
beaker detect **/bird.jpg --crop head --output-dir output/

# Count inputs vs outputs
INPUT_COUNT=$(find . -name "bird.jpg" | wc -l)
OUTPUT_COUNT=$(ls output/ | wc -l)
echo "Inputs: $INPUT_COUNT, Outputs: $OUTPUT_COUNT"
# If counts differ, collision occurred
```

## Proposed Solution

### Approach
Detect collisions and either: (A) error with helpful message, (B) auto-append counter, or (C) preserve directory structure.

### Implementation Plan

**Option A: Error on Collision (Recommended for safety)**
```rust
// In output_manager.rs
use std::collections::HashSet;

pub struct BatchOutputTracker {
    used_paths: HashSet<PathBuf>,
}

impl BatchOutputTracker {
    pub fn check_collision(&mut self, output_path: &Path) -> Result<()> {
        if self.used_paths.contains(output_path) {
            return Err(anyhow::anyhow!(
                "Output path collision detected: {}\n\
                 Multiple input files would generate the same output.\n\
                 Solutions:\n\
                 1. Process files separately\n\
                 2. Rename input files to have unique basenames\n\
                 3. Use --force-overwrite to allow (not recommended)",
                output_path.display()
            ));
        }
        self.used_paths.insert(output_path.to_path_buf());
        Ok(())
    }
}
```

**Option B: Auto-Append Counter**
```rust
pub fn generate_unique_path(&self, base_path: &Path) -> PathBuf {
    if !base_path.exists() {
        return base_path.to_path_buf();
    }

    let stem = base_path.file_stem().unwrap().to_str().unwrap();
    let extension = base_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    for counter in 2..=9999 {
        let new_name = if extension.is_empty() {
            format!("{stem}-{counter}")
        } else {
            format!("{stem}-{counter}.{extension}")
        };

        let new_path = base_path.with_file_name(new_name);
        if !new_path.exists() {
            log::warn!("Output collision avoided: {} → {}",
                base_path.display(), new_path.display());
            return new_path;
        }
    }

    panic!("Could not find unique output path after 9999 attempts");
}
```

**Option C: Preserve Directory Structure**
```rust
pub fn get_output_path_with_structure(&self) -> Result<PathBuf> {
    let output_dir = self.get_output_dir()?;

    // Find common ancestor between input files in batch
    // Preserve relative path structure from that ancestor
    // Example: photos/A/bird.jpg → output/A/bird_crop_head.jpg
    //          photos/B/bird.jpg → output/B/bird_crop_head.jpg

    let relative_path = self.input_path
        .strip_prefix(common_ancestor)?
        .to_path_buf();

    let output_subdir = output_dir.join(relative_path.parent().unwrap());
    fs::create_dir_all(&output_subdir)?;

    // Generate filename with suffix
    let filename = format!("{stem}_{suffix}.{ext}");
    Ok(output_subdir.join(filename))
}
```

### Files to Modify
1. `beaker/src/output_manager.rs`:
   - Add collision detection (Option A)
   - OR add auto-numbering (Option B)
   - OR add directory preservation (Option C)

2. `beaker/src/model_processing.rs`:
   - Pass `BatchOutputTracker` to processing loop (Option A)
   - Check collisions before processing each file

### Testing Requirements
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_basename_collision_detection() {
        // Create two files with same basename in different dirs
        // Process both to same output dir
        // Verify error is raised (Option A) OR both files created (Option B/C)
    }

    #[test]
    fn test_no_collision_when_unique() {
        // Process files with unique basenames
        // Verify no errors or warnings
    }

    #[test]
    fn test_counter_incrementing() {
        // Process same file 3 times (if Option B)
        // Verify: file.jpg, file-2.jpg, file-3.jpg
    }

    #[test]
    fn test_directory_structure_preserved() {
        // If Option C
        // Verify relative paths are maintained
    }
}
```

### Validation Criteria
- [ ] Two files with same basename can be processed without data loss
- [ ] Clear error message (Option A) or unique outputs (Option B/C)
- [ ] Existing behavior unchanged for single-file processing
- [ ] Works with glob patterns: `beaker detect **/bird.jpg`
- [ ] Works with explicit file lists

## Dependencies
- **Blockers**: None
- **Related**: CRITICAL-04 (overwrite protection - should implement together)
- **Complements**: UX-02 (read-only directory handling)

## Decisions Required

### DECISION-1: Which collision resolution strategy?
**Options**:
- **A**: Error on collision (safe, forces user to resolve)
- **B**: Auto-append counter (convenient, might surprise users)
- **C**: Preserve directory structure (intuitive, changes output layout)

**Recommendation**: **Option A** for safety + **Option B** as optional flag
```bash
# Default: error on collision
beaker detect dir1/bird.jpg dir2/bird.jpg --output-dir output/
# ERROR: Output collision detected

# With flag: auto-number
beaker detect dir1/bird.jpg dir2/bird.jpg --output-dir output/ --auto-number
# Creates: bird_crop_head.jpg, bird_crop_head-2.jpg
```

**Pros/Cons**:
- **Option A**: Safest, but requires user intervention
- **Option B**: Convenient, but might hide user error (two legitimately different files)
- **Option C**: Most intuitive, but breaking change (output layout differs)

**Decision needed from**: @maintainer

### DECISION-2: Counter format?
If Option B chosen:
- **A**: `bird-2.jpg` (dash separator)
- **B**: `bird_2.jpg` (underscore separator)
- **C**: `bird(2).jpg` (parentheses)

**Recommendation**: Option A (`bird-2.jpg`)
- Consistent with existing suffix pattern (`bird_crop_head.jpg` uses underscore for semantic suffix)
- Dash distinguishes "counter" from "semantic suffix"

**Decision needed from**: @maintainer

## Estimated Effort
- **Investigation**: ✅ Complete
- **Implementation**: 4-6 hours
  - Option A (error): 2-3 hours + tests
  - Option B (counter): 3-4 hours + tests
  - Option C (structure): 5-6 hours + tests
- **Review**: 1 hour
- **Total**: ~1 day

## Success Metrics
- Zero silent overwrites in batch processing
- Clear error messages guide users to resolution
- Integration tests pass with multiple files named `bird.jpg`

## Rollback Plan
If collision detection causes issues:
1. Add `--skip-collision-check` flag to bypass
2. Revert to current behavior with warning log
3. Document collision risk in README

## Breaking Changes
**Option A**: None (errors on previously silent failures)
**Option B**: None (only changes output when collision occurs)
**Option C**: YES - output directory structure changes

## References
- **Agent Report**: Output Management Analysis, Bug #2
- **Related Issues**:
  - CRITICAL-04: No overwrite protection
  - UX-02: Read-only directory handling

## Notes for Implementer
- Consider interaction with `--output-dir`: collisions only occur when explicitly set
- Without `--output-dir`, files go to their respective input directories (no collision)
- Test with glob patterns: `**/*.jpg` can produce many collisions
- Ensure error message includes paths of BOTH conflicting inputs
- Log which file "won" if overwrite occurs in current code (helps debugging)
