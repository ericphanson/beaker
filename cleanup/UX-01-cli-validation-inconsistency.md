# UX-01: CLI Validation Inconsistency Across Commands

## Status
🟡 **HIGH** - Inconsistent developer/user experience

## Priority
**P1** - Should fix soon

## Category
Code Quality / Developer Experience

## Affected Components
- `beaker/src/config.rs` (DetectCommand, CutoutCommand, QualityCommand)
- `beaker/src/main.rs` (command dispatch, lines 169-176, 225, 257)

## Problem Description

### What's Broken
DetectCommand can validate arguments during config creation (returns `Result<Self, String>`), but CutoutCommand and QualityCommand cannot because their `from_args()` methods return `Self` directly (not `Result`).

### Root Cause
Inconsistent function signatures across config creation methods.

**DetectCommand validation** (`main.rs:169-176`):
```rust
let internal_config =
    match DetectionConfig::from_args(cli.global.clone(), detect_cmd.clone()) {
        Ok(config) => config,  // ✓ Can validate
        Err(e) => {
            error!("{} Configuration error: {e}", symbols::operation_failed());
            std::process::exit(1);
        }
    };
```

**CutoutCommand NO validation** (`main.rs:225`):
```rust
let internal_config = CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone());
// ⚠️ Cannot return validation errors!
```

**QualityCommand NO validation** (`main.rs:257`):
```rust
let internal_config = QualityConfig::from_args(cli.global.clone(), quality_cmd.clone());
// ⚠️ Cannot return validation errors!
```

### Evidence (Implementation Details)

**DetectCommand signature** (supports validation):
```rust
impl DetectionConfig {
    pub fn from_args(global: GlobalArgs, cmd: DetectCommand) -> Result<Self, String> {
        // Can validate and return errors
        let crop_classes = if let Some(crop_str) = &cmd.crop {
            Some(parse_crop_classes(crop_str)?)  // ← Validation!
        } else {
            None
        };

        Ok(Self { /* ... */ })
    }
}
```

**CutoutCommand signature** (no validation):
```rust
impl CutoutConfig {
    pub fn from_args(global: GlobalArgs, cmd: CutoutCommand) -> Self {
        // Cannot validate - must return Self
        Self { /* ... */ }
    }
}
```

## Impact Assessment

### Developer Impact
- **Severity**: MEDIUM - Makes adding validation harder
- **Frequency**: Every new validation rule for cutout/quality
- **Symptoms**:
  - Cannot add parse-time validation to cutout/quality
  - Validation must happen during processing (late errors)
  - Inconsistent error handling patterns across commands

### User Impact
- Some validation errors caught early (detect)
- Other validation errors caught late (cutout/quality)
- Inconsistent error message formats
- Some errors show stack traces, others don't

### Maintenance Impact
- Confusing for contributors (why different patterns?)
- Copy-paste from detect→cutout requires refactoring
- Harder to add cross-command validation

## Reproduction Steps

### Demonstrate Inconsistency
```bash
# DetectCommand: Validation at parse time
beaker detect image.jpg --crop "invalid_class"
# ERROR: Configuration error: Unknown detection class: invalid_class
# ✓ Fails BEFORE model loading

# CutoutCommand: Cannot validate at parse time
beaker cutout image.jpg --alpha-matting-foreground-threshold 999
# (parameter accepts out-of-range value, no validation)
# Would need runtime validation in cutout_processing.rs
```

### Try to Add Validation to Cutout
```rust
impl CutoutConfig {
    pub fn from_args(global: GlobalArgs, cmd: CutoutCommand) -> Self {
        // Want to add validation here...
        if cmd.alpha_matting && cmd.background_color.is_some() {
            // ERROR: Cannot return Err! Signature is -> Self not -> Result<Self>
            ???
        }
        Self { /* ... */ }
    }
}
```

## Proposed Solution

### Approach
Standardize all `from_args()` methods to return `Result<Self, String>` for consistency.

### Implementation Plan

**Step 1: Update CutoutConfig**
```rust
// In config.rs
impl CutoutConfig {
    pub fn from_args(global: GlobalArgs, cmd: CutoutCommand) -> Result<Self, String> {
        // Add validations
        if cmd.alpha_matting && cmd.background_color.is_some() {
            return Err(
                "Cannot use both --alpha-matting and --background-color. Choose one.".to_string()
            );
        }

        // Validate threshold ranges
        if cmd.alpha_matting_foreground_threshold > 255 {
            return Err(format!(
                "Invalid foreground threshold: {} (must be 0-255)",
                cmd.alpha_matting_foreground_threshold
            ));
        }

        Ok(Self {
            base: BaseModelConfig { /* ... */ },
            // ... rest unchanged
        })
    }
}
```

**Step 2: Update QualityConfig**
```rust
// In config.rs
impl QualityConfig {
    pub fn from_args(global: GlobalArgs, cmd: QualityCommand) -> Result<Self, String> {
        // Can add validations in future
        // For now, just wrap in Ok()

        Ok(Self {
            base: BaseModelConfig { /* ... */ },
            // ... rest unchanged
        })
    }
}
```

**Step 3: Update Call Sites**
```rust
// In main.rs:225 (cutout)
let internal_config = match CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone()) {
    Ok(config) => config,
    Err(e) => {
        error!("{} Configuration error: {e}", symbols::operation_failed());
        std::process::exit(1);
    }
};

// In main.rs:257 (quality)
let internal_config = match QualityConfig::from_args(cli.global.clone(), quality_cmd.clone()) {
    Ok(config) => config,
    Err(e) => {
        error!("{} Configuration error: {e}", symbols::operation_failed());
        std::process::exit(1);
    }
};
```

### Files to Modify
1. `beaker/src/config.rs`:
   - Change `CutoutConfig::from_args()` signature (line ~400)
   - Change `QualityConfig::from_args()` signature (line ~500)
   - Add validations to both
2. `beaker/src/main.rs`:
   - Update cutout call site (line 225)
   - Update quality call site (line 257)

### Testing Requirements
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_cutout_conflicting_flags_validation() {
        let result = CutoutConfig::from_args(
            test_global_args(),
            CutoutCommand {
                alpha_matting: true,
                background_color: Some([255, 0, 0, 255]),
                ..Default::default()
            }
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot use both"));
    }

    #[test]
    fn test_cutout_valid_config() {
        let result = CutoutConfig::from_args(
            test_global_args(),
            CutoutCommand {
                alpha_matting: true,
                background_color: None,
                ..Default::default()
            }
        );
        assert!(result.is_ok());
    }
}
```

### Validation Criteria
- [ ] All three commands use `Result<Self, String>` signature
- [ ] Validation errors caught before model loading
- [ ] Error messages are consistent across commands
- [ ] Existing tests pass
- [ ] New validation tests added

## Dependencies
- **Blockers**: None
- **Enables**: Many other validation improvements
  - UX-02: Conflicting flags validation
  - UX-03: Numeric range validation
  - UX-04: Model flag validation
- **Related**: None (standalone refactoring)

## Decisions Required

### DECISION-1: Error type?
**Options**:
- **A**: Keep `String` (current detect pattern)
- **B**: Use `anyhow::Error` (more flexible)
- **C**: Define custom `ConfigError` enum

**Recommendation**: Keep `String` (Option A)
- Consistent with existing DetectCommand
- Simple validation errors don't need structured errors
- Can migrate to anyhow::Error later if needed

**Decision needed from**: @maintainer

### DECISION-2: What validations to add initially?
**For CutoutConfig**:
- [x] Conflicting flags (alpha-matting + background-color)
- [x] Threshold ranges (0-255)
- [ ] Erode size sanity check (< image size)

**For QualityConfig**:
- [ ] Currently no validation needed (just wrap in Ok())
- [ ] Future: quality threshold ranges if flags added

**Recommendation**: Add conflicting flags validation immediately, threshold ranges can be separate issue

**Decision needed from**: @maintainer

## Estimated Effort
- **Investigation**: ✅ Complete
- **Implementation**: 2-3 hours
  - Update signatures: 30 minutes
  - Update call sites: 30 minutes
  - Add initial validations: 1 hour
  - Add tests: 1 hour
- **Review**: 30 minutes
- **Total**: ~0.5 day

## Success Metrics
- All commands have consistent validation patterns
- New validations can be added easily
- Error messages are clear and actionable
- No runtime panics from invalid configs

## Rollback Plan
If breaking changes cause issues:
1. Keep new signature but make validations warnings instead of errors
2. Add `--skip-validation` flag temporarily
3. Revert to old signatures (lose validation improvements)

## Breaking Changes
**None** - This is internal API only. CLI interface unchanged.

## Future Enhancements
After this fix, can easily add:
- Numeric range validation (confidence 0-1, thresholds 0-255)
- Model path + URL mutual exclusion
- Device validation (valid values only)
- Output directory writability checks

## References
- **Agent Report**: CLI Routing Analysis, Bug #1
- **Related Issues**:
  - UX-02: Conflicting flags validation
  - UX-03: Numeric range validation
  - UX-04: Model flag validation

## Notes for Implementer
- This is a pure refactoring - behavior should not change yet
- Can add validations incrementally after signature change
- Consider making helper function for common error formatting:
  ```rust
  fn config_error(msg: impl Display) -> String {
      format!("Configuration error: {}", msg)
  }
  ```
- Update documentation if signatures are part of public API
- Ensure error messages include hints for fixing (not just "invalid value")
