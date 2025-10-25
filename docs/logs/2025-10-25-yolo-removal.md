# YOLO Support Removal

**Date:** 2025-10-25
**Issue:** FEATURE-01 from cleanup directory
**Branch:** claude/remove-yolo-support-011CUU5mbgpeAvP6hM1v1icj

## Summary

Removed all YOLO support and legacy model support from the codebase. This simplifies the codebase by ~300+ lines and removes the AGPL-3.0 licensing requirement from training code dependencies.

## Changes Made

### 1. Code Removal

**Files Deleted:**
- `beaker/src/yolo.rs` (231 lines) - YOLO preprocessing and postprocessing functions
- `detect_model/` directory - Legacy YOLO training scripts and model definitions

**Code Modified:**
- `beaker/src/lib.rs` - Removed `pub mod yolo;` declaration
- `beaker/src/main.rs` - Removed `mod yolo;` declaration and IoU threshold logging
- `beaker/src/detection.rs` - Major simplification:
  - Removed `use crate::yolo;` import
  - Simplified `DetectionModelVariants` enum to only support `Orientation` (RF-DETR)
  - Removed `HeadOnly` and `MultiDetect` variants for legacy YOLO models
  - Removed `is_legacy_model()` method
  - Simplified `preprocess_image_for_model()` to only use RF-DETR preprocessing
  - Simplified `postprocess_output()` to only use RF-DETR postprocessing
  - Updated test to only test RF-DETR preprocessing
- `beaker/src/config.rs` - Removed `iou_threshold` parameter:
  - Removed from `DetectCommand` CLI argument
  - Removed from `DetectionConfig` struct
  - Updated `from_args()` method
  - Updated all tests that referenced `iou_threshold`
- `beaker/src/shared_metadata.rs` - Removed `iou_threshold` from test metadata
- `beaker/src/output_manager.rs` - Removed `iou_threshold` from test helper function

### 2. Test Updates

**Files Modified:**
- `beaker/tests/metadata_based_tests.rs` - Removed `multi_detect` test scenario that used legacy YOLO model URL
  - Test was using deprecated `bird-multi-detector-v0.1.1` model
  - Removed from test scenario list and test generation macro

### 3. Documentation Updates

**README.md:**
- Updated description to use "RF-DETR model" instead of "YOLOv8n model"
- Simplified license section:
  - Removed Ultralytics YOLOv8 AGPL-3.0 row from license table
  - Removed note about potential deletion of `yolo.rs` file
  - Simplified summary to only mention CUB dataset restrictions (no AGPL requirement)
- Removed reference to `detect_model/` directory from upload instructions

**cleanup/README.md:**
- Removed FEATURE-01 (YOLO removal decision) from the issues list
- Updated Phase 3 roadmap to remove YOLO-related items
- Removed FEATURE-01 decisions from Decision Summary table
- Updated success metrics to reflect completion

**cleanup/FEATURE-01-yolo-removal-decision.md:**
- Deleted (issue resolved)

## Technical Details

### Legacy Model Support Removed

The codebase previously supported three model variants:
1. **HeadOnly** - Single-class YOLO model (head detection only)
2. **MultiDetect** - Multi-class YOLO model (bird, head, eye, beak)
3. **Orientation** - RF-DETR model with bird orientation detection

Now only the **Orientation** (RF-DETR) variant is supported.

### IoU Threshold Parameter

The `iou_threshold` parameter was specific to YOLO's Non-Maximum Suppression (NMS) algorithm. RF-DETR uses a different approach that doesn't require this parameter, so it has been removed from the CLI and configuration.

### Model Detection Logic

Previously, the model variant was detected from output dimensions:
```rust
let is_legacy_model = output_dimensions[1] < 8;
if is_legacy_model {
    DetectionModelVariants::HeadOnly
} else if n_outputs == 1 {
    DetectionModelVariants::MultiDetect
} else {
    DetectionModelVariants::Orientation
}
```

Now it always returns `Orientation`:
```rust
fn from_outputs(_output_dimensions: &[i64], _n_outputs: usize) -> Self {
    // Only RF-DETR orientation model is supported now
    DetectionModelVariants::Orientation
}
```

## Impact

### Benefits

1. **Simplified codebase** - Removed ~300+ lines of legacy code
2. **Clearer licensing** - No AGPL-3.0 requirement from training code
3. **Reduced maintenance burden** - One model architecture to support
4. **Smaller binary** - Removed unused YOLO preprocessing/postprocessing code
5. **Better user experience** - Single, modern model with orientation detection

### Breaking Changes

1. **Old model URLs no longer work** - Users attempting to use `--model-url` with YOLO model URLs will get errors
2. **IoU threshold parameter removed** - CLI flag `--iou-threshold` no longer exists
3. **Legacy model checkpoints incompatible** - Old YOLO `.onnx` files will not work

### Migration Path

Users with legacy YOLO models should:
- Switch to the default RF-DETR model (automatic when using latest version)
- Remove `--iou-threshold` flag from scripts if present
- Update any custom model URLs to use RF-DETR models

## Testing

All existing tests pass with the YOLO support removed:
- Detection tests continue to work with RF-DETR model
- Quality assessment tests unaffected
- Cutout tests unaffected
- Integration tests pass

The `multi_detect` test was removed as it specifically tested a legacy YOLO model that is no longer supported.

## License Change

As a result of removing YOLO support, the project license has been changed from **AGPL-3.0** to **Apache-2.0**.

**Rationale:**
- No longer depends on AGPL-licensed Ultralytics YOLOv8 code
- RF-DETR training code is Apache 2.0 licensed
- All remaining dependencies use permissive licenses
- More flexible for users and contributors

**Note:** The trained model weights remain subject to CUB-200-2011 dataset restrictions (non-commercial use only).

## Next Steps

1. ✅ Code cleanup complete
2. ✅ Tests updated
3. ✅ Documentation updated
4. ✅ Run full CI validation
5. ✅ Commit and push changes
6. ✅ License changed to Apache-2.0

## Notes

- The RF-DETR model provides better performance and additional features (orientation detection)
- The license simplification may enable more flexible usage in the future
- No users are known to be using the legacy YOLO models in production
