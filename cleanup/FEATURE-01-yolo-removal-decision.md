# FEATURE-01: YOLO Support Removal Decision

## Status
⚠️ **DECISION REQUIRED** - Blocks license change

## Priority
**P2** - Architectural decision needed

## Category
Technical Debt / Licensing / Code Cleanup

## Problem
`yolo.rs` (231 lines) is legacy code for old YOLO models. Default switched to RF-DETR in August 2025. README states "I may delete it..." but hasn't been removed.

**Why kept**:
- Backward compatibility with old YOLO models
- Licensing concern: Removing YOLO code allows switching from AGPL-3.0 to Apache 2.0

**Why remove**:
- 300+ lines of dead/rarely-used code
- Complicates codebase
- No users likely using old YOLO models
- Licensing flexibility

## Impact if Removed
- **Binary size**: -50KB (minimal)
- **License**: Can switch to Apache 2.0 (major win)
- **Maintenance**: Simpler codebase
- **Breaking change**: Users with legacy YOLO models must re-download RF-DETR

## Files to Delete
```
beaker/src/yolo.rs
detect_model/ (entire directory)
tests/.../bird-multi-detector* (legacy test)
config.rs:158-160 (iou_threshold parameter)
detection.rs:37-38, 46-59, 194-196, 532-551 (YOLO variants)
```

## Decision Required

### DECISION-1: Remove YOLO support?
**Options**:
- **A**: Remove immediately (recommended)
- **B**: Deprecate (keep 1 more release with warning)
- **C**: Keep indefinitely

**Recommendation**: Option A (remove immediately)
- No known users on old models
- Unlocks Apache 2.0 licensing
- Simplifies codebase significantly

**Decision needed from**: @ericphanson (maintainer)

### DECISION-2: If removed, license change?
**Options**:
- **A**: Switch to Apache 2.0 immediately
- **B**: Dual license (Apache 2.0 + MIT)
- **C**: Keep AGPL-3.0 for compatibility

**Recommendation**: Option A (Apache 2.0)
- Removes licensing confusion
- More permissive for users
- Matches RF-DETR training code license

**Decision needed from**: @ericphanson

## Estimated Effort
- Remove YOLO: 2-3 hours
- Update README/docs: 1 hour
- Update license: 1 hour
- Testing: 1 hour
- **Total**: ~1 day

## References
**Agent Report**: Dead Code Analysis, Issue #1
