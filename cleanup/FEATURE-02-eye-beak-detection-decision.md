# FEATURE-02: Eye/Beak Detection - Finish or Remove?

## Status
⚠️ **DECISION REQUIRED** - Half-implemented feature

## Priority
**P2** - Documentation/feature clarity needed

## Category
Feature Completeness / Documentation

## Problem
Eye and Beak detection classes are defined and model supports them, BUT:
- Bounding box visualization deliberately excludes them
- Test comment: "beak not being detected anymore! not sure why"
- Unreliable detection quality

**Current state**:
- Can crop: `--crop=eye,beak` ✓
- Cannot visualize: bounding boxes only show bird/head ✗
- Quality: Unreliable (test failures)

## Decision Required

### DECISION-1: Eye/Beak detection support?
**Options**:
- **A**: Document as experimental (keep but warn users)
- **B**: Finish implementation (make bounding box visualization work)
- **C**: Remove from DetectionClass enum entirely

**Recommendation**: Option A (document as experimental)
- Minimal code changes
- Users can opt-in with `--crop=eye,beak`
- Set expectations (unreliable)

**Decision needed from**: @ericphanson

### DECISION-2: If keeping, fix visualization?
Add `--show-all-classes` flag to include eye/beak in bounding box visualization

## Estimated Effort
- Option A: 30 min (docs only)
- Option B: 4 hours (visualization + tests)
- Option C: 2 hours (removal + tests)

## References
**Agent Report**: Dead Code Analysis, Issue #2
