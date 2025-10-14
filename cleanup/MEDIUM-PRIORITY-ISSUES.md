# Medium-Priority Issues (Quick Reference)

These issues are lower priority but should be addressed eventually. Each is scoped to 1-2 hours of work.

## MEDIUM-01: Env Var URL Override Bypasses Checksum
**File**: `model_access.rs:669-683`
**Problem**: Setting custom URL without checksum causes infinite download loop (uses wrong checksum)
**Fix**: Disable checksum verification if URL overridden without checksum
**Effort**: 1-2 hours

## MEDIUM-02: Inconsistent Checksum Algorithms
**File**: `cache_common.rs`, all model configs
**Problem**: All use MD5, no migration path to SHA256
**Fix**: Add algorithm prefix (`md5:...` or `sha256:...`) to checksums
**Effort**: 3-4 hours
**Decision**: Switch to SHA256?

## MEDIUM-03: Unbounded Cache Growth
**Files**: `model_access.rs:28-35`, `cache_common.rs`
**Problem**: Model cache grows indefinitely, no cleanup
**Fix**: Implement `beaker cache clean` command with LRU eviction
**Effort**: 6-8 hours

## MEDIUM-04: CoreML Cache Race Condition
**File**: `onnx_session.rs:184-242`
**Problem**: Multiple processes create unique cache dirs (PID+timestamp), wasting space
**Fix**: Use proper file locking for CoreML cache directory
**Effort**: 4-6 hours

## MEDIUM-05: No Overwrite Protection
**All commands**
**Problem**: Re-running same command silently overwrites previous outputs
**Fix**: Add `--force` flag, error by default if output exists
**Effort**: 3-4 hours
**Decision**: Always overwrite vs require flag?

## MEDIUM-06: Metadata Parse Errors Drop Entire File
**File**: `shared_metadata.rs:327-347`
**Problem**: Corrupted metadata loses all sections (detect + cutout + quality)
**Fix**: Create backup before overwriting, better error recovery
**Effort**: 2-3 hours

## MEDIUM-07: Quality Results Discarded by Wrapper
**File**: `model_processing.rs:88-100`
**Problem**: Wrapper function computes quality results but throws them away
**Fix**: Remove wrapper, call `_with_quality_outputs` directly
**Effort**: 1 hour

## MEDIUM-08: File Naming Repeated Suffixes
**File**: `output_manager.rs:77`
**Problem**: `bird_crop_head.jpg` → `bird_crop_head_crop_head.jpg`
**Fix**: Deduplicate suffixes before appending
**Effort**: 1 hour

## MEDIUM-09: Redundant Checksum Verification on Startup
**File**: `model_access.rs:495-506`
**Problem**: Computes MD5 of 100MB file on every run (1-2 seconds delay)
**Fix**: Trust filename-embedded checksum, or use `.verified` timestamp
**Effort**: 2 hours

## MEDIUM-10: Lock File Orphaning
**File**: `model_access.rs:317`
**Problem**: Process crash leaves lock file, blocks downloads for 5 minutes
**Fix**: Implemented by CRITICAL-01 (file locking), no separate work needed
**Effort**: N/A (covered by CRITICAL-01)
