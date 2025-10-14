# CRITICAL-01: Model Download Race Condition Causing Silent Corruption

## Status
🔴 **CRITICAL** - Can cause silent data corruption

## Priority
**P0** - Must fix before next release

## Category
Data Integrity / Concurrency

## Affected Components
- `beaker/src/model_access.rs` (lines 268-439)
- Model cache system

## Problem Description

### What's Broken
Two processes downloading the same model simultaneously can corrupt the cached model file, causing permanent checksum failures that require manual cache deletion to fix.

### Root Cause
The lock timeout mechanism (5 minutes) allows a second process to remove the lock file and start downloading while the first process is still writing, leading to interleaved writes to the same file.

**Race Condition Timeline**:
```
T=0s:    Process A creates lock, starts download (slow network)
T=280s:  Process B starts, waits for lock
T=300s:  Process B timeout, removes lock, starts download  ← BUG
T=310s:  Process A finishes, writes final bytes
T=315s:  Process B writes its bytes
Result:  File contains mixed data from both downloads
         Checksum will NEVER match either expected value
```

### Evidence (Code References)

**Lock timeout logic** (`model_access.rs:384-404`):
```rust
if start_time.elapsed() > MAX_WAIT_TIME {
    log::warn!("Download lock timeout ({}s), forcing download (removing stale lock)",
        MAX_WAIT_TIME.as_secs());
    let _ = fs::remove_file(lock_path);  // ← Removes potentially valid lock!
    return download_model(&model_info.url, model_path);  // ← Second download starts
}
```

**No file-level locking** (`model_access.rs:177`):
```rust
let mut file = fs::File::create(output_path)?;  // ← No exclusive lock
// Both processes can write to this simultaneously
```

## Impact Assessment

### User Impact
- **Severity**: CRITICAL - Requires manual intervention to fix
- **Frequency**: Rare on single-user systems, common on multi-user/CI systems
- **Symptoms**:
  - Persistent checksum mismatch errors
  - "Downloaded model failed checksum verification" message
  - Downloads succeed but immediately fail verification
  - Infinite download loop (download → verify → fail → delete → repeat)

### System Impact
- Corrupted cache files that cannot self-heal
- Wasted bandwidth (repeated download attempts)
- CI pipeline failures requiring cache cleanup

### Data Loss Risk
- **Model files**: Can be re-downloaded (annoying but not data loss)
- **User data**: Not affected

## Reproduction Steps

### Minimal Reproduction
```bash
# Terminal 1
export BEAKER_DETECT_MODEL_URL="http://slow-server.com/model.onnx"
beaker detect image.jpg

# Terminal 2 (start within 5 minutes of Terminal 1)
beaker detect image2.jpg

# Wait for Terminal 1 to reach 5:01 runtime
# Terminal 2 will timeout and start second download
# Terminal 1 and 2 now write to same file simultaneously
```

### Verification
```bash
# Check for corrupted cache
md5sum ~/.cache/onnx-models/*.onnx
# Compare against expected checksum from model_access.rs
# Corrupted file will have different checksum
```

## Proposed Solution

### Approach
Use proper file locking instead of lock files + timeout.

### Implementation Plan

**Option A: Advisory File Locks (Recommended)**
```rust
use fs2::FileExt;  // Add dependency: fs2 = "0.4"

pub fn download_model(url: &str, model_path: &Path) -> Result<PathBuf> {
    // Create lock file with exclusive lock
    let lock_path = model_path.with_extension("lock");
    let lock_file = fs::File::create(&lock_path)?;

    // Try to acquire exclusive lock (blocks until available)
    lock_file.lock_exclusive()?;

    // Check if model was downloaded while we waited
    if model_path.exists() {
        // Validate checksum
        return Ok(model_path.to_path_buf());
    }

    // Download to temp file
    let tmp_path = model_path.with_extension("tmp");
    // ... download logic ...

    // Atomic rename (only on success)
    fs::rename(&tmp_path, model_path)?;

    // Lock automatically released when lock_file is dropped
    Ok(model_path.to_path_buf())
}
```

**Option B: PID-Based Lock Validation**
```rust
fn is_lock_stale(lock_path: &Path) -> Result<bool> {
    let lock_content = fs::read_to_string(lock_path)?;
    let pid: u32 = lock_content.parse()?;

    // Check if process still exists (Unix-specific)
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        match kill(Pid::from_raw(pid as i32), Signal::SIGCONT) {
            Ok(_) => Ok(false),  // Process exists, lock is valid
            Err(_) => Ok(true),  // Process doesn't exist, lock is stale
        }
    }

    #[cfg(not(unix))]
    {
        // Fallback: check lock age (current behavior)
        let lock_age = lock_path.metadata()?.modified()?.elapsed()?;
        Ok(lock_age > Duration::from_secs(300))
    }
}
```

### Files to Modify
1. `beaker/Cargo.toml` - Add dependency: `fs2 = "0.4"` (for Option A)
2. `beaker/src/model_access.rs` - Rewrite `download_with_lock()` function (lines 268-439)

### Testing Requirements
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_concurrent_downloads() {
        // Spawn 3 threads downloading same model
        // Verify only one download occurs
        // Verify all threads get same valid file
    }

    #[test]
    fn test_lock_prevents_corruption() {
        // Simulate slow download
        // Start second download mid-first
        // Verify second waits for first
        // Verify file checksum is correct
    }
}
```

### Validation Criteria
- [ ] Two simultaneous downloads complete without corruption
- [ ] Checksum verification succeeds after concurrent access
- [ ] Lock is released if process crashes (test with SIGKILL)
- [ ] Lock is released if process completes normally
- [ ] Lock timeout still works for truly stale locks (Option B only)

## Dependencies
- **Blockers**: None
- **Blocked by this**: CRITICAL-02 (partial downloads - should be fixed together)
- **Related**: CRITICAL-03 (cache cleanup), MEDIUM-08 (cache size limits)

## Decisions Required

### DECISION-1: Which locking approach?
**Options**:
- **A**: Advisory file locks (fs2 crate) - Most robust, requires new dependency
- **B**: PID-based validation - No new dependency, less robust on Windows

**Recommendation**: Option A (advisory file locks)
- More reliable across platforms
- Standard approach in Rust ecosystem
- fs2 is lightweight (~50KB) and well-maintained

**Decision needed from**: @maintainer

### DECISION-2: Lock timeout behavior?
**Options**:
- **A**: No timeout - wait indefinitely (file locks block until available)
- **B**: Keep 5min timeout - fail with clear error message
- **C**: Configurable timeout via env var

**Recommendation**: Option A (no timeout with file locks)
- File locks automatically released on process termination
- No false positives from slow networks
- Simpler code

**Decision needed from**: @maintainer

## Estimated Effort
- **Investigation**: ✅ Complete
- **Implementation**: 4-6 hours
  - Add fs2 dependency: 5 minutes
  - Rewrite download_with_lock(): 2-3 hours
  - Add tests: 1-2 hours
  - Manual testing: 1 hour
- **Review**: 1 hour
- **Total**: ~1 day

## Success Metrics
- Zero checksum corruption reports after deployment
- CI pipelines with concurrent beaker runs succeed
- Lock files properly cleaned up on crash (manual test)

## Rollback Plan
If file locking causes issues:
1. Revert to current timeout-based approach
2. Add better logging to diagnose lock timeouts
3. Document that concurrent downloads are unsupported

## References
- **Agent Report**: Model Management Analysis, Bug #3
- **fs2 Crate**: https://docs.rs/fs2/latest/fs2/
- **Related Issues**:
  - CRITICAL-02: Partial downloads
  - MEDIUM-08: Cache size limits

## Notes for Implementer
- Test on both macOS and Linux (file locking behavior differs)
- Consider Windows behavior (file locks are mandatory, not advisory)
- Ensure lock files are cleaned up in all error paths
- Log when waiting for lock (user feedback that it's not hung)
