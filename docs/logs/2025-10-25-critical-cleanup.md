# Critical Cleanup Log - 2025-10-25

## CRITICAL-01: Model Download Race Condition - FIXED ✅

### Issue Summary
Two processes downloading the same model simultaneously could corrupt the cached model file, causing permanent checksum failures that required manual cache deletion to fix.

### Root Cause
The previous implementation used a lock file with a 5-minute timeout. When the timeout expired, the second process would:
1. Remove the lock file
2. Start downloading to the same file path
3. Cause interleaved writes with the still-running first process

This resulted in corrupted model files with checksums that would never match.

### Solution Implemented
Replaced the timeout-based lock file mechanism with proper advisory file locks using the `fs2` crate.

**Key Changes:**

1. **Added fs2 dependency** (`beaker/Cargo.toml`)
   - Added `fs2 = "0.4"` to dependencies

2. **Rewrote `download_with_concurrency_protection` function** (`beaker/src/model_access.rs:269-431`)
   - Uses `fs2::FileExt::lock_exclusive()` instead of create_new() flag
   - Blocks indefinitely until lock becomes available (no arbitrary timeout)
   - Downloads to temporary file first (`.tmp` extension)
   - Atomically renames temp file to final path on success
   - Lock automatically released when file handle drops
   - Cleans up temp file on failure

3. **Added comprehensive tests** (`beaker/src/model_access.rs:998-1333`)
   - `test_concurrent_downloads_with_file_locks`: Verifies 3 threads downloading simultaneously
   - `test_lock_prevents_file_corruption`: Verifies no mixed content from concurrent writes
   - `test_lock_released_on_success`: Verifies locks are properly released
   - `test_second_thread_waits_for_first`: Verifies blocking behavior

### Technical Details

**Before (Problematic Code):**
```rust
// Timeout after 5 minutes and force download
if start_time.elapsed() > MAX_WAIT_TIME {
    log::warn!("Download lock timeout ({}s), forcing download (removing stale lock)", ...);
    let _ = fs::remove_file(lock_path);  // ← BUG: Removes active lock!
    return download_model(&model_info.url, model_path);  // ← Race condition!
}
```

**After (Fixed Code):**
```rust
// Create or open the lock file
let lock_file = fs::OpenOptions::new()
    .write(true)
    .create(true)
    .truncate(false)
    .open(lock_path)?;

// Acquire exclusive lock (blocks until available, no timeout)
lock_file.lock_exclusive()?;

// Check if model was downloaded while waiting
if model_path.exists() && verify_valid() {
    return Ok(()); // Use existing file
}

// Download to temporary file
let tmp_path = model_path.with_extension("tmp");
download_model(&model_info.url, &tmp_path)?;

// Atomic rename to final path
fs::rename(&tmp_path, model_path)?;

// Lock automatically released when lock_file is dropped
```

### Benefits of New Approach

1. **No Race Conditions**: File locks ensure only one process writes at a time
2. **No Arbitrary Timeouts**: Waits indefinitely, appropriate for slow networks
3. **Automatic Cleanup**: Lock released even on process crash (OS handles it)
4. **Atomic Operations**: Temp file + rename ensures no partial files
5. **Cross-Platform**: Works on Linux, macOS, and Windows

### Test Results

All tests pass successfully:
```
test model_access::tests::test_concurrent_downloads_with_file_locks ... ok
test model_access::tests::test_lock_prevents_file_corruption ... ok
test model_access::tests::test_lock_released_on_success ... ok
test model_access::tests::test_second_thread_waits_for_first ... ok
```

### Files Modified

1. `beaker/Cargo.toml` - Added fs2 dependency
2. `beaker/src/model_access.rs` - Rewrote download_with_concurrency_protection and added tests

### Validation Criteria (All Met ✓)

- ✓ Two simultaneous downloads complete without corruption
- ✓ Checksum verification succeeds after concurrent access
- ✓ Lock is released if process completes normally
- ✓ No timeout issues with slow downloads
- ✓ Comprehensive test coverage

### Impact

- **Severity**: CRITICAL → RESOLVED
- **User Impact**: Eliminates persistent checksum failures in multi-user/CI environments
- **Performance**: Slightly improved (no polling/retry loops, just blocking wait)
- **Reliability**: Significantly improved (proper OS-level locking)

### Notes for Future

- Advisory locks work across platforms (Linux, macOS, Windows)
- On Windows, file locks are mandatory (even stronger guarantee)
- Lock files remain on disk but are harmless (contain no data)
- Consider periodic cleanup of stale .lock files in cache maintenance

### Related Issues

This fix also improves the foundation for:
- CRITICAL-02: Partial download corruption (temp file + atomic rename helps)
- CRITICAL-03: Basename collision issues (same lock mechanism can be extended)
- MEDIUM-08: Cache size limits (lock file management patterns)

---

## Next Steps

- Test in real-world multi-user scenarios
- Monitor for any file locking issues on different filesystems
- Consider adding lock file age monitoring/cleanup in cache maintenance
- Apply similar pattern to other concurrent operations if needed
