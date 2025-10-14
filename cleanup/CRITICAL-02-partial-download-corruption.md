# CRITICAL-02: Partial Downloads Leave Corrupt Files

## Status
🔴 **CRITICAL** - Data integrity issue

## Priority
**P0** - Must fix before next release

## Category
Data Integrity / Error Handling

## Affected Components
- `beaker/src/model_access.rs` (lines 112-266, specifically 177-206)
- Model cache system

## Problem Description

### What's Broken
If a model download is interrupted (network failure, CTRL+C, process kill), the partial file remains in the cache with an invalid checksum. On next run, beaker detects the checksum mismatch and re-downloads from scratch—no resume capability.

### Root Cause
Downloads write directly to the final destination path instead of using atomic writes (temp file + rename on success).

**Current Flow** (`model_access.rs:177-206`):
```rust
// Line 177: Creates final file immediately
let mut file = fs::File::create(output_path)?;

// Lines 189-206: Streams data directly to final path
loop {
    let bytes_read = response.read(&mut buffer)?;
    if bytes_read == 0 { break; }
    file.write_all(&buffer[..bytes_read])?;  // ← If interrupted here, partial file remains
    downloaded += bytes_read as u64;
}
```

**Why it fails**:
1. User starts download of 100MB model
2. Network fails at 50MB
3. Process exits with error
4. 50MB partial file remains at final path
5. Next run: checksum verification fails (lines 497-519)
6. Beaker deletes partial file (line 573)
7. Download starts from 0% again

### Evidence (Code References)

**Size mismatch handling** (`model_access.rs:234-240`):
```rust
if downloaded != expected_length {
    log::warn!("Size mismatch: expected {expected_length} bytes, got {downloaded} bytes");
    // ⚠️ WARNING ONLY - doesn't fail or clean up!
}
```

**Checksum failure cleanup** (`model_access.rs:573-601`):
```rust
if actual_checksum != checksum {
    fs::remove_file(&model_path)?;  // ← Deletes partial file
    return Err(anyhow!(
        "Downloaded model failed checksum verification.\n\
         Expected checksum: {}\n\
         Actual checksum:   {}\n\
         // ... tries again from scratch
```

## Impact Assessment

### User Impact
- **Severity**: HIGH - Wastes bandwidth and time
- **Frequency**: Common on unstable networks, mobile connections, CI timeouts
- **Symptoms**:
  - Downloads that "never finish" (keep restarting from 0%)
  - High bandwidth usage (100MB+ per attempt)
  - Slow first runs on new machines

### System Impact
- Wasted network bandwidth (no resume capability)
- Slow CI pipelines (re-download on timeout)
- Battery drain on laptops (repeated large downloads)

### Data Loss Risk
- **Model files**: Can be re-downloaded (annoying but not data loss)
- **User data**: Not affected

## Reproduction Steps

### Minimal Reproduction
```bash
# Start download
beaker detect image.jpg &
BEAKER_PID=$!

# Wait for download to start (check with du)
sleep 5
du -h ~/.cache/onnx-models/*.onnx

# Kill process mid-download
kill -9 $BEAKER_PID

# Check partial file exists
ls -lh ~/.cache/onnx-models/*.onnx
# File is ~50MB (should be 100MB)

# Try to run again - will delete partial file and restart from 0%
beaker detect image.jpg
```

### Verification
```bash
# Monitor download progress
watch -n 1 'du -h ~/.cache/onnx-models/*.onnx'

# Interrupt at 50%
# Verify file is deleted and download restarts from 0%
```

## Proposed Solution

### Approach
Use atomic writes: download to temporary file, verify checksum, then rename to final path.

### Implementation Plan

**Atomic Download Pattern**:
```rust
pub fn download_model(url: &str, model_path: &Path) -> Result<PathBuf> {
    // Download to temporary file with unique name
    let tmp_path = model_path.with_extension("tmp");

    // Clean up any stale .tmp file from previous failed attempt
    if tmp_path.exists() {
        log::debug!("Removing stale temporary file: {}", tmp_path.display());
        fs::remove_file(&tmp_path)?;
    }

    log::info!("Downloading model to temporary file: {}", tmp_path.display());

    // Download to temp file
    let mut file = fs::File::create(&tmp_path)?;
    let mut response = reqwest::blocking::get(url)?;
    let expected_length = response.content_length().unwrap_or(0);

    let mut downloaded = 0u64;
    let mut buffer = vec![0u8; 8192];

    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 { break; }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        // Optional: log progress
        if downloaded % (10 * 1024 * 1024) == 0 {
            log::debug!("Downloaded: {} MB", downloaded / 1_048_576);
        }
    }

    // Verify size before checksum
    if expected_length > 0 && downloaded != expected_length {
        fs::remove_file(&tmp_path)?;
        return Err(anyhow!(
            "Download incomplete: expected {} bytes, got {} bytes",
            expected_length, downloaded
        ));
    }

    // Verify checksum on temp file (not final path)
    let actual_checksum = cache_common::calculate_md5(&tmp_path)?;
    if actual_checksum != model_info.md5_checksum {
        fs::remove_file(&tmp_path)?;
        return Err(anyhow!(
            "Downloaded model failed checksum verification.\n\
             Expected: {}\n\
             Actual:   {}",
            model_info.md5_checksum, actual_checksum
        ));
    }

    // Atomic rename: only if download + verification succeeded
    fs::rename(&tmp_path, model_path)?;
    log::info!("Model successfully downloaded and verified: {}", model_path.display());

    Ok(model_path.to_path_buf())
}
```

### Files to Modify
1. `beaker/src/model_access.rs`:
   - Modify `download_model()` function (lines 112-266)
   - Update `download_with_lock()` to pass tmp_path pattern
   - Add cleanup for stale `.tmp` files on startup

### Testing Requirements
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_partial_download_cleanup() {
        // Create fake .tmp file
        // Call download_model
        // Verify .tmp is cleaned up before new download
    }

    #[test]
    fn test_atomic_rename_on_success() {
        // Mock successful download
        // Verify .tmp file created
        // Verify .tmp renamed to final path
        // Verify no .tmp remains
    }

    #[test]
    fn test_tmp_removed_on_checksum_failure() {
        // Mock download with wrong checksum
        // Verify .tmp is deleted
        // Verify final path not created
    }

    #[test]
    fn test_interrupted_download() {
        // Simulate download interruption
        // Verify .tmp remains (not final path)
        // Verify next run cleans up .tmp and restarts
    }
}
```

### Validation Criteria
- [ ] Interrupted download leaves `.tmp` file (not final path)
- [ ] Next run cleans up `.tmp` and starts fresh
- [ ] Successful download creates final path atomically
- [ ] Checksum failure does not create final path
- [ ] Size mismatch does not create final path
- [ ] No `.tmp` files remain after successful completion

## Dependencies
- **Blockers**: None
- **Related**: CRITICAL-01 (race condition - should implement both together)
- **Complements**: Future enhancement: resumable downloads (HTTP Range requests)

## Decisions Required

### DECISION-1: Resume capability?
**Options**:
- **A**: Simple atomic writes (this issue) - Download always starts from 0%
- **B**: Add resume support - Use HTTP Range headers to continue from last byte

**Recommendation**: Option A for this issue (simple atomic writes)
- Easier to implement and test
- Solves the corruption problem
- Resume can be added later as ENHANCEMENT issue

**Decision needed from**: @maintainer

### DECISION-2: Temp file naming?
**Options**:
- **A**: `model.onnx.tmp` - Simple extension swap
- **B**: `model.onnx.{pid}.tmp` - Unique per process (supports concurrent downloads)
- **C**: `model.onnx.{timestamp}.tmp` - Unique per attempt

**Recommendation**: Option A (simple `.tmp` extension)
- Simpler code
- CRITICAL-01 fix handles concurrency via locking
- Stale `.tmp` cleanup is straightforward

**Decision needed from**: @maintainer

## Estimated Effort
- **Investigation**: ✅ Complete
- **Implementation**: 3-4 hours
  - Rewrite download_model(): 1-2 hours
  - Add stale tmp cleanup: 30 minutes
  - Add tests: 1-2 hours
  - Manual testing: 30 minutes
- **Review**: 1 hour
- **Total**: ~0.5 day

## Success Metrics
- Interrupted downloads never corrupt final model file
- Next run after interruption starts fresh (expected behavior)
- No `.tmp` files accumulate in cache directory
- Checksum verification always passes for completed downloads

## Rollback Plan
If atomic writes cause issues:
1. Revert to direct writes
2. Add better cleanup of partial files on startup
3. Document that downloads cannot be interrupted safely

## Future Enhancements
After this fix is stable, consider:
- **ENHANCEMENT**: Resumable downloads using HTTP Range headers
- **ENHANCEMENT**: Progress bar for downloads
- **ENHANCEMENT**: Parallel chunk downloads for large models

## References
- **Agent Report**: Model Management Analysis, Bug #1
- **Related Issues**:
  - CRITICAL-01: Download race condition
  - MEDIUM-06: Redundant checksum verification

## Notes for Implementer
- Ensure `.tmp` files are cleaned up in ALL error paths
- Test with `SIGKILL` (simulates crash - no cleanup opportunity)
- Test with `SIGTERM` (simulates graceful shutdown - cleanup should happen)
- Consider adding `.tmp` to `.gitignore` if models are ever committed
- Atomic `fs::rename()` works across filesystems on Unix, might fail on some Windows configurations (document limitation)
