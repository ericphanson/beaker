# FEATURE-03: Quality Debug Images Always Created (Empty Directories)

## Status
🟡 **HIGH** - Directory pollution

## Priority
**P1** - Easy fix

## Category
Output Management / UX

## Problem
`quality_debug_images_{stem}/` directory created on EVERY quality run, even without `-vv` debug flag. Directory is created empty (images only written if debug logging enabled).

## Root Cause
`quality_processing.rs:146-151`:
```rust
let output_dir = output_manager
    .get_output_dir()?
    .join(format!("quality_debug_images_{input_stem}"));

let (w20, p20, _, global_blur_score) =
    crate::blur_detection::blur_weights_from_nchw(&input_array, Some(output_dir));
    // ☝️ Always passes Some(output_dir)
```

`blur_detection.rs:231`:
```rust
fs::create_dir_all(out_dir).ok();  // Creates directory
// But line 473 only writes if log::log_enabled!(Debug)
```

## Impact
- Empty directories accumulate
- User confusion ("what are these empty folders?")
- Directory pollution in output locations

## Solution
Only pass output directory when debug logging enabled:
```rust
let output_dir = if log::log_enabled!(log::Level::Debug) {
    Some(output_manager.get_output_dir()?.join(...))
} else {
    None
};
```

## Estimated Effort
1 hour (simple conditional)

## References
**Agent Report**: Output Management Analysis, Bug #3
