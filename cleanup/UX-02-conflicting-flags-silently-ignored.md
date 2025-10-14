# UX-02: Conflicting Flags Silently Ignored

## Status
🟡 **HIGH** - Confusing user experience

## Priority
**P1** - Should fix after UX-01

## Category
User Experience / Input Validation

## Problem
`beaker cutout --alpha-matting --background-color "255,0,0,255"` accepts both flags but silently ignores `--background-color`. No warning or error.

## Root Cause
`cutout_processing.rs:216-228` - if/else chain prioritizes alpha-matting:
```rust
let cutout_result = if config.alpha_matting {
    apply_alpha_matting(...)  // Uses alpha matting
} else if let Some(bg_color) = config.background_color {
    create_cutout_with_background(...)  // ← Never reached if alpha_matting is true
```

## Impact
User sets background color expecting it to work, silently ignored, confusion.

## Solution
After fixing UX-01 (Result return type), add validation:
```rust
if cmd.alpha_matting && cmd.background_color.is_some() {
    return Err("Cannot use both --alpha-matting and --background-color".to_string());
}
```

## Dependencies
- **Requires**: UX-01 (Result return type)

## Estimated Effort
1 hour (simple validation after UX-01)

## References
**Agent Report**: CLI Routing Analysis, Bug #4
