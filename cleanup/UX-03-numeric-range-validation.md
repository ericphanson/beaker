# UX-03: Missing Numeric Range Validation

## Status
🟡 **HIGH** - Can cause crashes or undefined behavior

## Priority
**P1** - Should fix with UX-01

## Category
Input Validation / Safety

## Problem
Users can pass invalid values: `--confidence 2.0`, `--confidence -0.5`, `--iou-threshold 5.0`

No validation at CLI parse time. Values outside 0.0-1.0 range are mathematically nonsensical for probabilities.

## Root Cause
`config.rs:154-160` - No `value_parser` attribute:
```rust
#[arg(short, long, default_value = "0.5")]
pub confidence: f32,  // ← No range validation!
```

## Impact
- Invalid values cause undefined behavior in model inference
- Confusing error messages (if any)
- No guidance at parse time

## Solution
Add value parser:
```rust
fn parse_probability(s: &str) -> Result<f32, String> {
    let val = s.parse::<f32>()
        .map_err(|_| format!("Invalid number: '{s}'"))?;
    if !(0.0..=1.0).contains(&val) {
        return Err(format!("Must be between 0.0 and 1.0, got {val}"));
    }
    Ok(val)
}

#[arg(short, long, default_value = "0.5", value_parser = parse_probability)]
pub confidence: f32,
```

## Estimated Effort
2 hours

## References
**Agent Report**: CLI Routing Analysis, Bug #2
