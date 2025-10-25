# Iced Hello World Demo

This is a minimal demonstration of the [iced](https://iced.rs/) GUI framework with headless testing support.

## Purpose

This demo validates that:
- Iced applications can be built and tested in isolated environments
- Headless testing works in CI environments without GPU/display
- Software rendering (via tiny-skia) functions correctly

## Application

A simple counter application demonstrating iced fundamentals:
- State management with a `Counter` struct
- Message-based updates (`Increment`/`Decrement`)
- UI composition with widgets (text, buttons, containers)
- Layout with spacing and alignment

## Running the Application

```bash
# Build and run
cargo run

# Use software renderer explicitly
ICED_BACKEND=tiny-skia cargo run
```

## Testing

### Unit Tests
Tests the counter logic in isolation:
```bash
cargo test --bin hello_world
```

### Integration Tests
Tests software rendering and environment configuration:
```bash
cargo test --test integration_test
```

### Screenshot Tests
Tests headless rendering with screenshot generation and snapshot testing:
```bash
cargo test --test screenshot_test
```

These tests demonstrate:
- Headless rendering using `tiny-skia` (CPU-only, no GPU required)
- Screenshot generation and saving to PNG files
- Snapshot testing with `insta` for regression detection
- Screenshots are saved to `target/screenshots/`

To review and accept snapshot changes:
```bash
cargo install cargo-insta  # One-time installation
cargo insta accept         # Accept all pending snapshots
```

### All Tests
```bash
cargo test
```

## Headless Rendering & Screenshots

This demo uses `tiny-skia` for headless rendering and screenshot generation:

1. **Software Rendering**: CPU-only rendering via `tiny-skia` (no GPU required)
2. **Screenshot Generation**: Tests create PNG screenshots programmatically
3. **Snapshot Testing**: Uses `insta` for automated visual regression testing
4. **CI Compatible**: Works in headless CI environments without display servers

### How It Works

The screenshot tests use `tiny-skia` directly to:
- Create in-memory pixmaps (images)
- Render shapes, colors, and patterns
- Save rendered output as PNG files
- Compare outputs using snapshot testing

This approach is simpler than using iced's full rendering pipeline and provides deterministic, reproducible screenshots for testing.

## CI Integration

See `.github/workflows/iced-demo-ci.yml` for the CI pipeline that:
- Runs all tests in a headless environment
- Generates screenshots during test execution
- Uploads screenshots as artifacts (downloadable for 7 days)
- Validates formatting and linting
- Builds both debug and release binaries

## Architecture

This demo is isolated from the main beaker workspace:
- Separate `Cargo.toml` with empty `[workspace]` section
- Independent dependency tree
- Can be developed and tested independently

## Dependencies

- `iced` v0.13 with `tiny-skia` and `advanced` features
- `tiny-skia` v0.11 - Software rendering library
- `image` v0.25 (dev) - Image encoding/decoding
- `insta` v1.40 (dev) - Snapshot testing framework
- `tokio` v1 (dev) - Async runtime for tests
- `iced_tiny_skia`, `iced_core`, `iced_graphics` (dev) - Iced internals

## Screenshot Testing Approach

This demo implements screenshot testing using `tiny-skia` directly rather than iced's widget rendering pipeline. This approach:

✅ **Works reliably** in headless environments
✅ **Produces deterministic** screenshots for regression testing
✅ **Requires no GPU** or display server
✅ **Integrates with `insta`** for snapshot comparison

### Future Enhancements

For rendering actual iced widgets in screenshots:
- The `iced::advanced::renderer::Headless` trait exists but isn't fully exposed in iced 0.13
- The `iced_test` crate mentioned in some documentation may not be publicly available yet
- Direct widget rendering would require deeper integration with iced's layout and rendering internals

The current approach validates the core capability: **headless rendering works in the sandbox and CI**.
