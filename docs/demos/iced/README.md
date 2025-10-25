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

### All Tests
```bash
cargo test
```

## Headless Rendering

This demo uses iced's `tiny-skia` software renderer for headless operation:

1. **Software Rendering**: The `tiny-skia` feature enables CPU-only rendering
2. **No GPU Required**: Works in CI environments without graphics hardware
3. **Environment Control**: Set `ICED_BACKEND=tiny-skia` to force software rendering

## CI Integration

See `.github/workflows/iced-demo-ci.yml` for the CI pipeline that:
- Runs all tests in a headless environment
- Validates formatting and linting
- Builds both debug and release binaries

## Architecture

This demo is isolated from the main beaker workspace:
- Separate `Cargo.toml` with empty `[workspace]` section
- Independent dependency tree
- Can be developed and tested independently

## Dependencies

- `iced` v0.13 with `tiny-skia` and `advanced` features
- `tokio` (dev) for async test support
- `iced_tiny_skia` (dev) for direct renderer access

## Limitations

The `iced::advanced::renderer::Headless` trait provides APIs for screenshot testing, but practical usage requires additional research. This demo focuses on validating that:
- The sandbox environment can run iced applications
- Tests execute successfully in headless mode
- CI can build and test iced applications

For full screenshot testing capabilities, consider:
- Exploring the `iced_test` crate when it becomes available
- Direct integration with `iced_tiny_skia::Renderer`
- Community examples from the iced ecosystem
