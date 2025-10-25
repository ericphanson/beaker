# egui Hello World Demo

A minimal egui application demonstrating basic GUI functionality with headless testing using egui_kittest.

## Purpose

This demo validates that:
1. egui can run in various environments (including sandboxes)
2. Headless tests work correctly with egui_kittest
3. CI can successfully run egui tests without a display

## Building and Running

### Run the application

```bash
cd docs/demos/egui
cargo run
```

This will open a window with a simple "Hello, World!" interface where you can enter your name.

### Run tests

```bash
cargo test
```

The tests run headlessly and verify:
- UI widgets render correctly
- Multiple widgets can be tested together
- Widget interaction works properly
- The Harness API functions correctly

## Project Structure

- `src/main.rs` - The main egui application
- `tests/kittest_integration.rs` - Integration tests using egui_kittest
- `Cargo.toml` - Project configuration (independent workspace)

## Dependencies

- `eframe` - The egui application framework
- `egui` - The immediate mode GUI library
- `egui_kittest` - Testing framework for egui (dev dependency)

## Testing Approach

This demo uses `egui_kittest::Harness` for headless testing. The tests:
- Create UI elements programmatically
- Verify widgets exist and can be queried
- Test interactions like button clicks
- Run without requiring a display server
- Generate visual regression snapshots (when GPU is available)

### Snapshot Testing

The tests include visual snapshot generation using `wgpu_snapshot`. Example snapshots are provided in `tests/snapshots/`:

- `hello_world_initial.png` - Initial hello world interface
- `simple_heading.png` - Simple heading widget
- `text_input_widget.png` - Text input widget
- `button_before_click.png` / `button_after_click.png` - Button interaction states
- `multiple_widgets.png` - Multiple widget types together
- `colored_text.png` - Colored text labels

**Note:** Snapshot generation requires a GPU or software rasterizer. In environments without GPU access (like some sandboxes or CI runners), snapshot tests gracefully skip snapshot generation while still validating widget functionality.

To generate/update snapshots on a GPU-enabled system:
```bash
UPDATE_SNAPSHOTS=true cargo test
```

## CI Integration

See `.github/workflows/egui-demo-ci.yml` for the CI configuration that runs these tests automatically.

The CI validates:
- ✅ egui can build and run headlessly
- ✅ Widget tests pass without GPU
- ✅ Tests run successfully in containerized environments
