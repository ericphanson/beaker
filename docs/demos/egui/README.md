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

The tests include visual snapshot generation using `wgpu_snapshot` with lavapipe (CPU-based Vulkan software rasterizer). Real egui snapshots are provided in `tests/snapshots/`:

- `hello_world_initial.png` - Initial hello world interface
- `simple_heading.png` - Simple heading widget
- `text_input_widget.png` - Text input widget
- `button_before_click.png` / `button_after_click.png` - Button interaction states
- `multiple_widgets.png` - Multiple widget types together
- `colored_text.png` - Colored text labels (with actual colors!)

**Prerequisites:** Install lavapipe for snapshot generation:

```bash
sudo apt-get install mesa-vulkan-drivers libvulkan1
```

To generate/update snapshots:
```bash
UPDATE_SNAPSHOTS=1 cargo test
```

egui_kittest 0.30+ automatically prefers software rasterizers, enabling real visual testing in headless environments.

## CI Integration

See `.github/workflows/egui-demo-ci.yml` for the CI configuration that runs these tests automatically.

The CI validates:
- ✅ egui can build and run headlessly
- ✅ Real visual snapshots can be generated with lavapipe
- ✅ Tests run successfully in containerized environments

To enable snapshot generation in CI, add lavapipe installation:
```yaml
- name: Install lavapipe
  run: sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers libvulkan1
```
