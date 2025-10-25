# Beaker GUI

A graphical user interface for the Beaker bird image analysis toolkit, built with egui.

## Features

- **Detection View**: Visualize bird detections with bounding boxes
- **High-DPI Support**: Optimized for retina displays with 2x pixel scaling
- **Native macOS Menu**: Full macOS menu bar integration
- **CLI Integration**: Launch with specific images via command-line flags
- **Runtime Asserts**: Comprehensive validation for reliability

## Building

```bash
cargo build --release
```

## Running

### Launch with an image

```bash
cargo run --release -- --image path/to/bird.jpg
```

### Launch without an image

```bash
cargo run --release
```

Then use File > Open to load an image (coming soon).

### Specify view

```bash
cargo run --release -- --image path/to/bird.jpg --view detection
```

## Testing

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test --verbose
```

## Architecture

### Structure

- `src/main.rs` - Application entry point with CLI argument parsing
- `src/app.rs` - Main app state and eframe::App implementation
- `src/style.rs` - Theme and styling configuration
- `src/views/` - View implementations (detection, etc.)
- `src/lib.rs` - Library exports for testing

### Views

The GUI uses a trait-based view system. Each view implements the `View` trait:

```rust
pub trait View {
    fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui);
    fn name(&self) -> &str;
}
```

Currently implemented views:
- **DetectionView**: Displays images with bird detection bounding boxes

### Testing Strategy

The MVP uses a two-layer validation approach:

1. **Runtime Asserts**: Strategic `assert!` calls in view code validate invariants during operation and testing
2. **Integration Tests**: Exercise GUI views and verify runtime asserts don't panic

Runtime asserts validate:
- Detection results are not empty if they exist
- Selected detections are in bounds
- Confidence scores are in [0.0, 1.0]
- Bounding boxes are within image dimensions
- Image dimensions are non-zero
- Rendered textures match original image sizes

## Platform Notes

### macOS

The app uses native macOS menu bars via the `muda` crate. To use egui's built-in menu instead:

```bash
USE_EGUI_MENU=1 cargo run
```

### Linux

Requires Vulkan for GPU acceleration:

```bash
sudo apt-get install mesa-vulkan-drivers libvulkan1
```

## Development

### Adding a New View

1. Create a new file in `src/views/` (e.g., `segmentation.rs`)
2. Implement the `View` trait
3. Add to `src/views/mod.rs`
4. Update CLI args in `src/main.rs` to allow selection
5. Add tests in `tests/gui_tests.rs`

### Style Customization

Edit `src/style.rs` to customize:
- Color scheme
- Spacing and padding
- Border radius
- Shadows
- DPI scaling

## CI/CD

The project includes a GitHub Actions workflow (`.github/workflows/beaker-gui-ci.yml`) that:
- Builds on Ubuntu and macOS
- Runs all tests
- Checks formatting with `cargo fmt`
- Runs `cargo clippy` for linting

## License

See the main Beaker project for license information.
