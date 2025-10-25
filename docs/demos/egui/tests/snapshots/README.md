# Test Snapshots

This directory contains visual snapshots of egui widgets for regression testing.

## Generating Snapshots

Snapshots are generated using `egui_kittest` with wgpu rendering via lavapipe. To generate or update snapshots:

```bash
UPDATE_SNAPSHOTS=1 cargo test
```

**Prerequisites:** Install lavapipe (CPU-based Vulkan software rasterizer):

```bash
sudo apt-get install mesa-vulkan-drivers libvulkan1
```

egui_kittest 0.30+ automatically prefers software rasterizers like lavapipe, making it possible to generate real egui snapshots in headless environments without a GPU.

## Expected Snapshots

The following snapshots are generated from actual egui rendering:

- `hello_world_initial.png` - Initial hello world interface with text input
- `simple_heading.png` - Simple heading widget
- `text_input_widget.png` - Text input widget with label
- `button_before_click.png` - Button widget
- `button_after_click.png` - Button after click interaction
- `multiple_widgets.png` - Multiple widget types (heading, separator, labels, buttons, checkbox)
- `colored_text.png` - Colored text labels (red, green, blue)

## CI Integration

To enable snapshot generation in CI environments, install lavapipe:

```yaml
- name: Install lavapipe for headless rendering
  run: sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers libvulkan1

- name: Run tests with snapshot generation
  run: UPDATE_SNAPSHOTS=1 cargo test
```

This allows snapshot tests to run successfully in GitHub Actions and other CI environments without dedicated GPUs.
