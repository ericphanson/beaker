# Test Snapshots

This directory contains visual snapshots of egui widgets for regression testing.

## Generating Snapshots

Snapshots are generated using `egui_kittest` with wgpu rendering. To generate or update snapshots:

```bash
UPDATE_SNAPSHOTS=true cargo test
```

**Note:** Snapshot generation requires a GPU or software rasterizer. In headless CI environments without GPU access, snapshot tests will be skipped but other tests will continue to run.

## Expected Snapshots

When generated on a system with GPU access, the following snapshots should be created:

- `hello_world_initial.png` - Initial hello world interface
- `simple_heading.png` - Simple heading widget
- `text_input_widget.png` - Text input widget
- `button_before_click.png` - Button before interaction
- `button_after_click.png` - Button after click (may show hover state)
- `multiple_widgets.png` - Multiple widget types together
- `colored_text.png` - Colored text labels

## CI Behavior

In CI environments without GPU access, snapshot tests gracefully skip snapshot generation while still validating that:
- egui widgets render without errors
- Widget queries work correctly
- Interactions function properly
