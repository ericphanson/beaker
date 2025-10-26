# GUI Snapshot Tests

Visual regression tests for the Beaker GUI using `egui_kittest`.

## Snapshots

- `welcome_view.png` - Welcome screen with drag & drop zone
- `directory_processing.png` - Directory/bulk mode during processing
- `directory_gallery.png` - Directory/bulk mode gallery view after processing

## Generating Snapshots

### Prerequisites

Install lavapipe (CPU-based Vulkan software rasterizer) for headless rendering:

```bash
sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers libvulkan1
```

For macOS, snapshot tests are skipped due to menu rendering differences.

### Generate/Update Snapshots

From the `/beaker-gui` directory:

```bash
UPDATE_SNAPSHOTS=1 cargo test snapshot_
```

This will create PNG files in `tests/snapshots/`.

### Compress Snapshots

After generating, compress with pngquant to reduce file size:

```bash
# Install pngquant
sudo apt-get install pngquant
# or on macOS: brew install pngquant

# Compress all snapshots
pngquant --force --ext .png tests/snapshots/*.png
```

### Check In

After compressing, add the snapshots to git:

```bash
git add tests/snapshots/*.png
git commit -m "Update GUI snapshots"
```

## Running Tests

Snapshot tests run automatically but only compare if snapshots exist:

```bash
cargo test snapshot_
```

## CI Integration

To enable snapshot generation in CI, add lavapipe installation to the workflow:

```yaml
- name: Install lavapipe
  run: sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers libvulkan1
```

## Troubleshooting

**"No adapter found" error:**
- Lavapipe is not installed
- Run the installation command above

**Network issues during installation:**
- Firewall may be blocking package repositories
- Try from a different network or environment

**Snapshots differ between runs:**
- Font rendering may vary between systems
- Small pixel differences are expected
- Use visual inspection to verify changes are intentional
