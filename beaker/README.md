# Beaker

The Beaker bird head detection CLI tool.

## Goals

- Self-contained
- Cross-platform
- use CoreML on Apple Silicon

## Usage

```bash
# Build from source (automatically downloads latest model)
cargo build --release

# Run head detection - no environment variables needed!
./target/release/beaker head ../example.jpg --confidence 0.75

./target/release/beaker -v head ../example.jpg --confidence 0.75

# Show help
./target/release/beaker --help

# Show version information
./target/release/beaker version

# Run with crop and bounding box outputs
./target/release/beaker head ../example.jpg --crop --bounding-box

# Run with custom confidence and IoU thresholds
./target/release/beaker head ../example.jpg --confidence 0.5 --iou-threshold 0.4

# Use global output directory
./target/release/beaker --output-dir ./results head ../example.jpg --crop

# Run on CPU explicitly
./target/release/beaker head ../example.jpg --device cpu
```

## Building

```bash
# Build debug version
cargo build

# Build optimized release version
cargo build --release
```

The build process automatically:
1. Downloads the latest ONNX head model from GitHub releases
2. Embeds it into the binary as bytes
3. Downloads and bundles ONNX Runtime libraries with proper rpath
4. Creates a fully self-contained executable
5. Detects platform capabilities (CoreML on macOS, CPU elsewhere)

## `beaker head`

When running detection, `beaker head` can generate several output files:

- **`--crop`**: Individual square crops for each detected bird head (`image_crop-1.jpg`, `image_crop-2.jpg`, etc.)
  - Single detection: `image_crop.jpg` (no suffix)
  - Multiple detections: Numbered with zero-padding for 10+ detections
- **`--bounding-box`**: Single image showing all detections with green boxes (`image_bounding-box.jpg`)
- **`--metadata`**: Structured TOML output with detection coordinates, confidence scores, and relative file paths (`image.beaker.toml`)

## Performance Benchmarks

To run the benchmarks, run

```sh
cargo build --release
python3 benchmark.py
```

after installing `rembg`. The full results on a M1 macbook pro are in [benchmark_results.json](./benchmark_results.json) and are summarized below.

### Single Image Processing

| Task | Device | Load Time (ms) | Inference Time (ms) | Total Time (ms) |
|------|--------|----------------|-------------------|-----------------|
| **Head Detection** | CPU | 40 | 57 | 136 |
| **Head Detection** | CoreML | 156 | 25 | 226 |
| **Background Removal** | CPU | 88-117 | 1441-1687 | 1916-2200 |
| **Background Removal** | CoreML | 501-540 | 651-691 | 1654-1754 |
| **rembg** | CPU | - | - | 3540-3651 |

### Batch Processing (per image, 10x batches)

| Task | Device | Load Time (ms) | Inference Time (ms) | Total Time (ms/img) |
|------|--------|----------------|-------------------|-------------------|
| **Head Detection** | CPU | 23-37 | 38-39 | 63-72 |
| **Head Detection** | CoreML | 152-155 | 14-16 | 53-59 |
| **Background Removal** | CPU | 87 | 1491-1498 | 1540-1545 |
| **Background Removal** | CoreML | 515-540 | 4863-5022 | 590-602 |
| **rembg** | CPU | - | - | 1620-1710 |

### Notes

- Small head model is 4x slower to load with CoreML (156ms vs 40ms) but 2.3x faster (25ms vs 57ms). Worth it for batches, not single images
- Larger `isnet-general-use` model for background removal is worth loading with CoreML even for single images. And CoreML provides 2-3x speedup for batches.
- rembg here is only configured with ONNX on CPU. It has some overhead relative to beaker but that overhead is amortized over batches, so it comes out to the approximately the same time as beaker on CPU in the batch case.

## Environment Variables

The following environment variables can be used to customize model behavior:

### Head Detection Model
- `BEAKER_HEAD_MODEL_PATH`: Override the embedded head detection model with a custom ONNX model file

### Cutout Model
- `BEAKER_CUTOUT_MODEL_PATH`: Override the default cutout model with a custom ONNX model file
- `BEAKER_CUTOUT_MODEL_URL`: Use a custom download URL for the cutout model
- `BEAKER_CUTOUT_MODEL_CHECKSUM`: Specify custom MD5 checksum for the cutout model

### General Configuration
- `ONNX_MODEL_CACHE_DIR`: Set custom cache directory for downloaded models (default: system cache directory)
- `RUST_LOG`: Control log level (e.g., `RUST_LOG=debug` for detailed logging)

### Example Usage

```bash
# Use custom head detection model
BEAKER_HEAD_MODEL_PATH=/path/to/my/head_model.onnx ./target/release/beaker head example.jpg

# Use custom cutout model from local file
BEAKER_CUTOUT_MODEL_PATH=/path/to/my/cutout_model.onnx ./target/release/beaker cutout example.jpg

# Use custom cutout model from alternative URL with custom checksum
BEAKER_CUTOUT_MODEL_URL=https://example.com/my_cutout_model.onnx \
BEAKER_CUTOUT_MODEL_CHECKSUM=abcd1234567890abcd1234567890abcd \
./target/release/beaker cutout example.jpg

# Set custom cache directory
ONNX_MODEL_CACHE_DIR=~/.my_cache ./target/release/beaker cutout example.jpg

# Enable debug logging to see model loading details
RUST_LOG=debug ./target/release/beaker head example.jpg --crop
```
