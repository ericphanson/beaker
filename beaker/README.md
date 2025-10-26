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
./target/release/beaker head ../example.jpg --metadata --device cpu
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
  - Single detection: `image_crop.jpg`
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

## Quality Assessment

Beaker includes a no-reference image quality assessment model (PaQ-2-PiQ) combined with blur detection.

### Basic Usage

```bash
# Assess single image
beaker quality image.jpg

# Assess multiple images
beaker quality *.jpg

# Write metadata to sidecar files
beaker quality --metadata image.jpg
```

### Parameter Tuning

Quality assessment uses several tunable parameters for blur detection:

```bash
# Adjust blur sensitivity (lower = more sensitive)
beaker quality --tau 0.01 image.jpg

# Adjust blur weight impact (higher = more penalty for blur)
beaker quality --alpha 0.8 image.jpg

# Adjust probability curve steepness
beaker quality --beta 1.5 image.jpg
```

**Parameter Reference:**

- `--alpha` (0.0-1.0, default 0.7): Weight coefficient - how much blur reduces quality score
- `--beta` (0.5-2.0, default 1.2): Probability curve exponent - steeper = more aggressive blur detection
- `--tau` (0.001-0.1, default 0.02): Tenengrad threshold - lower = more sensitive to blur

### Programmatic API

```rust
use beaker::quality_processing::{compute_quality_raw, load_onnx_session_default};
use beaker::quality_types::{QualityParams, QualityScores};

// Load model once
let session = load_onnx_session_default()?;

// Compute raw data (expensive, cached automatically)
let raw = compute_quality_raw("image.jpg", &session)?;

// Compute scores with default parameters
let params = QualityParams::default();
let scores = QualityScores::compute(&raw, &params);

println!("Quality: {:.1}", scores.final_score);

// Adjust parameters and recompute instantly
let strict_params = QualityParams {
    tau_ten_224: 0.01,
    ..Default::default()
};
let strict_scores = QualityScores::compute(&raw, &strict_params);
```

### Performance

- First run: ~60ms per image (preprocessing + ONNX inference + blur detection)
- Cached run: <1ms per image (cache hit for raw computation)
- Parameter adjustment: <0.1ms per image (recomputes scores from cached raw data)

This makes real-time parameter tuning feasible for GUI applications.
