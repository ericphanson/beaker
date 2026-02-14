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

## Detection Metadata Schema

This section documents the detection-related fields written to `image.beaker.toml` for `beaker detect`.

### Detection record (`[[detect.detections]]`)

Each detection has basic geometry and confidence fields:

- `class_name`: Predicted class (`bird`, `head`, `eye`, `beak`)
- `class_id`: Numeric class id from the detection model
- `confidence`: Detection confidence (`0.0..1.0`)
- `x1`, `y1`, `x2`, `y2`: Bounding box corners in original image pixel coordinates
- `angle_radians`: Predicted orientation angle for the detection

### Baseline per-detection quality (`[detect.detections.quality]`)

This block is derived from the full-image quality pass (`beaker quality` internals) by pooling metrics over each detection ROI.

- `roi_quality_mean`: This is the average PaQ-2-PiQ local quality value over the detection box (`0..100`), where higher is better. Source: **PaQ-2-PiQ model output** (local grid) with ROI averaging.
- `roi_blur_probability_mean`: This is the average (over the detection box) of a multi-scale blur-likelihood map, where each value is a `0..1` probability of blur fused from 224- and 112-scale sharpness cues. Source: **direct numerical calculation** (Tenengrad-derived probabilities + probabilistic fusion + ROI averaging).
- `roi_blur_weight_mean`: This is the average blur weight over the detection box, where weight is computed as `W = 1 - alpha * P` and higher values mean less blur penalty. Source: **direct numerical calculation** from fused blur probability.
- `roi_detail_probability`: This is a native-resolution (`0..1`) detail/sharpness probability computed directly from image gradients in the detection region. Source: **direct numerical heuristic**.
- `core_ring_sharpness_ratio`: This is the ratio of core sharpness to ring sharpness (`core / ring`), used to compare subject sharpness against surrounding context. Source: **direct numerical calculation** (Tenengrad on core/ring regions).
- `tenengrad_core_mean`: This is the mean Tenengrad sharpness measured in the inner core of the detection box. Source: **direct numerical calculation**.
- `tenengrad_ring_mean`: This is the mean Tenengrad sharpness measured in the outer ring of the detection box. Source: **direct numerical calculation**.
- `grid_cells_covered`: This is the approximate number of quality-grid cells intersected by the detection box. Source: **direct geometric calculation**.
- `grid_coverage_prior`: This is a normalized (`0..1`) prior computed from `grid_cells_covered` to represent how much of the quality grid the detection spans. Source: **direct numerical calculation**.
- `size_prior_factor`: This is a normalized (`0..1`) prior computed from detection box size to down-weight very small detections. Source: **direct numerical calculation**.
- `triage_decision`: This is the final heuristic triage label (`bad`, `unknown`, `good`) derived from threshold rules. Source: **direct rule-based decision**.
- `triage_rationale`: This is a human-readable explanation of the triage rule that fired and the relevant threshold comparison. Source: **direct string formatting from rule evaluation**.

### Refined per-detection quality (`[detect.detections.quality_refined]`)

This optional block appears only when `--refine-detection-quality` is enabled and refinement succeeds for a detection.

Refinement means Beaker does a second quality pass focused on each selected detection, instead of relying only on ROI pooling from a full-image quality map.

In detail, refinement works as follows:

1. Beaker runs normal detection and gets boxes in original-image coordinates.
2. Beaker keeps only `bird` and `head` detections that are at least `64x64` pixels.
3. Beaker selects up to `--refine-detection-max-per-image` detections (default `8`) by confidence.
4. For each selected detection, Beaker crops a padded region from the original image (`--refine-detection-padding`, default `0.25`).
5. Beaker re-runs the quality pipeline on that crop (PaQ-2-PiQ + blur calculations), then recomputes per-detection quality metrics in crop coordinates.
6. The refined metrics are written under `quality_refined`, while baseline `quality` is preserved unchanged.

This makes `quality_refined` more object-centric, because it evaluates quality on a crop centered on the detection with controlled context.

Refinement behavior:

- Only `bird` and `head` detections are considered.
- Detections smaller than `64x64` pixels are skipped.
- Up to `--refine-detection-max-per-image` detections are refined (default `8`), chosen by confidence.
- Refinement runs on a padded crop around each detection (`--refine-detection-padding`, default `0.25`).

`quality_refined` includes all fields listed in `quality`, recomputed on the refined crop context, plus crop-level scalar quality outputs:

- `crop_paq2piq_global`: This is the refined crop's global PaQ-2-PiQ score (`0..100`). Source: **PaQ-2-PiQ model output** (global scalar).
- `crop_blur_score`: This is the refined crop's global blur score (`0..1`), computed as the mean fused blur probability over the full crop. Source: **direct numerical calculation**.
- `crop_final_quality_score`: This is the refined crop's final fused quality score (`0..100`) after applying blur-based weighting to the PaQ-2-PiQ score. Source: **hybrid** (PaQ-2-PiQ model output + direct numerical fusion).

Notes:

- `quality` is preserved as the baseline full-image-derived result.
- `quality_refined` is additive and non-breaking; it is omitted when refinement is disabled or skipped.
- `quality_refined` is also omitted for detections that fail refinement (for example, invalid crop geometry or inference failure), while baseline `quality` remains available.

## Full Sidecar Field Reference (`example.beaker.toml`)

This reference documents every field currently present in `/Users/eph/beaker/example.beaker.toml`.

The metadata sidecar can include multiple top-level tool sections (`[detect]`, `[cutout]`, `[quality]`) when multiple tools have been run for the same image over time.

### Detect section (`[detect]`)

- `model_version`: The detection model version string used for inference.
- `input_img_width`: Width of the original input image in pixels.
- `input_img_height`: Height of the original input image in pixels.
- `[[detect.detections]]`: One table per detection.
- `detect.detections.angle_radians`: Predicted orientation angle for that detection.
- `detect.detections.class_id`: Numeric class id from the model output.
- `detect.detections.class_name`: Predicted class label.
- `detect.detections.confidence`: Detection confidence score (`0.0..1.0`).
- `detect.detections.x1`, `y1`, `x2`, `y2`: Bounding-box corners in original-image pixel coordinates.

Per-detection quality blocks are covered in the earlier schema section:

- `[detect.detections.quality]` (baseline)
- `[detect.detections.quality_refined]` (optional refinement)

### Detect config (`[detect.config]`)

- `bounding_box`: Whether bounding-box image generation was enabled.
- `confidence`: Detection confidence threshold requested by CLI.
- `crop_classes`: Classes selected for crop output generation.
- `refine_detection_quality`: Whether refinement mode was enabled.
- `refine_detection_padding`: Padding fraction used for refined crops.
- `refine_detection_max_per_image`: Max detections refined per image.

### Detect execution (`[detect.execution]`)

- `timestamp`: UTC timestamp when this tool invocation started.
- `beaker_version`: Beaker version that generated this metadata.
- `command_line`: Full command-line arguments used.
- `exit_code`: Process exit status for this tool run.
- `model_processing_time_ms`: Total model processing time in milliseconds.

#### Detect file I/O (`[detect.execution.file_io]`)

- `read_time_ms`: Cumulative image-read time in milliseconds for this image.
- `write_time_ms` (optional): Cumulative write time in milliseconds (only present when writes are timed in this tool path).

### Detect system (`[detect.system]`)

- `device_requested`: Requested device mode (`auto`, `cpu`, `coreml`).
- `device_selected`: Actual device selected at runtime.
- `device_selection_reason`: Human-readable reason for selected device.
- `execution_providers`: ONNX Runtime execution providers used.
- `model_source`: Model source type (for example file-based cache path).
- `model_path`: Resolved model file path used for inference.
- `model_size_bytes`: Model file size in bytes.
- `model_load_time_ms`: Time to load model/session in milliseconds.
- `model_checksum`: Model checksum recorded for provenance.

#### Detect ONNX cache (`[detect.system.onnx_cache]`)

- `model_cache_hit`: Whether the ONNX model was already in cache.
- `download_time_ms` (optional): Download time if a model download occurred.
- `cached_models_count`: Count of cached ONNX models.
- `cached_models_size_mb`: Total ONNX cache size in MB.

### Detect input (`[detect.input]`)

- `image_path`: Canonical image path used for processing.
- `source`: Original source argument string that resolved to this image.
- `source_type`: Source type classification (for example `file`, `directory`, `glob`).
- `strict_mode`: Whether strict input validation mode was active.

### Cutout section (`[cutout]`)

- `model_version`: Cutout model version used.
- `input_img_width`: Width of original image in pixels.
- `input_img_height`: Height of original image in pixels.
- `output_path`: Relative output path for the generated cutout image.
- `mask_path` (optional): Relative output path for saved mask image when mask output is enabled.

### Cutout config (`[cutout.config]`)

- `alpha_matting`: Whether alpha matting was enabled.
- `alpha_matting_background_threshold`: Background threshold for alpha matting.
- `alpha_matting_erode_size`: Erode size for alpha matting trimap generation.
- `alpha_matting_foreground_threshold`: Foreground threshold for alpha matting.
- `post_process_mask`: Whether mask post-processing was enabled.
- `save_mask`: Whether a separate mask image was requested.

### Cutout execution (`[cutout.execution]`)

- `timestamp`: UTC timestamp for this cutout invocation.
- `beaker_version`: Beaker version used.
- `command_line`: Full command line for this invocation.
- `exit_code`: Process exit status.
- `model_processing_time_ms`: Total cutout model processing time in milliseconds.

#### Cutout file I/O (`[cutout.execution.file_io]`)

- `read_time_ms`: Cumulative image read time in milliseconds.
- `write_time_ms`: Cumulative file write time in milliseconds.

### Cutout system (`[cutout.system]`)

Field semantics are the same as `detect.system`:

- `device_requested`, `device_selected`, `device_selection_reason`
- `execution_providers`
- `model_source`, `model_path`, `model_size_bytes`, `model_load_time_ms`, `model_checksum`

#### Cutout ONNX cache (`[cutout.system.onnx_cache]`)

- `model_cache_hit`, `download_time_ms` (optional), `cached_models_count`, `cached_models_size_mb`

#### Cutout CoreML cache (`[cutout.system.coreml_cache]`)

- `cache_hit`: Whether a reusable CoreML cache artifact was used.
- `cache_count`: Number of CoreML cache artifacts.
- `cache_size_mb`: Aggregate CoreML cache size in MB.

### Cutout input (`[cutout.input]`)

Field semantics are the same as `detect.input`:

- `image_path`, `source`, `source_type`, `strict_mode`

### Cutout mask (`[cutout.mask]`)

- `width`: Mask width in pixels.
- `height`: Mask height in pixels.
- `format`: Encoding descriptor string (for example `rle-binary-v1 | gzip | base64`).
- `start_value`: First binary value for run-length decoding (`0` or `1`).
- `order`: Pixel traversal order used for encoding (for example `row-major`).
- `data`: Encoded mask payload (`base64(gzip(rle_csv))`).

#### Cutout mask preview (`[cutout.mask.preview]`)

- `format`: Preview format identifier (`ascii`).
- `width`: ASCII preview width in characters.
- `height`: ASCII preview height in rows.
- `rows`: ASCII preview rows (`#` for foreground-like cells, `.` for background-like cells).

### Quality section (`[quality]`)

- `model_version`: Quality model version used.
- `input_img_width`: Original image width in pixels.
- `input_img_height`: Original image height in pixels.
- `global_paq2piq_score`: Global PaQ-2-PiQ score (`0..100`).
- `global_blur_score`: Global fused blur score (`0..1`).
- `global_quality_score`: Final fused quality score (`0..100`) after blur weighting.
- `local_paq2piq_grid`: 20x20 local PaQ-2-PiQ quality grid (`0..100` values).
- `local_blur_weights`: 20x20 blur-weight grid (`0..1` values).
- `local_fused_probability`: 20x20 fused blur-probability grid (`0..1` values).

### Quality config (`[quality.config]`)

- `debug_dump_images`: Whether debug quality images were requested.
- `overlay`: Whether heatmap overlay mode was requested.

### Quality execution (`[quality.execution]`)

Field semantics are the same as `detect.execution`:

- `timestamp`, `beaker_version`, `command_line`, `exit_code`, `model_processing_time_ms`

#### Quality file I/O (`[quality.execution.file_io]`)

- `read_time_ms`: Cumulative image-read time in milliseconds.
- `write_time_ms` (optional): Cumulative write time in milliseconds when applicable.

### Quality system (`[quality.system]`)

Field semantics are the same as `detect.system`:

- `device_requested`, `device_selected`, `device_selection_reason`
- `execution_providers`
- `model_source`, `model_path`, `model_size_bytes`, `model_load_time_ms`, `model_checksum`

#### Quality ONNX cache (`[quality.system.onnx_cache]`)

- `model_cache_hit`, `download_time_ms` (optional), `cached_models_count`, `cached_models_size_mb`

### Quality input (`[quality.input]`)

Field semantics are the same as `detect.input`:

- `image_path`, `source`, `source_type`, `strict_mode`

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
