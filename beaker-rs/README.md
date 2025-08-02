# Beaker (Rust)

A fast, self-contained Rust implementation of the Beaker bird head detection CLI tool.

## Features

- ✅ **Fully self-contained**: No external dependencies or library paths required
- ✅ **Embedded model**: Downloads latest ONNX model from GitHub releases during build
- ✅ **Cross-platform**: Supports Linux, macOS, and Windows
- ✅ **Fast**: Optimized release builds with LTO and strip
- ✅ **Real-time inference**: YOLOv8n bird head detection with ONNX Runtime
- ✅ **Multiple outputs**: Individual crops per detection, bounding box visualization, TOML metadata
- ✅ **Smart file naming**: Automatic numbering for multiple detections with zero-padding
- ✅ **Relative paths**: TOML output uses paths relative to output location
- ✅ **Modular architecture**: Separated preprocessing, postprocessing, and detection modules

## Usage

```bash
# Build from source (automatically downloads latest model)
cargo build --release

# Run head detection - no environment variables needed!
./target/release/beaker head example.jpg --confidence 0.75

# Show help
./target/release/beaker --help

# Run with crop and bounding box outputs
./target/release/beaker head example.jpg --crop --bounding-box

# Run with custom confidence and IoU thresholds
./target/release/beaker head example.jpg --confidence 0.5 --iou-threshold 0.4

# Use global output directory and skip TOML output
./target/release/beaker --output-dir ./results --skip-toml head example.jpg --crop

# Run on CPU explicitly
./target/release/beaker head example.jpg --device cpu
```

## Building

```bash
# Build debug version
cargo build

# Build optimized release version
cargo build --release
```

The build process automatically:
1. Downloads the latest ONNX model from GitHub releases
2. Embeds it into the binary as bytes
3. Downloads and bundles ONNX Runtime libraries with proper rpath
4. Creates a fully self-contained executable
5. Detects platform capabilities (CoreML on macOS, CPU elsewhere)

## Architecture

- **Modular design**: Separated into `head_detection`, `yolo_preprocessing`, and `yolo_postprocessing` modules
- **Model loading**: Embedded 12MB ONNX model loaded from memory
- **ONNX Runtime**: Automatically downloaded and linked during build
- **Image processing**: Letterbox resizing and tensor preprocessing
- **Inference**: YOLOv8n object detection with configurable confidence and IoU thresholds
- **Post-processing**: Non-Maximum Suppression (NMS) for clean detection results
- **Output generation**: Individual crops per detection, combined bounding box visualization, structured TOML metadata
- **Cross-platform**: Uses optimal execution providers (CoreML on macOS, CPU elsewhere)

## Output Files

When running detection, beaker can generate several output files:

- **Crops**: Individual square crops for each detected bird head (`image-crop-1.jpg`, `image-crop-2.jpg`, etc.)
  - Single detection: `image-crop.jpg` (no suffix)
  - Multiple detections: Numbered with zero-padding for 10+ detections
- **Bounding boxes**: Single image showing all detections with green boxes (`image-bounding-box.jpg`)
- **TOML metadata**: Structured output with detection coordinates, confidence scores, and relative file paths (`image-beaker.toml`)

Example TOML output:
```toml
[head]
model_version = "bird-head-detector-v1.0.0"
confidence_threshold = 0.25
iou_threshold = 0.45
bounding_box_path = "example-bounding-box.jpg"

[[head.detections]]
x1 = 786.0
y1 = 392.6
x2 = 954.2
y2 = 475.8
confidence = 0.774
crop_path = "example-crop-1.jpg"

[[head.detections]]
x1 = 378.0
y1 = 407.8
x2 = 557.4
y2 = 582.3
confidence = 0.473
crop_path = "example-crop-2.jpg"
```
