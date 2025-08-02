# Beaker (Rust)

A fast, self-contained Rust implementation of the Beaker bird head detection CLI tool.

## Features

- ✅ **Fully self-contained**: No external dependencies or library paths required
- ✅ **Embedded model**: Downloads latest ONNX model from GitHub releases during build
- ✅ **Cross-platform**: Supports Linux, macOS, and Windows
- ✅ **Fast**: Optimized release builds with LTO and strip
- ✅ **Real-time inference**: YOLOv8n bird head detection with ONNX Runtime

## Usage

```bash
# Build from source (automatically downloads latest model)
cargo build --release

# Run detection - no environment variables needed!
./target/release/beaker detect example.jpg --confidence 0.75

# Show help
./target/release/beaker --help

# Run with options
./target/release/beaker detect example.jpg --confidence 0.5 --device cpu

# Show version
./target/release/beaker version
```

## Building

```bash
# Build debug version
cargo build

# Build optimized release version
cargo build --release

# Build with Core ML support (macOS only)
cargo build --release --features coreml
```

The build process automatically:
1. Downloads the latest ONNX model from GitHub releases
2. Embeds it into the binary as bytes
3. Downloads and bundles ONNX Runtime libraries with proper rpath
4. Creates a fully self-contained executable

## Architecture

- **Model loading**: Embedded 12MB ONNX model loaded from memory
- **ONNX Runtime**: Automatically downloaded and linked during build
- **Image processing**: Letterbox resizing and tensor preprocessing
- **Inference**: YOLOv8n object detection with configurable confidence
- **Cross-platform**: Uses default execution providers (CPU, CoreML on macOS)
