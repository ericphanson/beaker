# Beaker (Rust)

A Rust implementation of the Beaker bird head detection CLI tool.

## Usage

Command:
```sh
ORT_DYLIB_PATH=./onnxruntime-libs/onnxruntime-osx-arm64-1.16.3/lib/libonnxruntime.dylib ./beaker-rs/target/debug/beaker detect example.jpg --confidence 0.75
```

```bash
# Show help
./beaker -h

# Detect bird heads in an image (skeleton - not implemented yet)
./beaker detect example.jpg

# Run with options
./beaker detect example.jpg --confidence 0.5 --device cpu --output crops/

# Show version
./beaker version
```

## Building

```bash
# Build debug version
cargo build

# Build release version
cargo build --release

# Build with Core ML support (macOS only)
cargo build --release --features coreml
```

## Status

This is currently a skeleton implementation. The actual YOLO inference, image processing, and detection functionality is not yet implemented.
