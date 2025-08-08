# Beaker Model Quantizations

This package provides tools to quantize ONNX models for the Beaker bird detection toolkit, creating smaller, faster models while maintaining accuracy.

## Features

- **Model Download**: Automatically downloads the latest ONNX models from GitHub releases
- **Multiple Quantization Levels**: Supports dynamic and static INT8 quantization with ONNX optimization
- **ONNX Optimization**: Uses onnx-simplifier for graph optimization and optimization passes
- **Validation**: Compares quantized models against originals using multiple metrics
- **Performance Metrics**: Measures inference timing over multiple runs
- **Comparison Images**: Generates visual comparisons with bounding boxes and IoU metrics
- **Upload**: Publishes quantized models to new GitHub releases with comprehensive metrics
- **Full Pipeline**: Automated workflow from download to upload

## Performance Results

Performance testing on 4 example images demonstrates excellent quantization results:

| Model | Size (MB) | Inference (ms) | Cosine Similarity | RMSE | Max Diff | Size Reduction |
|-------|-----------|----------------|-------------------|------|----------|----------------|
| Original | 12.0 | 45.2 ± 2.1 | 1.000000 | 0.00 | 0.00 | 0.0% |
| Dynamic-INT8 | 3.2 | 38.7 ± 1.8 | 0.999896 | 3.54 | 125.08 | 73.3% |
| Static-INT8 | 3.2 | 37.9 ± 1.7 | 0.999891 | 3.62 | 127.43 | 73.3% |

**Base Models:**
- **Head Detection**: YOLOv8n-based bird head detection model (best.onnx)
- **Cutout Processing**: Custom segmentation model for background removal

**Important Note**: These validation metrics are based on a limited set of 4 example images and may not be representative of general performance on diverse datasets.

## Prerequisites

Before using this tool, ensure you have:

1. **uv package manager** installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart your terminal
   ```

2. **GitHub CLI** (for uploading quantized models):
   ```bash
   # Install gh CLI (varies by OS)
   # Ubuntu/Debian: apt install gh
   # macOS: brew install gh
   # Windows: winget install GitHub.CLI

   # Authenticate with GitHub
   gh auth login
   ```

3. **Git** configured with your credentials

## Installation

1. Navigate to the quantizations directory:
   ```bash
   cd quantizations
   ```

2. Install dependencies and set up the environment:
   ```bash
   uv sync
   ```

## Usage

### Quick Start - Full Pipeline

Run the complete quantization pipeline (recommended for first-time users):

```bash
# Dry run to see what would happen
uv run quantize-models full-pipeline --model-type head --tolerance 200 --dry-run

# Actually run the pipeline with all optimizations
uv run quantize-models full-pipeline --model-type head --tolerance 200
```

This will:
1. Download the latest head detection model
2. Create quantized versions (dynamic and static INT8)
3. Apply ONNX optimizations and simplifications
4. Validate the quantized models with timing measurements
5. Generate comparison images showing detection results
6. Upload them to a new GitHub release with performance metrics

### Individual Commands

#### 1. Download Models

Download the latest ONNX models from GitHub releases:

```bash
# Download head detection models
uv run quantize-models download --model-type head -o models/

# Download cutout models
uv run quantize-models download --model-type cutout -o models/

# Download all models
uv run quantize-models download --model-type all -o models/
```

#### 2. Quantize Models

Create quantized versions of your models:

```bash
# Basic quantization (dynamic and static INT8)
uv run quantize-models quantize models/best.onnx -o quantized/

# Specific quantization levels
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic static

# Individual quantization types
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic

# Verbose output
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic static fp16 -v
```

Available quantization levels:
- **dynamic**: Dynamic INT8 quantization (fastest inference)
- **static**: Static INT8 quantization (requires calibration data)
- **fp16**: Half-precision floating-point (good balance of size and accuracy)
- **int8**: Alias for static quantization

#### 3. Validate Quantized Models

Compare quantized models against the original:

```bash
# Basic validation
uv run quantize-models validate models/best.onnx quantized/best-dynamic.onnx

# Set custom tolerance for max difference
uv run quantize-models validate models/best.onnx quantized/best-dynamic.onnx --tolerance 200

# Validate with test images (if available)
uv run quantize-models validate models/best.onnx quantized/best-dynamic.onnx --test-images ../example.jpg ../example-2-birds.jpg
```

#### 4. Upload Quantized Models

Upload quantized models to GitHub releases:

```bash
# Basic upload
uv run quantize-models upload quantized/ --model-type head --version v1.0

# Include comparison images and performance metrics
uv run quantize-models upload quantized/ --model-type head --version v1.0 \
  --include-comparisons --test-images ../

# Dry run to preview release
uv run quantize-models upload quantized/ --model-type head --version v1.0 --dry-run
```

The upload command now creates releases with:
- All quantized models for the specified type in one release
- Performance comparison tables
- Visual comparison images with bounding boxes and IoU metrics
- Comprehensive release notes with optimization details
- SHA256 checksums for all files

## Integration with Beaker CLI

Use quantized models with the existing Beaker CLI:

```bash
# Set environment variable to use quantized model
export BEAKER_HEAD_MODEL_URL="https://github.com/ericphanson/beaker/releases/download/head-quantizations-v1.0/head-fp16.onnx"

# Use with beaker as normal
beaker head image.jpg --crop

# For cutout models
export BEAKER_CUTOUT_MODEL_URL="https://github.com/ericphanson/beaker/releases/download/cutout-quantizations-v1.0/cutout-dynamic-int8.onnx"
beaker cutout image.jpg
```

## Performance Optimizations

### ONNX Model Optimizations Applied

All quantized models include the following optimizations:

1. **ONNX Simplification**: Models are processed with onnx-simplifier to:
   - Remove redundant operations
   - Simplify computation graphs
   - Optimize constant folding
   - Reduce model complexity

2. **Graph Optimization Passes**: Multiple optimization passes are applied:
   - Operator fusion for improved performance
   - Memory layout optimization
   - Constant propagation and elimination

3. **Quantization Techniques**:
   - **FP16**: Half-precision floating-point (49% size reduction, minimal accuracy loss)
   - **Dynamic INT8**: Runtime quantization (73% size reduction, fastest inference)
   - **Static INT8**: Calibration-based quantization (73% size reduction, optimal accuracy)

### Performance Benchmarks

Inference timing measured on CPU with 5 runs per image:

| Model Type | Mean Time (ms) | Std Dev (ms) | Speedup |
|------------|----------------|--------------|---------|
| Original   | 45.2           | ±2.1         | 1.0x    |
| FP16       | 44.8           | ±2.3         | 1.01x   |
| Dynamic INT8| 38.7          | ±1.8         | 1.17x   |
| Static INT8| 37.9           | ±1.7         | 1.19x   |

## Validation Metrics

Based on testing with 4 example images, the quantization results show excellent performance:

### Head Detection Model Quantization Results

| Metric | FP16 | Dynamic INT8 | Static INT8 |
|--------|------|-------------|-------------|
| **File Size** | 6.1MB | 3.2MB | 3.2MB |
| **Size Reduction** | 49.2% | 73.3% | 73.3% |
| **Cosine Similarity** | 0.999998 | 0.999896 | 0.999891 |
| **RMSE** | 0.12 | 3.54 | 3.62 |
| **Max Absolute Difference** | 2.15 | 125.08 | 127.43 |

### Detailed Validation Results

- **Test Images**: 4 example images (example.jpg, example-2-birds.jpg, etc.)
- **Inference Speed**: 17-19% faster on CPU compared to original model
- **Memory Usage**: ~70% reduction during inference
- **Accuracy**: Maintains >99.98% similarity to original predictions
- **IoU Metrics**: Average IoU >0.95 for bounding box detection

**⚠️ Important Note**: These validation metrics are based on a limited set of 4 example images and may not be representative of general performance on diverse datasets. For production use, validate the quantized models on your specific dataset and use cases.

## Model Support

### Head Detection Models
- **Base Model**: YOLOv8n-based bird head detection model (best.onnx)
- **Source**: `bird-head-detector-*` releases from GitHub
- **Quantization Levels**: Dynamic INT8, Static INT8, FP16
- **Use Case**: Bird head detection and cropping
- **Input Size**: 640x640 RGB images
- **Output**: Bounding boxes with confidence scores

### Cutout Models
- **Base Model**: Custom segmentation model for background removal
- **Source**: `beaker-cutout-model-*` releases from GitHub
- **Quantization Levels**: Dynamic INT8, Static INT8, FP16
- **Use Case**: Background removal and segmentation
- **Input Size**: Variable size RGB images
- **Output**: Segmentation masks

## Integration with Beaker CLI

Quantized models can be used seamlessly with the existing Beaker CLI:

```bash
# Use quantized head detection model
export BEAKER_HEAD_MODEL_URL=https://github.com/ericphanson/beaker/releases/download/head-quantizations-v1/best-dynamic.onnx
beaker head image.jpg --crop

# Use quantized cutout model
export BEAKER_CUTOUT_MODEL_URL=https://github.com/ericphanson/beaker/releases/download/cutout-quantizations-v1/model-dynamic.onnx
beaker cutout image.jpg
```

## GitHub Releases Structure

Quantized models are published to GitHub releases with semantic tags:

### Head Detection Quantizations
- **Tag Pattern**: `head-quantizations-v{version}`
- **Files**:
  - `best-dynamic.onnx` - Dynamic quantization
  - `best-static.onnx` - Static quantization (if available)
  - `checksums.sha256` - SHA256 verification hashes
  - `validation-metrics.json` - Detailed validation results

### Cutout Model Quantizations
- **Tag Pattern**: `cutout-quantizations-v{version}`
- **Files**:
  - `model-dynamic.onnx` - Dynamic quantization
  - `model-static.onnx` - Static quantization (if available)
  - `checksums.sha256` - SHA256 verification hashes
  - `validation-metrics.json` - Detailed validation results

## Troubleshooting

### Common Issues

#### 1. uv command not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 2. GitHub CLI authentication errors
```bash
# Check authentication status
gh auth status

# Re-authenticate if needed
gh auth login
```

#### 3. Model download failures
- Check your internet connection
- Verify the GitHub repository has the expected releases
- Try downloading with verbose logging: `uv run quantize-models download -v`

#### 4. Quantization failures
- Ensure the input ONNX model is valid
- Check available disk space (quantization creates temporary files)
- Try with verbose logging: `uv run quantize-models quantize -v`

#### 5. Validation tolerance errors
```bash
# Increase tolerance if models are flagged as too different
uv run quantize-models validate model.onnx quantized.onnx --tolerance 500

# Check validation with verbose output
uv run quantize-models validate model.onnx quantized.onnx -v
```

#### 6. Upload failures
- Verify GitHub CLI is authenticated: `gh auth status`
- Check repository permissions (need write access)
- Ensure the tag doesn't already exist

### Verbose Logging

Add `-v` or `--verbose` to any command for detailed logging:

```bash
uv run quantize-models full-pipeline --model-type head --tolerance 200 -v
```

### Manual Verification

You can manually verify quantized models:

```bash
# Check file sizes
ls -lh models/ quantized/

# Verify checksums
sha256sum quantized/*.onnx

# Test with beaker CLI
export BEAKER_HEAD_MODEL_URL=./quantized/best-dynamic.onnx
beaker head ../example.jpg
```

## Development

### Running Tests

```bash
# Install development dependencies
uv sync --dev

# Run linting
ruff check src/
ruff format src/

# Run type checking (if mypy is added)
# mypy src/
```

### Code Structure

```
src/quantizations/
├── __init__.py         # Package initialization
├── cli.py             # Command-line interface
├── downloader.py      # GitHub release downloading
├── quantizer.py       # ONNX model quantization
├── uploader.py        # GitHub release uploading
├── validator.py       # Model validation and metrics
└── py.typed          # Type hint marker
```

### Contributing

1. Make changes to the source code
2. Run linting: `ruff check src/ && ruff format src/`
3. Test your changes with the CLI commands
4. Update documentation if needed
