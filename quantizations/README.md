# Beaker Model Quantizations

This package provides tools to quantize ONNX models for the Beaker bird detection toolkit, creating smaller, faster models while maintaining accuracy.

## Features

- **Model Download**: Automatically downloads the latest ONNX models from GitHub releases
- **Multiple Quantization Levels**: Supports dynamic and static INT8 quantization with ONNX optimization
- **ONNX Optimization**: Uses onnx-simplifier for graph optimization and optimization passes
- **Model Comparison**: Compares quantized models against originals using the beaker CLI for visual side-by-side results
- **Upload**: Publishes quantized models to new GitHub releases with comparison images
- **Full Pipeline**: Automated workflow from download to upload

## Model Support

### Head Detection Models
- **Base Model**: YOLOv8n-based bird detect detection model (best.onnx)
- **Source**: `bird-detect-detector-*` releases from GitHub
- **Quantization Levels**: Dynamic INT8, Static INT8, FP16
- **Use Case**: Bird detect detection and cropping

### Cutout Models
- **Base Model**: Custom segmentation model for background removal
- **Source**: `beaker-cutout-model-*` releases from GitHub
- **Quantization Levels**: Dynamic INT8, Static INT8, FP16
- **Use Case**: Background removal and segmentation

## Prerequisites

Before using this tool, ensure you have:

1. **uv package manager** installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart your terminal
   ```

2. **Beaker CLI** for running inference comparisons:
   ```bash
   # Install beaker CLI (adjust path as needed)
   # Make sure 'beaker' is available in your PATH
   beaker --help
   ```

3. **GitHub CLI** (for uploading quantized models):
   ```bash
   # Install gh CLI (varies by OS)
   # Ubuntu/Debian: apt install gh
   # macOS: brew install gh
   # Windows: winget install GitHub.CLI

   # Authenticate with GitHub
   gh auth login
   ```

4. **Git** configured with your credentials

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
uv run quantize-models full-pipeline --model-type detect --dry-run

# Actually run the pipeline
uv run quantize-models full-pipeline --model-type detect --test-image ../example.jpg
```

This will:
1. Download the latest detect detection model
2. Create quantized versions (dynamic and static INT8)
3. Apply ONNX optimizations and simplifications
4. Generate comparison images using the beaker CLI
5. Upload them to a new GitHub release with comparison images

### Individual Commands

#### 1. Download Models

Download the latest ONNX models from GitHub releases:

```bash
# Download detect detection models
uv run quantize-models download --model-type detect -o models/

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

#### 3. Compare Quantized Models

Compare quantized models against the original using the beaker CLI:

```bash
# Basic comparison (will auto-find example images)
uv run quantize-models compare models/best.onnx quantized/best-dynamic.onnx

# Specify test image and model type
uv run quantize-models compare models/best.onnx quantized/best-dynamic.onnx \
  --test-image ../example.jpg --model-type detect

# Custom output directory
uv run quantize-models compare models/best.onnx quantized/best-dynamic.onnx \
  --output-dir comparisons/
```

This command will:
- Run inference on both models using the beaker CLI
- Generate side-by-side comparison images
- Save results to the specified output directory

#### 4. Upload Quantized Models

Upload quantized models to GitHub releases:

```bash
# Basic upload
uv run quantize-models upload quantized/ --model-type detect --version v1.0

# Include comparison images
uv run quantize-models upload quantized/ --model-type detect --version v1.0 \
  --include-comparisons --test-image ../example.jpg

# Dry run to preview release
uv run quantize-models upload quantized/ --model-type detect --version v1.0 --dry-run
```

The upload command now creates releases with:
- All quantized models for the specified type in one release
- Visual comparison images generated using the beaker CLI
- Comprehensive release notes with optimization details
- SHA256 checksums for all files

## Integration with Beaker CLI

Use quantized models with the existing Beaker CLI:

```bash
# Set environment variable to use quantized model
export BEAKER_DETECT_MODEL_URL="https://github.com/ericphanson/beaker/releases/download/detect-quantizations-v1.0/detect-fp16.onnx"

# Use with beaker as normal
beaker detect image.jpg --crop

# For cutout models
export BEAKER_CUTOUT_MODEL_URL="https://github.com/ericphanson/beaker/releases/download/cutout-quantizations-v1.0/cutout-dynamic-int8.onnx"
beaker cutout image.jpg
```

## Quantization Benefits

Quantization provides several advantages:

- **Size Reduction**: INT8 quantization typically reduces model size by ~70%
- **Faster Inference**: Quantized models run faster on CPU hardware
- **Lower Memory Usage**: Reduced memory footprint during inference
- **Maintained Accuracy**: Visual comparison shows minimal quality degradation

Use the `compare` command to generate side-by-side visual comparisons and verify that quantized models meet your quality requirements.

## Integration with Beaker CLI

Quantized models can be used seamlessly with the existing Beaker CLI:

```bash
# Use quantized detect detection model
export BEAKER_DETECT_MODEL_URL=https://github.com/ericphanson/beaker/releases/download/detect-quantizations-v1/best-dynamic.onnx
beaker detect image.jpg --crop

# Use quantized cutout model
export BEAKER_CUTOUT_MODEL_URL=https://github.com/ericphanson/beaker/releases/download/cutout-quantizations-v1/model-dynamic.onnx
beaker cutout image.jpg
```

## GitHub Releases Structure

Quantized models are published to GitHub releases with semantic tags:

### Head Detection Quantizations
- **Tag Pattern**: `detect-quantizations-v{version}`
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

#### 5. Comparison failures
```bash
# Make sure beaker CLI is available
beaker --help

# Check if models are compatible with beaker
beaker detect --help
beaker cutout --help

# Try with verbose logging
uv run quantize-models compare model.onnx quantized.onnx -v
```

#### 6. Upload failures
- Verify GitHub CLI is authenticated: `gh auth status`
- Check repository permissions (need write access)
- Ensure the tag doesn't already exist

### Verbose Logging

Add `-v` or `--verbose` to any command for detailed logging:

```bash
uv run quantize-models full-pipeline --model-type detect --tolerance 200 -v
```

### Manual Verification

You can manually verify quantized models:

```bash
# Check file sizes
ls -lh models/ quantized/

# Verify checksums
sha256sum quantized/*.onnx

# Test with beaker CLI
export BEAKER_DETECT_MODEL_URL=./quantized/best-dynamic.onnx
beaker detect ../example.jpg
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
├── inference.py       # Model comparison using beaker CLI
└── py.typed          # Type hint marker
```

### Contributing

1. Make changes to the source code
2. Run linting: `ruff check src/ && ruff format src/`
3. Test your changes with the CLI commands
4. Update documentation if needed
