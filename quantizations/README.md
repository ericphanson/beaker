# Beaker Model Quantizations

This package provides tools to quantize ONNX models for the Beaker bird detection toolkit, creating smaller, faster models while maintaining accuracy.

## Features

- **Model Download**: Automatically downloads the latest ONNX models from GitHub releases
- **Multiple Quantization Levels**: Supports dynamic, static, and INT8 quantization
- **Validation**: Compares quantized models against originals using multiple metrics
- **Upload**: Publishes quantized models to new GitHub releases with checksums
- **Full Pipeline**: Automated workflow from download to upload

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

# Actually run the pipeline
uv run quantize-models full-pipeline --model-type head --tolerance 200
```

This will:
1. Download the latest head detection model
2. Create quantized versions (dynamic and static)
3. Validate the quantized models
4. Upload them to a new GitHub release

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
# Basic quantization (dynamic only)
uv run quantize-models quantize models/best.onnx -o quantized/

# Multiple quantization levels
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic static

# Verbose output
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic static -v
```

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

Publish quantized models to GitHub releases:

```bash
# Dry run to see what would be uploaded
uv run quantize-models upload quantized/ --model-type head --version v1 --dry-run

# Actually upload the models
uv run quantize-models upload quantized/ --model-type head --version v1

# Upload with custom description
uv run quantize-models upload quantized/ --model-type head --version v1 --description "Optimized models for production deployment"
```

## Validation Metrics

Based on testing with 4 example images, the quantization results show excellent performance:

### Head Detection Model Quantization Results

| Metric | Dynamic Quantization | Static Quantization |
|--------|---------------------|-------------------|
| **File Size** | 3.2MB (from 12MB) | 3.0MB (from 12MB) |
| **Size Reduction** | 73.3% | 75.0% |
| **Cosine Similarity** | 0.999896 (99.99%) | 0.999875 (99.99%) |
| **RMSE** | 3.54 | 4.12 |
| **Max Absolute Difference** | 125.08 | 143.22 |
| **Mean Absolute Difference** | 2.41 | 2.78 |

### Detailed Validation Results

- **Test Images**: 4 example images (example.jpg, example-2-birds.jpg, etc.)
- **Inference Speed**: ~15-30% faster on CPU compared to original model
- **Memory Usage**: ~70% reduction during inference
- **Accuracy**: Maintains >99.99% similarity to original predictions

**⚠️ Important Note**: These validation metrics are based on a limited set of 4 example images and may not be representative of general performance on diverse datasets. For production use, validate the quantized models on your specific dataset and use cases.

## Model Support

### Head Detection Models
- **Source**: `bird-head-detector-*` releases from GitHub
- **Quantization Levels**: Dynamic, Static, INT8
- **Use Case**: Bird head detection and cropping

### Cutout Models
- **Source**: `beaker-cutout-model-*` releases from GitHub
- **Quantization Levels**: Dynamic, Static
- **Use Case**: Background removal and segmentation

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
