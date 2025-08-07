# Beaker Model Quantizations

This package provides tools to quantize ONNX models for the Beaker bird detection toolkit, creating smaller, faster models while maintaining accuracy.

## Features

- **Model Download**: Automatically downloads the latest ONNX models from GitHub releases
- **Multiple Quantization Levels**: Supports dynamic, static, and INT8 quantization
- **Validation**: Compares quantized models against originals using multiple metrics
- **Upload**: Publishes quantized models to new GitHub releases with checksums
- **Full Pipeline**: Automated workflow from download to upload

## Installation

```bash
cd quantizations
uv sync
```

## Usage

### Quick Start - Full Pipeline

Run the complete quantization pipeline:

```bash
uv run quantize-models full-pipeline --model-type head --tolerance 200 --dry-run
```

### Individual Commands

Download models:
```bash
uv run quantize-models download --model-type head -o models/
```

Quantize a model:
```bash
uv run quantize-models quantize models/best.onnx -o quantized/ --levels dynamic static
```

Validate quantized model:
```bash
uv run quantize-models validate models/best.onnx quantized/best-dynamic.onnx --tolerance 200
```

Upload quantizations (requires GitHub CLI authentication):
```bash
uv run quantize-models upload quantized/ --model-type head --version v1 --dry-run
```

## Quantization Results

- **Dynamic Quantization**: ~74% size reduction (12MB → 3.2MB)
- **Accuracy**: >99.99% cosine similarity maintained
- **Performance**: Faster inference on CPU

## Model Support

- **Head Detection**: Bird head detection models (`bird-head-detector-*` releases)
- **Cutout**: Background removal models (`beaker-cutout-model-*` releases)

## Integration with Beaker CLI

Quantized models can be used with the Beaker CLI via environment variables:

```bash
export BEAKER_HEAD_MODEL_URL=<quantized_model_download_url>
beaker head image.jpg --crop

export BEAKER_CUTOUT_MODEL_URL=<quantized_cutout_model_url>
beaker cutout image.jpg
```

## GitHub Releases

Quantized models are published to GitHub releases with tags:
- `head-quantizations-v1` - Head detection model quantizations
- `cutout-quantizations-v1` - Cutout model quantizations

Each release includes:
- Multiple quantization levels (dynamic, static)
- SHA256 checksums for verification
- Performance and size reduction metrics
