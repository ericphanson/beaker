# Beaker CLI Integration for Model Comparisons

## Summary of Changes

I've completely refactored the model comparison system to use the actual Beaker Rust CLI tool instead of the problematic Python preprocessing/postprocessing. This should provide much more accurate and reliable comparisons.

## Key Changes Made

### 1. **Refactored `comparisons.py`**

**Before**: Used Python-based ONNX inference with manual preprocessing/postprocessing (prone to errors)

**After**: Uses Beaker Rust CLI with proper preprocessing/postprocessing for accurate results

#### New Functions:
- `run_beaker_inference()` - Calls Beaker CLI with appropriate environment variables
- `collect_beaker_outputs()` - Collects and renames output files with model prefixes to avoid clashes
- `parse_beaker_metadata()` - Parses `.beaker.toml` files for quantitative comparisons
- `compare_head_detections()` - Compares head detection results using IoU metrics
- `generate_head_detection_summary()` - Creates JSON summary of detection comparisons

#### Key Features:
- **Environment variable handling**: Sets `BEAKER_HEAD_MODEL_PATH` or `BEAKER_CUTOUT_MODEL_PATH`
- **Temporary output directories**: Each model gets its own temp directory to avoid clashes
- **File prefixing**: All outputs are prefixed with model name (e.g., `original_example.jpg`, `dynamic-int8_example.jpg`)
- **Metadata parsing**: Extracts bounding boxes and other metrics from `.beaker.toml` files
- **IoU calculations**: Compares detection accuracy between models

### 2. **Updated CLI Commands**

#### Added new options:
- `--beaker-cargo-path`: Specify path to Beaker's Cargo.toml (defaults to `../beaker/Cargo.toml`)

#### Commands updated:
- `upload` - Now passes beaker_cargo_path to comparison generation
- `full_pipeline` - Now supports beaker_cargo_path parameter

### 3. **File Organization Improvements**

The new system creates organized output structure:
```
output/
├── comparisons/
│   ├── head/
│   │   ├── original_example.jpg
│   │   ├── original_example.beaker.toml
│   │   ├── dynamic-int8_example.jpg
│   │   ├── dynamic-int8_example.beaker.toml
│   │   └── comparison_summary.json
│   └── cutout/
│       ├── original_example_cutout.png
│       ├── dynamic-int8_example_cutout.png
│       └── ...
```

### 4. **Added Dependencies**

- Added `tomli` for Python < 3.11 TOML parsing compatibility
- Uses built-in `tomllib` for Python 3.11+

## Usage Examples

### Basic Usage with CLI:

```bash
# Upload with comparisons using custom Beaker path
python -m quantizations.cli upload output/quantized head \
  --include-comparisons \
  --test-images examples/ \
  --beaker-cargo-path /path/to/beaker/Cargo.toml

# Full pipeline with comparisons
python -m quantizations.cli full-pipeline \
  --model-type head \
  --include-comparisons \
  --beaker-cargo-path ../beaker/Cargo.toml
```

### Programmatic Usage:

```python
from quantizations.comparisons import generate_model_comparison_images

models = {
    "Original": Path("output/quantized/head/head-optimized.onnx"),
    "Dynamic-Int8": Path("output/quantized/head/head-dynamic-int8.onnx"),
}

test_images = [Path("examples/example.jpg")]
output_dir = Path("output/comparisons/head")
beaker_cargo_path = Path("../beaker/Cargo.toml")

comparison_files = generate_model_comparison_images(
    models, test_images, output_dir, beaker_cargo_path=beaker_cargo_path
)
```

## How It Works

1. **For each model**:
   - Creates temporary output directory
   - Sets appropriate environment variable (`BEAKER_HEAD_MODEL_PATH` or `BEAKER_CUTOUT_MODEL_PATH`)
   - Runs Beaker CLI: `cargo run --manifest-path ../beaker/Cargo.toml head image.jpg --output-dir temp_dir --crop --bounding-box --metadata`

2. **Collects outputs**:
   - Copies all files from temp directory to final location
   - Prefixes filenames with model name to avoid clashes
   - Preserves directory structure

3. **For head models**:
   - Parses `.beaker.toml` metadata files
   - Extracts bounding box coordinates
   - Calculates IoU between original and quantized model detections
   - Generates JSON summary with quantitative metrics

4. **For cutout models**:
   - Outputs cutout images for visual comparison
   - Preserves metadata for analysis

## Benefits

✅ **Accurate preprocessing/postprocessing** - Uses the same logic as production Beaker CLI
✅ **No file clashes** - Each model gets prefixed outputs in separate temp directories
✅ **Quantitative metrics** - IoU calculations for head detection accuracy
✅ **Visual outputs** - Both cropped heads and cutout images for inspection
✅ **Metadata preservation** - All Beaker metadata available for analysis
✅ **Flexible paths** - Can specify custom Beaker repository location

## Testing

Created `test_beaker_comparisons.py` to test the new functionality:

```bash
cd /Users/eph/beaker/quantizations
python test_beaker_comparisons.py
```

This will test the new comparison system and show you exactly what files are generated.

## Files Changed

- `src/quantizations/comparisons.py` - Complete rewrite to use Beaker CLI
- `src/quantizations/cli.py` - Added beaker_cargo_path parameters
- `pyproject.toml` - Added tomli dependency
- `test_beaker_comparisons.py` - New test script

The new system should provide much more reliable and accurate model comparisons by leveraging the battle-tested Beaker CLI preprocessing and postprocessing logic.
