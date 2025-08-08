# File Path Clash Analysis and Resolution

## Problem Analysis

The quantization codebase had several file path conflicts between the two models (head and cutout) being processed:

### 1. **Shared Output Directories**
- Both models used the same directories:
  - `output/models/` - for downloaded models
  - `output/quantized/` - for quantized models
  - `output/comparisons/` - for comparison images

### 2. **Generic Model Names**
- Head model downloads as `best.onnx` (very generic)
- No clear prefix to distinguish head vs cutout models
- Risk of overwriting files when processing multiple model types

### 3. **Quantized Model Naming Conflicts**
- All quantized models go to the same `output/quantized/` directory
- Potential for files like `head-dynamic-int8.onnx` to be overwritten
- Comparison images use generic names like `comparison_example.png`

### 4. **Validation and Comparison Issues**
- Comparison images don't distinguish between model types
- Performance metrics could be mixed between different model types
- Upload process might upload wrong quantized models for a given model type

## Solutions Implemented

### 1. **Model-Type-Specific Directory Structure**

**Before:**
```
output/
├── models/
│   ├── best.onnx                 # Generic name - clash risk
│   └── cutout-model.onnx
├── quantized/                    # All models mixed together
│   ├── head-dynamic-int8.onnx
│   ├── head-static-int8.onnx
│   └── cutout-dynamic-int8.onnx
└── comparisons/                  # All comparison images mixed
    ├── comparison_example.png
    └── comparison_example_crop.png
```

**After:**
```
output/
├── models/
│   ├── head/
│   │   └── head-best.onnx        # Prefixed to avoid clashes
│   └── cutout/
│       └── cutout-model.onnx
├── quantized/
│   ├── head/                     # Separate directory for head quantizations
│   │   ├── head-optimized.onnx
│   │   ├── head-dynamic-int8.onnx
│   │   └── head-static-int8.onnx
│   └── cutout/                   # Separate directory for cutout quantizations
│       ├── cutout-optimized.onnx
│       ├── cutout-dynamic-int8.onnx
│       └── cutout-static-int8.onnx
└── comparisons/
    ├── head/                     # Model-specific comparison images
    │   ├── head-comparison_example.png
    │   └── head-comparison_example_crop.png
    └── cutout/
        ├── cutout-comparison_example.png
        └── cutout-comparison_example_crop.png
```

### 2. **Enhanced File Naming**

#### **Download Phase** (`downloader.py`)
- Head models: Ensure filename includes "head" prefix if not present
- Cutout models: Ensure filename includes "cutout" prefix if not present
- `best.onnx` → `head-best.onnx` to avoid generic naming

#### **Quantization Phase** (`quantizer.py` via `cli.py`)
- Auto-detect model type from path or parent directory name
- Create model-type-specific output directories
- Maintain existing naming convention within each type-specific directory

#### **Comparison Phase** (`comparisons.py`)
- Add model-type prefix to comparison image filenames
- `comparison_example.png` → `head-comparison_example.png`
- Store in model-type-specific subdirectories

### 3. **CLI Command Updates**

#### **Download Command**
```bash
# Before: Downloads to single directory
beaker download --output-dir ./models --model-type head

# After: Downloads to model-type subdirectories
beaker download --output-dir ./models --model-type head
# Creates: ./models/head/head-best.onnx
```

#### **Quantize Command**
```bash
# Before: All quantized models in same directory
beaker quantize model.onnx --output-dir ./quantized

# After: Model-type-specific directories
beaker quantize ./models/head/head-best.onnx --output-dir ./quantized --model-type head
# Creates: ./quantized/head/head-dynamic-int8.onnx
```

#### **Upload Command**
```bash
# Before: Uploads from flat quantized directory
beaker upload ./quantized --model-type head

# After: Uploads from model-type-specific directory
beaker upload ./quantized --model-type head
# Looks in: ./quantized/head/ for head-specific quantized models
```

### 4. **Backward Compatibility**

The implementation includes fallback logic for existing directory structures:

- If model-type-specific directories don't exist, fall back to original structure
- CLI commands work with both old and new directory layouts
- Gradual migration path for existing workflows

### 5. **Full Pipeline Updates**

The `full_pipeline` command now:
- Downloads models to type-specific directories
- Quantizes to type-specific output directories
- Validates and generates comparisons separately for each model type
- Uploads each model type independently with correct file sets

## Benefits of the Resolution

### 1. **Eliminates File Clashes**
- No more risk of overwriting models or quantized outputs
- Clear separation between head and cutout processing
- Comparison images are properly labeled and organized

### 2. **Improved Organization**
- Easy to find and manage models by type
- Clear directory structure for CI/CD and automation
- Better debugging and troubleshooting capabilities

### 3. **Scalable Architecture**
- Easy to add new model types in the future
- Each model type can have independent processing pipelines
- Upload process correctly handles model-specific files

### 4. **Enhanced Validation**
- Model-specific validation metrics
- Separate performance tracking for each model type
- Reduced risk of cross-contamination in results

## Migration Guide

### For Existing Users:
1. **Immediate**: New commands automatically use the new structure
2. **Existing Data**: Old directory structures continue to work with fallback logic
3. **Gradual Migration**: Can run `full_pipeline` command to reorganize existing data

### For CI/CD:
1. Update build scripts to expect model-type-specific directories
2. Adjust upload logic to handle separate model type releases
3. Update validation scripts to check correct directories

## Testing Recommendations

1. **Test Both Structures**: Verify fallback logic works with existing data
2. **Cross-Model Testing**: Ensure head and cutout processing don't interfere
3. **Upload Validation**: Confirm correct models are uploaded for each type
4. **Directory Creation**: Test that all required directories are created automatically
