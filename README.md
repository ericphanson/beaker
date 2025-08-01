# bird-head-detector

A bird head detection dataset converter and tools for YOLO training.

## Convert CUB-200-2011 to YOLO Format

The `convert_to_yolo.py` script converts the CUB-200-2011 bird dataset into YOLO format for bird head detection training. It extracts head-related parts (beak, crown, forehead, eyes, nape, throat) and creates bounding boxes with proper train/validation splits.

### Usage

```bash
# Install dependencies
uv sync

# Run the conversion script
uv run python convert_to_yolo.py
```

### Output

- Creates `data/yolo/` directory with train/val splits
- Generates YOLO format labels (normalized cx, cy, w, h)
- Includes `dataset.yaml` configuration file
- Processes ~11,784 images with progress tracking

| Split | Images | Labels |
|-------|--------|--------|
| Train | 5,990  | 5,990  |
| Val   | 5,794  | 5,794  |
| Total | 11,784 | 11,784 |

## Train YOLOv8n Model on M1 MacBook Pro

After converting the dataset, you can train a YOLOv8n model for bird head detection using Ultralytics.

### Setup

Verify MPS is available:

```bash
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Training

```bash
# Train YOLOv8n model using the training script
uv run python train.py
```

The training script (`train.py`) will:
- Automatically detect MPS availability on M1/M2 Macs
- Load pretrained YOLOv8n weights
- Train for 100 epochs with optimized M1 settings
- Save the best model to `runs/detect/bird_head_yolov8n/weights/best.pt`

#### Debug Mode

For quick testing, enable debug mode by editing `train.py`:

```python
# In TRAINING_CONFIG dictionary
'debug_run': True,  # Set to True for quick testing
'debug_epochs': 5,   # Reduced epochs for debug
'debug_fraction': 0.1,  # Use 10% of data for debug
```

Debug mode will:
- Use only 10% of the training/validation data
- Train for only 5 epochs (vs 100 normal)
- Create a separate `data/yolo_debug/` dataset
- Save model as `bird_head_yolov8n_debug`

#### Comet.ml Integration (Optional)

For experiment tracking with Comet.ml, set these environment variables:

```bash
# Required: Your Comet.ml API key
export COMET_API_KEY="your-api-key-here"

# Optional: Custom project name (default: bird-head-detector)
export COMET_PROJECT_NAME="bird-head-detector"

# Optional: Your Comet.ml workspace/username
export COMET_WORKSPACE="your-username"

# Run training with Comet.ml tracking
uv run python train.py
```

Get your API key from: https://www.comet.ml/api/my/settings/

The integration will automatically log:
- Training hyperparameters
- Loss curves and metrics
- Model artifacts (best.pt)
- System information

### Performance Tips for M1 MacBooks

- **Batch size**: Set to 16 by default, reduce if you get memory errors
- **Image size**: 640 is optimal, can reduce to 416 for faster training
- **Workers**: Set to 0 to prevent multiprocessing issues
- **Mixed precision**: Automatically enabled with MPS device

### Inference

Run detection on new images using the trained model. The inference script will automatically:
1. Look for locally trained models
2. Check for previously downloaded models 
3. Download the latest model from GitHub releases if none found

```bash
# Run inference on a single image
uv run python infer.py --source path/to/image.jpg --show

# Run inference on a directory of images and save results
uv run python infer.py --source path/to/images/ --save

# Create square crops around detected bird heads
uv run python infer.py --source path/to/images/ --crop --crop-dir head_crops

# Use a specific model file
uv run python infer.py --model path/to/model.pt --source image.jpg

# Adjust confidence threshold
uv run python infer.py --source image.jpg --conf 0.5 --show

# Combine options: save results and create crops
uv run python infer.py --source image.jpg --save --crop --show
```

#### Model Auto-Download

The inference script automatically downloads models from GitHub releases:
- Checks local training outputs first (`runs/detect/bird_head_yolov8n/weights/best.pt`)
- Falls back to downloaded models directory (`models/`)
- Downloads latest release model if none found locally
- Supports any GitHub repository with `.pt` model files in releases

The script intelligently handles:
- Repository detection from git remote
- Model caching to avoid re-downloads
- Graceful fallbacks with helpful error messages

#### Inference Features

- **Auto-discovery**: Finds models locally or downloads from releases
- **Multi-format support**: Single images, videos, or directories  
- **Adjustable confidence**: Custom detection thresholds
- **Flexible output**: Display and/or save results
- **M1/M2 optimized**: Uses MPS acceleration on Apple Silicon
- **Detection counting**: Reports number of bird heads found
- **Square cropping**: Creates square crops around detected bird heads (--crop)

#### Head Cropping

The `--crop` option creates square image crops centered on detected bird heads:
- **Highest confidence**: Uses the detection with highest confidence when multiple found
- **Square format**: Automatically expands bounding box to square dimensions
- **Smart padding**: Adds 20% padding around the detection for context
- **Boundary handling**: Adjusts crop to stay within image boundaries
- **Custom output**: Specify crop directory with `--crop-dir` (default: `crops/`)
- **Filename format**: Saves as `{original_name}_head_crop.jpg`

```bash
# Create crops for all images with bird head detections
uv run python infer.py --source bird_photos/ --crop

# Save crops to specific directory
uv run python infer.py --source image.jpg --crop --crop-dir my_crops
```

## Release Management

The `release.py` script automates the process of creating GitHub releases with trained models.

### Prerequisites

```bash
# Install GitHub CLI
# macOS: brew install gh
# Other: https://cli.github.com/

# Authenticate with GitHub
gh auth login
```

### Creating a Release

```bash
# Create a release with the trained model
uv run python release.py
```

The release script will:
1. **Check prerequisites** - Verify gh CLI and authentication
2. **Validate repository** - Ensure no uncommitted changes
3. **Show existing versions** - Display current tags/releases
4. **Prompt for version** - Request new semantic version (e.g., 1.0.0)
5. **Create git tag** - Tag the current commit
6. **Upload model** - Add best.pt as release asset
7. **Push to GitHub** - Create public release

### Safety Features

- ❌ **Blocks dirty repos** - Won't proceed with uncommitted changes
- 🏷️ **Version validation** - Enforces semantic versioning
- 🔍 **Duplicate detection** - Prevents overwriting existing releases
- 📋 **Confirmation prompt** - Shows summary before proceeding
