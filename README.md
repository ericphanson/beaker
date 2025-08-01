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

```bash
# Run inference on a single image
uv run python infer.py --source path/to/image.jpg --show

# Run inference on a directory of images and save results
uv run python infer.py --source path/to/images/ --save

# Adjust confidence threshold
uv run python infer.py --source image.jpg --conf 0.5 --show
```

The inference script (`infer.py`) supports:
- Single images, videos, or directories
- Adjustable confidence threshold
- Option to display or save results
- Automatic detection counting
