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
