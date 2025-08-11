# RF-DETR Detection Project

This project contains tools for working with the RF-DETR object detection model, including dataset conversion utilities.

Data lives in `../data`

## CUB Dataset Conversion

The `convert_cub_to_coco_format.py` script converts the CUB-200-2011 bird dataset to COCO JSON format for training object detection models.

### Usage

```bash
# Convert CUB dataset with default paths (outputs to ../data/coco_annotations)
uv run python convert_cub_to_coco_format.py

# Convert using parts annotations for 4-class detection (bird, head, eye, beak)
uv run python convert_cub_to_coco_format.py --parts

# Specify custom paths
uv run python convert_cub_to_coco_format.py --cub_root /path/to/CUB_200_2011 --output_dir ./annotations
```

### Conversion Modes

**Standard Mode (200 bird species):**
- Creates annotations for 200 bird species classification
- One annotation per image using the provided bounding box
- Output files: `cub_train.json`, `cub_test.json`

**Parts Mode (4-class detection):**
- Creates annotations for 4 object classes: bird, head, eye, beak
- Multiple annotations per image (up to 4)
- Uses CUB parts annotations to generate bounding boxes
- Output files: `cub_train_parts.json`, `cub_test_parts.json`
- Part definitions:
  - **bird**: Whole bird (original bounding box)
  - **head**: Beak, crown, forehead, left eye, nape, right eye, throat
  - **eye**: Left eye, right eye
  - **beak**: Beak only

### CUB Dataset Structure

The script expects the CUB-200-2011 dataset to have the following structure:
```
CUB_200_2011/
├── images.txt
├── image_class_labels.txt
├── classes.txt
├── bounding_boxes.txt
├── train_test_split.txt
└── images/
    ├── 001.Black_footed_Albatross/
    ├── 002.Laysan_Albatross/
    └── ...
```

### Output

The script generates two COCO format JSON files:
- `cub_train.json` - Training set annotations
- `cub_test.json` - Test set annotations

Each annotation includes:
- Image metadata (dimensions, file path)
- Bounding box coordinates
- Bird species category labels
- COCO format structure for compatibility with detection frameworks

## Installation

```bash
pip install -e .
```
