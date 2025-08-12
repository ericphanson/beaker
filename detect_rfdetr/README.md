# RF-DETR Detection Project

This project contains tools for working with the RF-DETR object detection model, including dataset conversion utilities.

Data lives in `../data`

## CUB Dataset Conversion

The `convert_cub_to_coco_format.py` script converts the CUB-200-2011 bird dataset to COCO JSON format for training object detection models.

### Usage

```bash

# Convert using parts annotations for 4-class detection (bird, head, eye, beak)
uv run python convert_cub_to_coco_format.py --parts

# symlinks paths so the data is where RFDETR expects it
uv run python symlink_data.py

# train

uv run python train.py
```
