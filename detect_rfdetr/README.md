# RF-DETR Detection Project

This project contains tools for working with the RF-DETR object detection model, including dataset conversion utilities.

Data lives in `../data`

## CUB Dataset Conversion

The `convert_cub_to_coco_format.py` script converts the CUB-200-2011 bird dataset to COCO JSON format for training object detection models.

## Steps

```bash

# Convert using parts annotations for 4-class detection (bird, head, eye, beak)
uv run python convert_cub_to_coco_format.py --parts

# symlinks paths so the data is where RFDETR expects it
uv run python symlink_data.py

# train
uv run python train.py
```

Then ONNX export can be done with `./export_output/run_export.py` and quantization with `../quantizations/rfdetr.py`.

Additionally:

- `plot_class_map.py` can parse a `log.txt` from RFDETR training and generate plots
- `visualize_samples` can load data from a dataloader, and optionally an ONNX model, and plot labels/predictions
- `visualize_attention.py` can load a pytorch model and attempt to plot the deformable attention from RFDETR
- `lens.py` uses torchlens to try to visualize an RFDETR model.
- `rfdetr` contains a fork of https://github.com/roboflow/rf-detr at commit `cf066357f42ffae1d12325f3df6a09d602b849e8`, modified to add an orientation head
