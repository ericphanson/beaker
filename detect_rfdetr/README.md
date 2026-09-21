# RF-DETR Detection Model

This directory trains Beaker's detection model: RF-DETR Medium, pretrained on COCO and fine-tuned on CUB-200-2011. The model detects four classes (bird, head, eye, beak) and predicts an orientation angle for bird and head boxes.

The released model is `bird-orientation-detector-v1.0.0` (see `beaker/src/detection.rs`). It came from the run in `output5/`.

## Recipe

### 1. Data

Download CUB-200-2011 from https://www.vision.caltech.edu/datasets/cub_200_2011/ and extract it to `../data/CUB_200_2011`. The conversion needs `images/`, `images.txt`, `image_class_labels.txt`, `classes.txt`, `bounding_boxes.txt`, `train_test_split.txt` and `parts/part_locs.txt`. `data/` is gitignored.

`convert_cub_to_coco_format.py` writes COCO JSON to `../data/coco_annotations/`. It builds each class's box from CUB's annotations. CUB gives one bird box per image and 15 part keypoints, each marked visible or not.

| Class | Box | Orientation target |
|---|---|---|
| 0 bird | CUB bird box | angle from crown to tail |
| 1 head | box around visible beak, crown, forehead, eyes, nape and throat keypoints, plus 10 px | angle from eye centre to beak |
| 2 eye | box around visible eye keypoints, plus 10 px | none |
| 3 beak | box around the beak keypoint, plus 10 px | none |

Angles are `atan2(dy, dx)` in image coordinates (y down), stored in an extra `orient` field on each annotation. A box or angle is omitted when a required keypoint is not visible.

`symlink_data.py` builds `../data/cub_coco_parts/{train,valid,test}` in the layout RF-DETR expects. Training uses CUB's official train split (5,994 images). CUB's test split (5,794 images) is shuffled with seed 42 and split in half into `valid` and `test`.

### 2. Model changes

`rfdetr/` is a fork of https://github.com/roboflow/rf-detr at commit `cf066357f42ffae1d12325f3df6a09d602b849e8`, with these changes:

- `orient_embed` (`models/lwdetr.py`): a 3-layer MLP that predicts (cos, sin) per query.
- `loss_orient` (`models/lwdetr.py`): `1 − cos_sim` between the normalized prediction and the target unit vector. It is computed only for queries matched to a ground-truth box with an angle label, on the final and auxiliary decoder layers (not on the two-stage encoder outputs). Weight: `orient_loss_coef=1`.
- Augmentations update angles (`datasets/transforms.py`): horizontal flip maps θ to π−θ; anisotropic resize rescales (cos, sin) and recomputes θ.
- ONNX export has a third output, `orients`.

### 3. Training

`train.py` loads `RFDETRMedium` with `num_queries=100`, which downloads COCO weights (`rf-detr-medium.pth`) from Roboflow's storage. `reinitialize_detection_head(5)` replaces the 91-way COCO classifier with a 5-way one (4 classes plus one extra index). The orientation head is not in the pretrained checkpoint and starts from random initialization.

All other settings are the fork's `TrainConfig` defaults (`rfdetr/src/rfdetr/config.py`): learning rate 1e-4 (encoder 1.5e-4, layer decay 0.8), batch size 4 with 4 gradient-accumulation steps, EMA decay 0.993, multi-scale training around resolution 576, 100 epochs, early stopping off.

`train.py` writes to `output5/` and refuses to overwrite it without `--force`. It logs to Comet and reads `COMET_API_KEY`, `COMET_WORKSPACE` and `COMET_PROJECT_NAME` from the environment (`.envrc` sets the last two).

The `output5` run was done on a CUDA machine. Its `log.txt` has 28 epochs (0–27), not 100: the run stopped early with no record of why. The best regular checkpoint is from epoch 6 and the best EMA checkpoint from epoch 4. EMA mAP@50:95 reached about 0.56 by epoch 4 and stayed between 0.53 and 0.56 after that.

### 4. Export and quantization

`onnx_export/run_export.py` loads `output5/checkpoint_best_regular.pth`, exports ONNX with outputs `dets`, `labels`, `orients`, simplifies it, and checks that ONNX Runtime can run it. It writes `onnx_export/export_output/inference_model.sim.onnx`.

`../quantizations/rfdetr.py` copies that file and applies dynamic INT8 quantization. The quantizer names its output `unknown-dynamic-int8.onnx` for this model; the released file was renamed to `rfdetr-medium-dynamic-int8.onnx`.

`create_release.sh` creates a GitHub release and uploads the files in a folder.

## Steps

```bash
# Convert CUB annotations to COCO format (4 classes with orientation)
uv run python convert_cub_to_coco_format.py

# Build train/valid/test directories for RF-DETR
uv run python symlink_data.py

# Train (requires Comet credentials; writes output5/)
uv run python train.py

# Export to ONNX
cd onnx_export && uv run python run_export.py && cd ..

# Quantize
cd ../quantizations && uv run python rfdetr.py
```

## Reproducibility

The scripts reproduce the recipe but not the exact weights:

- CUB-200-2011 and the pretrained RF-DETR Medium weights are external downloads.
- `train.py` runs 100 epochs; the released run stopped after 28. Checkpoint selection picked epoch 6, so a full run should give a model of similar quality.
- The fork sets a seed, but CUDA nondeterminism and different hardware will change the weights.
- Training was only run on CUDA.

## Other tools

- `train_debug.py`: 1-epoch RF-DETR Nano run on the same data, for testing the pipeline.
- `plot_class_map.py`: plots per-class mAP from a training `log.txt`.
- `visualize_samples.py`: plots labels and, optionally, ONNX predictions for samples from a dataloader.
- `visualize_attention.py`: plots deformable attention from a PyTorch checkpoint.
- `lens.py`: visualizes the model with torchlens.
- `pull_runs.sh`: rsyncs an output directory from the training machine.
