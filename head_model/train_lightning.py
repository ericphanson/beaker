#!/usr/bin/env python3
"""
PyTorch Lightning-based YOLO training for bird detection with 4 classes.
Supports multiple YOLO architectures and multi-scale training for small/large objects.
"""

import os
import random
import shutil
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert, nms
from torchvision.transforms.functional import to_tensor

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Model Architecture
    "model_type": "yolov8n",  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov5n, yolov5s
    "pretrained": True,
    "pretrained_checkpoint": "yolov8n.pt",  # Path to pretrained checkpoint
    "freeze_backbone_epochs": 0,  # Freeze backbone for N epochs
    # Dataset Configuration
    "data_path": "../data/yolo-4-class/dataset.yaml",
    "num_classes": 80,  # Using COCO classes for validation
    "class_names": [
        "bird",
        "head",
        "eye",
        "beak",
    ],  # Will be overridden by COCO classes
    # Training Parameters
    # Training Parameters
    "epochs": 100,
    "batch_size": 4,  # Smaller batch size for multi-scale training
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "lr_scheduler": "cosine",  # cosine, onecycle, step
    # Multi-scale Training (important for small/large objects)
    "multiscale_training": False,  # Disable for initial testing
    "img_sizes": [416, 512, 608, 736, 832],  # Multi-scale input sizes
    "mosaic_prob": 0.0,  # Disable mosaic for initial testing
    "mixup_prob": 0.0,  # Disable mixup for initial testing
    # Data Augmentation
    "hsv_h": 0.015,  # HSV-Hue augmentation
    "hsv_s": 0.7,  # HSV-Saturation augmentation
    "hsv_v": 0.4,  # HSV-Value augmentation
    "degrees": 10.0,  # Rotation degrees
    "translate": 0.1,  # Translation ratio
    "scale": 0.5,  # Scale ratio
    "shear": 0.0,  # Shear degrees
    "flipud": 0.0,  # Vertical flip probability
    "fliplr": 0.5,  # Horizontal flip probability
    # Loss Configuration
    "box_loss_gain": 7.5,  # Box regression loss gain
    "cls_loss_gain": 0.5,  # Classification loss gain
    "obj_loss_gain": 1.0,  # Objectness loss gain
    "focal_loss_gamma": 1.5,  # Focal loss gamma (0 = CE loss)
    "label_smoothing": 0.0,  # Label smoothing epsilon
    # NMS Configuration
    "conf_threshold": 0.001,  # Confidence threshold for NMS
    "iou_threshold": 0.6,  # IoU threshold for NMS
    "max_det": 300,  # Maximum detections per image
    # Validation & Monitoring
    "val_check_interval": 1.0,  # Validate every N epochs
    "log_every_n_steps": 50,
    "save_top_k": 3,
    "monitor_metric": "val/mAP50",
    "monitor_mode": "max",
    # System Configuration
    "num_workers": 0,  # Set to 0 to avoid multiprocessing issues
    "pin_memory": True,
    "persistent_workers": False,  # Disable when num_workers=0
    "precision": "16-mixed",  # 16-mixed, 32, bf16-mixed
    # Debug Configuration
    "debug_run": True,  # Set to True for quick testing
    "debug_epochs": 2,  # Reduced epochs for debug
    "debug_fraction": 0.02,  # Use 2% of data for debug (0.02 = 2%)
    # Comet ML Configuration
    "project_name": "bird-detection-lightning",
    "experiment_name": None,  # Auto-generated if None
    "log_model": True,
    "log_predictions": True,
    "log_confusion_matrix": True,
}

# ============================================================================
# YOLO MODEL IMPLEMENTATIONS
# ============================================================================


class YOLOv8Head(nn.Module):
    """YOLOv8 detection head with DFL (Distribution Focal Loss)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: list[int] | None = None,
        reg_max: int = 16,
    ):
        if in_channels is None:
            in_channels = [64, 128, 256]
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max  # DFL regression range
        self.in_channels = in_channels

        # Box regression heads (cv2) - predicts 4 * reg_max values per anchor
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_conv(ch, 64, 3),
                    self._make_conv(64, 64, 3),
                    nn.Conv2d(64, 4 * reg_max, 1, 1, 0),
                )
                for ch in in_channels
            ]
        )

        # Classification heads (cv3) - predicts num_classes per anchor
        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_conv(ch, max(16, ch // 4, reg_max * 4), 3),
                    self._make_conv(
                        max(16, ch // 4, reg_max * 4), max(16, ch // 4, reg_max * 4), 3
                    ),
                    nn.Conv2d(max(16, ch // 4, reg_max * 4), num_classes, 1, 1, 0),
                )
                for ch in in_channels
            ]
        )

        # DFL layer for converting distribution to bbox coordinates
        self.dfl = nn.Conv2d(reg_max, 1, 1, 1, 0, bias=False)
        # Initialize DFL weights
        self.dfl.weight.data.fill_(1.0 / reg_max)

    def _make_conv(self, in_ch: int, out_ch: int, k: int) -> nn.Module:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, 1, k // 2, bias=False),
            nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through detection head."""
        outputs = []
        for i, feat in enumerate(features):
            # Box regression with DFL
            box_output = self.cv2[i](feat)
            b, _, h, w = box_output.shape
            box_output = box_output.view(b, 4, self.reg_max, h, w)

            # Apply DFL to each of the 4 bbox components
            box_final = []
            for bbox_idx in range(4):
                # Apply DFL to one bbox component at a time
                bbox_component = box_output[:, bbox_idx]  # [b, reg_max, h, w]
                bbox_coord = self.dfl(bbox_component)  # [b, 1, h, w]
                box_final.append(bbox_coord)

            box_final = torch.cat(box_final, dim=1)  # [b, 4, h, w]

            # Classification
            cls_output = self.cv3[i](feat)

            # Combine box and classification outputs
            # Format: [batch, 4 + num_classes, height, width]
            output = torch.cat([box_final, cls_output], dim=1)
            outputs.append(output)

        return outputs

    def forward_with_dfl(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass returning both final outputs and raw DFL distributions for loss computation."""
        outputs = []
        dfl_outputs = []

        for i, feat in enumerate(features):
            # Box regression with DFL
            box_output = self.cv2[i](feat)  # [b, 4*reg_max, h, w]
            b, _, h, w = box_output.shape
            box_reshaped = box_output.view(b, 4, self.reg_max, h, w)

            # Store raw DFL output for loss computation
            dfl_outputs.append(box_output)

            # Apply DFL to each of the 4 bbox components
            box_final = []
            for bbox_idx in range(4):
                # Apply DFL to one bbox component at a time
                bbox_component = box_reshaped[:, bbox_idx]  # [b, reg_max, h, w]
                bbox_coord = self.dfl(bbox_component)  # [b, 1, h, w]
                box_final.append(bbox_coord)

            box_final = torch.cat(box_final, dim=1)  # [b, 4, h, w]

            # Classification
            cls_output = self.cv3[i](feat)

            # Combine box and classification outputs
            # Format: [batch, 4 + num_classes, height, width]
            output = torch.cat([box_final, cls_output], dim=1)
            outputs.append(output)

        return outputs, dfl_outputs


class YOLOv8Model(nn.Module):
    """YOLOv8 model with pretrained backbone and custom head."""

    def __init__(self, num_classes: int = 4, pretrained_path: str | None = None):
        super().__init__()
        self.num_classes = num_classes

        # Create YOLOv8 backbone that matches the pretrained architecture
        self.backbone = self._create_yolov8_backbone()
        self.feature_channels = [64, 128, 256]  # P3, P4, P5 channels

        # Custom detection head for our classes
        self.head = YOLOv8Head(num_classes, self.feature_channels)

        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

    def _create_yolov8_backbone(self):
        """Create YOLOv8-style backbone that matches pretrained architecture."""

        class Conv(nn.Module):
            """Standard convolution block: Conv + BatchNorm + SiLU."""

            def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_ch, out_ch, k, s, p if p is not None else k // 2, bias=False
                )
                self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
                self.act = nn.SiLU(inplace=True)

            def forward(self, x):
                return self.act(self.bn(self.conv(x)))

        class Bottleneck(nn.Module):
            """Standard bottleneck block."""

            def __init__(self, in_ch, out_ch, shortcut=True, e=0.5):
                super().__init__()
                hidden_ch = int(out_ch * e)
                self.cv1 = Conv(in_ch, hidden_ch, 3, 1)
                self.cv2 = Conv(hidden_ch, out_ch, 3, 1)
                self.add = shortcut and in_ch == out_ch

            def forward(self, x):
                return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

        class C2f(nn.Module):
            """CSP Bottleneck with 2 convolutions."""

            def __init__(self, in_ch, out_ch, n=1, shortcut=False, e=0.5):
                super().__init__()
                self.c = int(out_ch * e)  # hidden channels
                self.cv1 = Conv(in_ch, 2 * self.c, 1, 1)
                self.cv2 = Conv((2 + n) * self.c, out_ch, 1)
                self.m = nn.ModuleList(
                    Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
                )

            def forward(self, x):
                y = list(self.cv1(x).chunk(2, 1))
                y.extend(m(y[-1]) for m in self.m)
                return self.cv2(torch.cat(y, 1))

        class SPPF(nn.Module):
            """Spatial Pyramid Pooling - Fast (SPPF) layer."""

            def __init__(self, in_ch, out_ch, k=5):
                super().__init__()
                c_ = in_ch // 2  # hidden channels
                self.cv1 = Conv(in_ch, c_, 1, 1)
                self.cv2 = Conv(c_ * 4, out_ch, 1, 1)
                self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

            def forward(self, x):
                x = self.cv1(x)
                y1 = self.m(x)
                y2 = self.m(y1)
                return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

        # Create the exact YOLOv8n backbone structure
        backbone = nn.ModuleDict()

        # model.0: Conv(3, 16, 3, 2) - stem
        backbone["0"] = Conv(3, 16, 3, 2)

        # model.1: Conv(16, 32, 3, 2)
        backbone["1"] = Conv(16, 32, 3, 2)

        # model.2: C2f(32, 32, 1, True)
        backbone["2"] = C2f(32, 32, 1, True)

        # model.3: Conv(32, 64, 3, 2)
        backbone["3"] = Conv(32, 64, 3, 2)

        # model.4: C2f(64, 64, 2, True)
        backbone["4"] = C2f(64, 64, 2, True)

        # model.5: Conv(64, 128, 3, 2)
        backbone["5"] = Conv(64, 128, 3, 2)

        # model.6: C2f(128, 128, 2, True)
        backbone["6"] = C2f(128, 128, 2, True)

        # model.7: Conv(128, 256, 3, 2)
        backbone["7"] = Conv(128, 256, 3, 2)

        # model.8: C2f(256, 256, 1, True)
        backbone["8"] = C2f(256, 256, 1, True)

        # model.9: SPPF(256, 256, 5)
        backbone["9"] = SPPF(256, 256, 5)

        return backbone

    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load compatible weights from pretrained YOLOv8."""
        try:
            print(f"📦 Loading pretrained YOLOv8 weights from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            pretrained_model = checkpoint["model"].float()
            pretrained_state = pretrained_model.state_dict()

            # Load compatible weights
            our_state = self.state_dict()
            loaded_count = 0
            skipped_count = 0

            print(f"📊 Pretrained model has {len(pretrained_state)} parameters")
            print(f"📊 Our model has {len(our_state)} parameters")

            # Load backbone weights (model.0 to model.9) with direct mapping
            backbone_mapping = {
                # Direct backbone layer mapping - model.0 to model.9
                "backbone.0": "model.0",
                "backbone.1": "model.1",
                "backbone.2": "model.2",
                "backbone.3": "model.3",
                "backbone.4": "model.4",
                "backbone.5": "model.5",
                "backbone.6": "model.6",
                "backbone.7": "model.7",
                "backbone.8": "model.8",
                "backbone.9": "model.9",
            }

            # Load backbone weights with direct layer mapping
            for our_backbone_key, pretrained_key in backbone_mapping.items():
                # Get all parameters for this layer
                our_layer_keys = [
                    k for k in our_state.keys() if k.startswith(our_backbone_key + ".")
                ]
                pretrained_layer_keys = [
                    k
                    for k in pretrained_state.keys()
                    if k.startswith(pretrained_key + ".")
                ]

                # Try to match parameters by their suffix (e.g., .conv.weight, .bn.bias, etc.)
                for our_key in our_layer_keys:
                    # Extract the parameter suffix (everything after the layer number)
                    our_suffix = our_key[
                        len(our_backbone_key) :
                    ]  # e.g., ".conv.weight"
                    pretrained_candidate = pretrained_key + our_suffix

                    if pretrained_candidate in pretrained_state:
                        if (
                            our_state[our_key].shape
                            == pretrained_state[pretrained_candidate].shape
                        ):
                            our_state[our_key].copy_(
                                pretrained_state[pretrained_candidate]
                            )
                            loaded_count += 1
                        else:
                            print(
                                f"⚠️ Shape mismatch: {our_key} {our_state[our_key].shape} vs {pretrained_candidate} {pretrained_state[pretrained_candidate].shape}"
                            )
                            skipped_count += 1
                    else:
                        # Try some common variations for complex modules like C2f
                        found = False
                        for pretrained_full_key in pretrained_layer_keys:
                            if pretrained_full_key.endswith(our_suffix):
                                if (
                                    our_state[our_key].shape
                                    == pretrained_state[pretrained_full_key].shape
                                ):
                                    our_state[our_key].copy_(
                                        pretrained_state[pretrained_full_key]
                                    )
                                    loaded_count += 1
                                    found = True
                                    break

                        if not found:
                            skipped_count += 1

            # Try to load some head weights (DFL layer)
            if (
                "head.dfl.weight" in our_state
                and "model.22.dfl.conv.weight" in pretrained_state
            ):
                if (
                    our_state["head.dfl.weight"].shape
                    == pretrained_state["model.22.dfl.conv.weight"].shape
                ):
                    our_state["head.dfl.weight"].copy_(
                        pretrained_state["model.22.dfl.conv.weight"]
                    )
                    loaded_count += 1
                    print("✅ Loaded DFL layer weights")
                else:
                    print(
                        f"⚠️ DFL shape mismatch: {our_state['head.dfl.weight'].shape} vs {pretrained_state['model.22.dfl.conv.weight'].shape}"
                    )
                    skipped_count += 1

            print("📈 Weight Loading Summary:")
            print(f"✅ Successfully loaded: {loaded_count} parameter tensors")
            print(f"⚠️ Skipped (incompatible): {skipped_count} parameter tensors")
            print(
                f"📊 Model coverage: {loaded_count}/{len(our_state)} ({100*loaded_count/len(our_state):.1f}%)"
            )
            print("🎯 Backbone: Fully loaded with pretrained YOLOv8 weights")
            print(
                f"🎯 Detection Head: Custom initialized for {self.num_classes} classes"
            )

        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")
            print("🔄 Using random initialization")
            import traceback

            traceback.print_exc()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through model."""
        # Extract multi-scale features
        features = self._extract_features(x)

        # Pass through custom detection head
        outputs = self.head(features)
        return outputs

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from backbone."""
        # Forward through backbone layers
        x = self.backbone["0"](x)  # model.0: stem - 16 channels, stride 2
        x = self.backbone["1"](x)  # model.1: conv - 32 channels, stride 4
        x = self.backbone["2"](x)  # model.2: C2f - 32 channels, stride 4

        x = self.backbone["3"](x)  # model.3: conv - 64 channels, stride 8
        c3 = self.backbone["4"](x)  # model.4: C2f - 64 channels, stride 8 -> P3

        x = self.backbone["5"](c3)  # model.5: conv - 128 channels, stride 16
        c4 = self.backbone["6"](x)  # model.6: C2f - 128 channels, stride 16 -> P4

        x = self.backbone["7"](c4)  # model.7: conv - 256 channels, stride 32
        x = self.backbone["8"](x)  # model.8: C2f - 256 channels, stride 32
        c5 = self.backbone["9"](x)  # model.9: SPPF - 256 channels, stride 32 -> P5

        # Return the multi-scale features
        # P3: 64 channels, 1/8 scale
        # P4: 128 channels, 1/16 scale
        # P5: 256 channels, 1/32 scale
        return [c3, c4, c5]


class YOLOModel(nn.Module):
    """Complete YOLO model - legacy wrapper for compatibility."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        # Use YOLOv8 model
        self.model = YOLOv8Model(num_classes)

        # Legacy attributes for compatibility
        self.backbone = self.model.backbone
        self.head = self.model.head

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        return self.model(x)


# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================


class YOLODataset(Dataset):
    """YOLO format dataset with multi-scale training and augmentations."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 640,
        multiscale: bool = False,
        img_sizes: list[int] | None = None,
        augment: bool = True,
        config: dict | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.multiscale = multiscale
        self.img_sizes = img_sizes or [640]
        self.augment = augment and split == "train"
        self.config = config or {}

        # Load image and label paths
        self.img_dir = self.data_dir / split / "images"
        self.label_dir = self.data_dir / split / "labels"

        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.label_paths = [self.label_dir / (p.stem + ".txt") for p in self.img_paths]

        print(f"Loaded {len(self.img_paths)} {split} images from {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a sample."""
        # Dynamic image size for multi-scale training
        if self.multiscale and self.split == "train":
            img_size = random.choice(self.img_sizes)
        else:
            img_size = self.img_size

        # Mosaic augmentation
        if self.augment and random.random() < self.config.get("mosaic_prob", 0.5):
            return self._get_mosaic_sample(idx, img_size)

        # Regular single image loading
        return self._get_single_sample(idx, img_size)

    def _get_single_sample(
        self, idx: int, img_size: int
    ) -> dict[str, torch.Tensor | str]:
        """Get a single image sample."""
        # Load image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load labels
        label_path = self.label_paths[idx]
        labels = self._load_labels(label_path)

        # Apply augmentations and preprocessing
        if self.augment:
            image, labels = self._apply_augmentations(image, labels)

        # Resize and letterbox
        image, labels, pad = self._letterbox(image, labels, img_size)

        # Convert to tensor
        image = to_tensor(image)

        # Prepare target format
        targets = self._prepare_targets(labels, img_size)

        return {
            "image": image,
            "targets": targets,
            "img_path": str(img_path),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "img_size": torch.tensor([img_size, img_size]),
        }

    def _get_mosaic_sample(
        self, idx: int, img_size: int
    ) -> dict[str, torch.Tensor | str]:
        """Get a mosaic augmented sample (4 images combined)."""
        # Select 3 additional random images
        indices = [idx] + random.choices(range(len(self.img_paths)), k=3)

        # Load 4 images and labels
        images = []
        all_labels = []

        for i, img_idx in enumerate(indices):
            img_path = self.img_paths[img_idx]
            image = Image.open(img_path).convert("RGB")

            # Load labels
            label_path = self.label_paths[img_idx]
            labels = self._load_labels(label_path)

            # Simple resize to quarter size
            quarter_size = img_size // 2
            image = image.resize((quarter_size, quarter_size), Image.Resampling.LANCZOS)

            images.append(image)

            # Adjust label coordinates for mosaic position
            if len(labels) > 0:
                labels_adj = labels.clone()
                # Scale coordinates to quarter size and offset based on quadrant
                labels_adj[:, 1] = labels_adj[:, 1] * 0.5  # x scale
                labels_adj[:, 2] = labels_adj[:, 2] * 0.5  # y scale
                labels_adj[:, 3] = labels_adj[:, 3] * 0.5  # w scale
                labels_adj[:, 4] = labels_adj[:, 4] * 0.5  # h scale

                # Offset based on quadrant position
                if i == 1:  # Top right
                    labels_adj[:, 1] += 0.5
                elif i == 2:  # Bottom left
                    labels_adj[:, 2] += 0.5
                elif i == 3:  # Bottom right
                    labels_adj[:, 1] += 0.5
                    labels_adj[:, 2] += 0.5

                all_labels.append(labels_adj)

        # Create mosaic image
        mosaic = Image.new("RGB", (img_size, img_size), (114, 114, 114))
        mosaic.paste(images[0], (0, 0))  # Top left
        mosaic.paste(images[1], (img_size // 2, 0))  # Top right
        mosaic.paste(images[2], (0, img_size // 2))  # Bottom left
        mosaic.paste(images[3], (img_size // 2, img_size // 2))  # Bottom right

        # Combine all labels
        if all_labels:
            combined_labels = torch.cat(all_labels, 0)
        else:
            combined_labels = torch.zeros((0, 5))

        # Convert to tensor
        image_tensor = to_tensor(mosaic)

        # Prepare target format
        targets = self._prepare_targets(combined_labels, img_size)

        return {
            "image": image_tensor,
            "targets": targets,
            "img_path": str(self.img_paths[idx]),  # Use primary image path
            "orig_size": torch.tensor([img_size, img_size]),
            "img_size": torch.tensor([img_size, img_size]),
        }

    def _load_labels(self, label_path: Path) -> torch.Tensor:
        """Load YOLO format labels."""
        if not label_path.exists():
            return torch.zeros((0, 5))

        with open(label_path) as f:
            lines = f.read().strip().split("\n")

        if not lines or lines == [""]:
            return torch.zeros((0, 5))

        labels = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                labels.append([class_id, x, y, w, h])

        return torch.tensor(labels, dtype=torch.float32)

    def _apply_augmentations(
        self, image: Image.Image, labels: torch.Tensor
    ) -> tuple[Image.Image, torch.Tensor]:
        """Apply data augmentations."""
        # HSV augmentation
        if random.random() < 0.5:
            image = self._hsv_augment(image)

        # Horizontal flip
        if random.random() < self.config.get("fliplr", 0.5):
            image = ImageOps.mirror(image)
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinate

        # Vertical flip (usually disabled for object detection)
        if random.random() < self.config.get("flipud", 0.0):
            image = ImageOps.flip(image)
            if len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]  # Flip y coordinate

        # Rotation (simplified)
        degrees = self.config.get("degrees", 10.0)
        if degrees > 0 and random.random() < 0.5:
            angle = random.uniform(-degrees, degrees)
            image = image.rotate(angle, expand=False, fillcolor=(114, 114, 114))
            # Note: Would need to rotate bounding boxes too for proper implementation

        # Translation and scaling would be implemented in letterbox or separate functions
        # For now, we handle them in the multi-scale training

        return image, labels

    def _hsv_augment(self, image: Image.Image) -> Image.Image:
        """Apply HSV augmentation."""
        import cv2

        img_array = np.array(image)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Augmentation gains
        h_gain = self.config.get("hsv_h", 0.015)
        s_gain = self.config.get("hsv_s", 0.7)
        v_gain = self.config.get("hsv_v", 0.4)

        # Apply random gains
        gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        img_hsv[..., 0] = (img_hsv[..., 0] * gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * gains[2], 0, 255)

        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_rgb)

    def _letterbox(
        self, image: Image.Image, labels: torch.Tensor, size: int
    ) -> tuple[Image.Image, torch.Tensor, tuple[int, int]]:
        """Letterbox resize while maintaining aspect ratio."""
        w, h = image.size

        # Calculate scaling
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create padded image
        padded = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        padded.paste(image, (pad_x, pad_y))

        # Adjust labels
        if len(labels) > 0:
            # Scale and translate coordinates
            labels[:, 1] = labels[:, 1] * scale * w / size + pad_x / size
            labels[:, 2] = labels[:, 2] * scale * h / size + pad_y / size
            labels[:, 3] = labels[:, 3] * scale * w / size
            labels[:, 4] = labels[:, 4] * scale * h / size

        return padded, labels, (pad_x, pad_y)

    def _prepare_targets(self, labels: torch.Tensor, img_size: int) -> torch.Tensor:
        """Prepare targets for training."""
        if len(labels) == 0:
            return torch.zeros((0, 6))  # batch_idx, class, x, y, w, h

        # Add batch index (will be filled by collate_fn)
        targets = torch.zeros((len(labels), 6))
        targets[:, 1:] = labels  # class, x, y, w, h

        return targets


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    """Custom collate function for YOLO data with MixUp support."""
    # Handle variable image sizes by finding the max size and padding
    max_size = max(item["img_size"][0].item() for item in batch)

    # Pad all images to the same size
    images = []
    for item in batch:
        img = item["image"]
        current_size = img.shape[-1]  # Assuming square images

        if current_size != max_size:
            # Pad image to max_size
            pad = (max_size - current_size) // 2
            img = F.pad(img, (pad, pad, pad, pad), value=0.114)  # Gray padding

        images.append(img)

    images = torch.stack(images)

    # Combine targets and add batch indices
    targets = []
    for i, item in enumerate(batch):
        target = item["targets"]
        if len(target) > 0:
            target[:, 0] = i  # Set batch index
            targets.append(target)

    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 6))

    return {
        "images": images,
        "targets": targets,
        "img_paths": [item["img_path"] for item in batch],
        "orig_sizes": torch.stack([item["orig_size"] for item in batch]),
        "img_sizes": torch.stack([item["img_size"] for item in batch]),
    }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class YOLOv8Loss(nn.Module):
    """YOLOv8 loss function with distribution focal loss (DFL) and anchor-free design."""

    def __init__(self, num_classes: int, config: dict):
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        # Loss gains
        self.box_gain = config.get("box_loss_gain", 7.5)
        self.cls_gain = config.get("cls_loss_gain", 0.5)
        self.dfl_gain = config.get("dfl_loss_gain", 1.5)  # DFL loss weight

        # DFL parameters
        self.reg_max = 16  # DFL regression range

        # BCE loss for classification
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")

        # Grid strides for each scale
        self.strides = torch.tensor([8.0, 16.0, 32.0])

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        dfl_predictions: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute YOLOv8 loss with optional DFL loss."""
        device = predictions[0].device
        self.strides = self.strides.to(device)

        # Initialize losses
        loss_box = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)

        # Build targets for all scales
        target_boxes, target_classes, target_indices = self._build_targets(
            predictions, targets, device
        )

        # Process each scale
        for i, pred in enumerate(predictions):
            # pred shape: [batch, 4 + num_classes, h, w]
            b, c, h, w = pred.shape

            # Separate box and class predictions
            pred_boxes = pred[:, :4]  # [batch, 4, h, w]
            pred_classes = pred[:, 4:]  # [batch, num_classes, h, w]

            # Reshape for processing
            pred_boxes = (
                pred_boxes.permute(0, 2, 3, 1).contiguous().view(-1, 4)
            )  # [batch*h*w, 4]
            pred_classes = (
                pred_classes.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            )  # [batch*h*w, nc]

            # Get targets for this scale
            if i < len(target_indices):
                indices = target_indices[i]
                if len(indices) > 0:
                    # Get positive samples
                    t_boxes = target_boxes[i]
                    t_classes = target_classes[i]

                    # Classification loss
                    if len(t_classes) > 0:
                        cls_targets = torch.zeros_like(pred_classes)
                        cls_targets[indices, t_classes] = 1.0
                        cls_loss = self.bce_cls(pred_classes, cls_targets).mean()
                        loss_cls += cls_loss * self.cls_gain

                    # Box regression loss (IoU loss)
                    if len(t_boxes) > 0:
                        pred_boxes_pos = pred_boxes[indices]
                        iou = self._bbox_iou(pred_boxes_pos, t_boxes, x1y1x2y2=False)
                        box_loss = (1.0 - iou).mean()
                        loss_box += box_loss * self.box_gain

                    # DFL loss (if DFL predictions are provided)
                    if (
                        dfl_predictions is not None
                        and i < len(dfl_predictions)
                        and len(t_boxes) > 0
                    ):
                        dfl_pred = dfl_predictions[i]  # [batch, 4*reg_max, h, w]
                        dfl_loss = self._compute_dfl_loss(
                            dfl_pred, t_boxes, indices, h, w, i
                        )
                        loss_dfl += dfl_loss * self.dfl_gain

        # Total loss
        loss_total = loss_box + loss_cls + loss_dfl

        return {
            "loss": loss_total,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_dfl": loss_dfl,
        }

    def _compute_dfl_loss(
        self,
        dfl_pred: torch.Tensor,
        target_boxes: torch.Tensor,
        indices: torch.Tensor,
        h: int,
        w: int,
        scale_idx: int,
    ) -> torch.Tensor:
        """Compute Distribution Focal Loss."""
        device = dfl_pred.device

        # Reshape DFL predictions: [batch, 4*reg_max, h, w] -> [batch*h*w, 4, reg_max]
        dfl_pred = dfl_pred.permute(0, 2, 3, 1).contiguous()  # [batch, h, w, 4*reg_max]
        dfl_pred = dfl_pred.view(-1, 4, self.reg_max)  # [batch*h*w, 4, reg_max]

        # Get positive predictions
        if len(indices) == 0:
            return torch.zeros(1, device=device)

        dfl_pred_pos = dfl_pred[indices]  # [num_pos, 4, reg_max]

        # Convert target boxes to DFL targets
        # target_boxes are in normalized xywh format, need to convert to grid coordinates
        target_boxes_scaled = target_boxes.clone()
        target_boxes_scaled[:, [0, 2]] *= w  # x, w scaled to grid width
        target_boxes_scaled[:, [1, 3]] *= h  # y, h scaled to grid height

        # Convert from center format to corner format for DFL
        # xywh -> x1y1x2y2 (in grid coordinates)
        x1 = target_boxes_scaled[:, 0] - target_boxes_scaled[:, 2] / 2
        y1 = target_boxes_scaled[:, 1] - target_boxes_scaled[:, 3] / 2
        x2 = target_boxes_scaled[:, 0] + target_boxes_scaled[:, 2] / 2
        y2 = target_boxes_scaled[:, 1] + target_boxes_scaled[:, 3] / 2

        # Stack to get [num_pos, 4] in x1y1x2y2 format
        target_corners = torch.stack([x1, y1, x2, y2], dim=1)

        # Convert corner coordinates to DFL distribution targets
        dfl_targets = []
        for corner_idx in range(4):
            coord = target_corners[:, corner_idx]  # [num_pos]

            # Clamp coordinates to valid range
            coord = torch.clamp(coord, 0, self.reg_max - 1)

            # Create distribution targets (simple approach: one-hot at rounded coordinate)
            coord_int = coord.round().long()
            coord_int = torch.clamp(coord_int, 0, self.reg_max - 1)

            # Create one-hot targets
            targets_onehot = torch.zeros(len(coord_int), self.reg_max, device=device)
            targets_onehot.scatter_(1, coord_int.unsqueeze(1), 1.0)
            dfl_targets.append(targets_onehot)

        # Stack targets: [4, num_pos, reg_max] -> [num_pos, 4, reg_max]
        dfl_targets = torch.stack(dfl_targets, dim=1)  # [num_pos, 4, reg_max]

        # Compute cross entropy loss between predictions and targets
        dfl_pred_flat = dfl_pred_pos.view(-1, self.reg_max)  # [num_pos*4, reg_max]
        dfl_targets_flat = dfl_targets.view(-1, self.reg_max)  # [num_pos*4, reg_max]

        # Apply softmax to predictions and compute cross entropy
        dfl_loss = F.cross_entropy(
            dfl_pred_flat, dfl_targets_flat.argmax(dim=1), reduction="mean"
        )

        return dfl_loss

    def _build_targets(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        device: torch.device,
    ):
        """Build targets for anchor-free YOLOv8."""
        target_boxes = []
        target_classes = []
        target_indices = []

        for scale_idx, pred in enumerate(predictions):
            b, c, h, w = pred.shape
            stride = self.strides[scale_idx]

            # Initialize for this scale
            scale_target_boxes = []
            scale_target_classes = []
            scale_indices = []

            if len(targets) > 0:
                # Scale targets to current grid
                gt = targets.clone()
                gt[:, 2:6] *= torch.tensor([w, h, w, h], device=device) / stride

                for gt_box in gt:
                    batch_idx = int(gt_box[0])
                    cls = int(gt_box[1])
                    gx, gy, gw, gh = gt_box[2:6]

                    # Check if target center is within grid bounds
                    if 0 <= gx < w and 0 <= gy < h:
                        gi, gj = int(gx), int(gy)

                        # Add to targets
                        flat_idx = batch_idx * h * w + gj * w + gi
                        scale_indices.append(flat_idx)
                        scale_target_classes.append(cls)
                        scale_target_boxes.append([gx - gi, gy - gj, gw, gh])

            # Convert to tensors
            target_boxes.append(
                torch.tensor(scale_target_boxes, device=device, dtype=torch.float32)
                if scale_target_boxes
                else torch.zeros((0, 4), device=device)
            )
            target_classes.append(
                torch.tensor(scale_target_classes, device=device, dtype=torch.long)
                if scale_target_classes
                else torch.zeros(0, device=device, dtype=torch.long)
            )
            target_indices.append(
                torch.tensor(scale_indices, device=device, dtype=torch.long)
                if scale_indices
                else torch.zeros(0, device=device, dtype=torch.long)
            )

        return target_boxes, target_classes, target_indices

    def _bbox_iou(
        self,
        box1: torch.Tensor,
        box2: torch.Tensor,
        x1y1x2y2: bool = True,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        if x1y1x2y2:
            # Box format: x1, y1, x2, y2
            b1_x1, b1_y1, b1_x2, b1_y2 = (
                box1[..., 0],
                box1[..., 1],
                box1[..., 2],
                box1[..., 3],
            )
            b2_x1, b2_y1, b2_x2, b2_y2 = (
                box2[..., 0],
                box2[..., 1],
                box2[..., 2],
                box2[..., 3],
            )
        else:
            # Box format: x, y, w, h
            b1_x1, b1_x2 = (
                box1[..., 0] - box1[..., 2] / 2,
                box1[..., 0] + box1[..., 2] / 2,
            )
            b1_y1, b1_y2 = (
                box1[..., 1] - box1[..., 3] / 2,
                box1[..., 1] + box1[..., 3] / 2,
            )
            b2_x1, b2_x2 = (
                box2[..., 0] - box2[..., 2] / 2,
                box2[..., 0] + box2[..., 2] / 2,
            )
            b2_y1, b2_y2 = (
                box2[..., 1] - box2[..., 3] / 2,
                box2[..., 1] + box2[..., 3] / 2,
            )

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
        ).clamp(0)

        # Union area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        return inter / union


# ============================================================================
# LIGHTNING MODULE
# ============================================================================


class YOLOLightningModule(L.LightningModule):
    """Lightning module for YOLO training."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Model
        pretrained_path = (
            config.get("pretrained_checkpoint")
            if config.get("pretrained", True)
            else None
        )

        # Use YOLOv8Model directly for DFL support
        self.model = YOLOv8Model(config["num_classes"], pretrained_path)

        # Loss
        self.criterion = YOLOv8Loss(config["num_classes"], config)

        # Metrics - mAP calculation
        self.val_map = MeanAveragePrecision(
            box_format="cxcywh",  # YOLO format
            iou_type="bbox",
            class_metrics=True,
        )

        # Freeze backbone if specified
        self.freeze_epochs = config.get("freeze_backbone_epochs", 0)
        if self.freeze_epochs > 0:
            self._freeze_backbone()

        # Automatic optimization
        self.automatic_optimization = True

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if self.model.backbone is not None:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print(f"🧊 Frozen backbone for {self.freeze_epochs} epochs")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        if self.model.backbone is not None:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            print("🔥 Unfroze backbone parameters")

    def _postprocess_predictions(
        self, predictions: list[torch.Tensor], img_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLOv8 predictions with NMS."""
        batch_size = img_sizes.shape[0]
        processed_predictions = []

        for batch_idx in range(batch_size):
            # Collect all detections for this image
            all_detections = []

            # Process each scale
            for scale_idx, pred in enumerate(predictions):
                # pred shape: [batch, 4 + num_classes, h, w]
                pred_batch = pred[batch_idx]  # [4 + num_classes, h, w]

                # Extract boxes and classes
                pred_boxes = pred_batch[:4]  # [4, h, w]
                pred_classes = pred_batch[4:]  # [num_classes, h, w]

                # Reshape to [h*w, 4] and [h*w, num_classes]
                h, w = pred_boxes.shape[1], pred_boxes.shape[2]
                boxes = pred_boxes.permute(1, 2, 0).contiguous().view(-1, 4)  # [h*w, 4]
                classes = (
                    pred_classes.permute(1, 2, 0)
                    .contiguous()
                    .view(-1, self.config["num_classes"])
                )  # [h*w, nc]

                # Get class scores and labels
                class_scores, class_labels = torch.max(torch.sigmoid(classes), dim=1)

                # Create grid coordinates for center point calculation
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h, device=pred.device),
                    torch.arange(w, device=pred.device),
                    indexing="ij",
                )
                grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).float()

                # Convert box predictions to actual coordinates
                # YOLOv8 predicts offsets from grid cells
                stride = [8, 16, 32][scale_idx]
                centers = (grid + 0.5) * stride  # Grid center points
                boxes[:, :2] = (
                    centers + boxes[:, :2] * stride
                )  # Apply predicted offsets
                boxes[:, 2:] = (
                    torch.exp(boxes[:, 2:]) * stride
                )  # Convert to width/height

                # Apply confidence threshold
                conf_mask = class_scores > self.config["conf_threshold"]

                if conf_mask.any():
                    filtered_boxes = boxes[conf_mask]
                    filtered_scores = class_scores[conf_mask]
                    filtered_labels = class_labels[conf_mask]

                    # Convert to detection format [x, y, w, h, score, class]
                    detections = torch.cat(
                        [
                            filtered_boxes,
                            filtered_scores.unsqueeze(1),
                            filtered_labels.unsqueeze(1).float(),
                        ],
                        dim=1,
                    )

                    all_detections.append(detections)

            if not all_detections:
                # No detections
                processed_predictions.append(
                    {
                        "boxes": torch.zeros((0, 4), device=predictions[0].device),
                        "scores": torch.zeros(0, device=predictions[0].device),
                        "labels": torch.zeros(
                            0, dtype=torch.long, device=predictions[0].device
                        ),
                    }
                )
                continue

            # Concatenate all detections
            all_dets = torch.cat(all_detections, dim=0)

            # Extract components
            boxes = all_dets[:, :4]  # x, y, w, h (center format)
            scores = all_dets[:, 4]
            labels = all_dets[:, 5].long()

            # Convert to corner format for NMS and normalize
            img_size = img_sizes[batch_idx][0].item()
            boxes_corner = box_convert(boxes, "cxcywh", "xyxy")
            boxes_corner = boxes_corner / img_size  # Normalize to [0, 1]

            # Apply NMS
            keep_indices = nms(
                boxes_corner * img_size, scores, self.config["iou_threshold"]
            )

            # Limit detections and convert back
            keep_indices = keep_indices[: self.config["max_det"]]
            final_boxes = box_convert(boxes_corner[keep_indices], "xyxy", "cxcywh")

            processed_predictions.append(
                {
                    "boxes": final_boxes,
                    "scores": scores[keep_indices],
                    "labels": labels[keep_indices],
                }
            )

        return processed_predictions

    def _prepare_targets_for_map(
        self, targets: torch.Tensor, batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare ground truth targets for mAP calculation."""
        target_list = []

        for batch_idx in range(batch_size):
            # Filter targets for this batch
            batch_mask = targets[:, 0] == batch_idx
            batch_targets = targets[batch_mask]

            if len(batch_targets) == 0:
                # No targets for this image
                target_list.append(
                    {
                        "boxes": torch.zeros((0, 4), device=targets.device),
                        "labels": torch.zeros(
                            0, dtype=torch.long, device=targets.device
                        ),
                    }
                )
                continue

            # Extract boxes and labels
            # targets format: [batch_idx, class, x, y, w, h]
            boxes = batch_targets[:, 2:6]  # x, y, w, h (center format)
            labels = batch_targets[:, 1].long()  # class labels

            target_list.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        return target_list

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Unfreeze backbone after specified epochs
        if self.current_epoch == self.freeze_epochs and self.freeze_epochs > 0:
            self._unfreeze_backbone()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step with DFL loss computation."""
        # Forward pass with DFL outputs
        if hasattr(self.model.head, "forward_with_dfl"):
            predictions, dfl_predictions = self.model.head.forward_with_dfl(
                self.model._extract_features(batch["images"])
            )
            loss_dict = self.criterion(predictions, batch["targets"], dfl_predictions)
        else:
            predictions = self(batch["images"])
            loss_dict = self.criterion(predictions, batch["targets"])

        # Log all losses at once
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss_dict["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        predictions = self(batch["images"])

        # Compute loss
        loss_dict = self.criterion(predictions, batch["targets"])

        # Log losses
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Update mAP metric
        processed_preds = self._postprocess_predictions(predictions, batch["img_sizes"])
        target_list = self._prepare_targets_for_map(
            batch["targets"], batch["images"].shape[0]
        )
        self.val_map.update(processed_preds, target_list)

        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        map_dict = self.val_map.compute()

        # Log mAP metrics with automatic handling
        metrics_to_log = {
            "val/mAP50": map_dict["map_50"],
            "val/mAP50-95": map_dict["map"],
        }

        # Add per-class mAP if available
        if "map_per_class" in map_dict and len(self.config["class_names"]) > 1:
            for i, class_name in enumerate(self.config["class_names"]):
                if i < len(map_dict["map_per_class"]):
                    metrics_to_log[f"val/mAP50_{class_name}"] = map_dict[
                        "map_per_class"
                    ][i]

        self.log_dict(metrics_to_log, prog_bar=True, sync_dist=True)
        self.val_map.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Simplified scheduler configuration
        scheduler_config = {
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["epochs"]
            ),
            "onecycle": lambda: torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=int(self.trainer.estimated_stepping_batches),
            ),
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            ),
        }

        scheduler = scheduler_config.get(
            self.config["lr_scheduler"], scheduler_config["step"]
        )()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# DATA MODULE
# ============================================================================


class YOLODataModule(L.LightningDataModule):
    """Lightning data module for YOLO dataset."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Load dataset configuration
        with open(config["data_path"]) as f:
            self.data_config = yaml.safe_load(f)

        self.data_dir = Path(self.data_config["path"])
        self.num_classes = self.data_config["nc"]
        self.class_names = self.data_config["names"]

    def setup(self, stage: str | None = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = YOLODataset(
                data_dir=str(self.data_dir),
                split="train",
                img_size=self.config["img_sizes"][0],  # Base size
                multiscale=self.config["multiscale_training"],
                img_sizes=self.config["img_sizes"],
                augment=True,
                config=self.config,
            )

            # Validation dataset
            self.val_dataset = YOLODataset(
                data_dir=str(self.data_dir),
                split="val",
                img_size=self.config["img_sizes"][0],
                multiscale=False,
                augment=False,
                config=self.config,
            )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=self.config["persistent_workers"]
            and self.config["num_workers"] > 0,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=self.config["persistent_workers"]
            and self.config["num_workers"] > 0,
            collate_fn=collate_fn,
        )


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def create_debug_dataset(config: dict) -> str:
    """Create debug dataset for quick testing."""
    debug_dir = Path("../data/yolo-4-class-debug")

    # Remove existing debug directory
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
        print(f"🗑️ Removed existing debug directory: {debug_dir}")

    debug_dir.mkdir(exist_ok=True)

    # Create debug directories
    for split in ["train", "val"]:
        (debug_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (debug_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy subset of files
    original_dir = Path("../data/yolo-4-class")
    fraction = config["debug_fraction"]

    for split in ["train", "val"]:
        original_images = list((original_dir / split / "images").glob("*.jpg"))
        sample_size = max(1, int(len(original_images) * fraction))
        sampled_images = random.sample(original_images, sample_size)

        print(
            f"📊 Debug: Using {len(sampled_images)}/{len(original_images)} {split} images ({fraction * 100:.1f}%)"
        )

        for img_path in sampled_images:
            # Copy image
            dst_img = debug_dir / split / "images" / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = original_dir / split / "labels" / label_name
            dst_label = debug_dir / split / "labels" / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

    # Create debug dataset.yaml
    debug_yaml = {
        "path": str(debug_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": 4,
        "names": ["bird", "head", "eye", "beak"],
    }

    yaml_path = debug_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(debug_yaml, f, default_flow_style=False)

    return str(yaml_path)


@rank_zero_only
def setup_comet_logger(config: dict) -> CometLogger | None:
    """Setup Comet ML logger."""
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("⚠️ COMET_API_KEY not found. Training without experiment tracking.")
        return None

    project_name = os.getenv("COMET_PROJECT_NAME", config["project_name"])
    workspace = os.getenv("COMET_WORKSPACE")

    try:
        logger = CometLogger(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            experiment_name=config.get("experiment_name"),
        )

        # Log hyperparameters
        logger.log_hyperparams(config)

        print(f"✅ Comet ML experiment started: {logger.experiment.url}")
        return logger

    except Exception as e:
        print(f"❌ Failed to initialize Comet ML: {e}")
        return None


def main():
    """Main training function."""
    # Set random seeds
    L.seed_everything(42)

    # Check device
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA not available, using CPU")

    # Configure for debug or full training
    config = TRAINING_CONFIG.copy()

    if config["debug_run"]:
        print("🐛 DEBUG MODE: Quick training run enabled")
        print(f"   - Epochs: {config['debug_epochs']} (vs {config['epochs']} normal)")
        print(f"   - Data subset: {config['debug_fraction'] * 100:.1f}% of dataset")

        # Create debug dataset and update config
        debug_data_path = create_debug_dataset(config)
        config["data_path"] = debug_data_path
        config["epochs"] = config["debug_epochs"]
        config["experiment_name"] = (
            f"{config.get('experiment_name', 'yolo-debug')}_debug"
        )
    else:
        print("🚀 FULL TRAINING MODE")
        config["experiment_name"] = config.get("experiment_name", "yolo-full")

    # Setup Comet logger
    logger = setup_comet_logger(config)

    # Create data module
    data_module = YOLODataModule(config)

    # Create model
    model = YOLOLightningModule(config)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=config["monitor_metric"],
            mode=config["monitor_mode"],
            save_top_k=config["save_top_k"],
            filename="yolo-{epoch:02d}-{val/mAP50:.3f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=config["monitor_metric"],
            mode=config["monitor_mode"],
            patience=20,
            verbose=True,
        ),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision=config["precision"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config["log_every_n_steps"],
        val_check_interval=config["val_check_interval"],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train model
    print("🚀 Starting training...")
    trainer.fit(model, data_module)

    # Log final results
    if logger:
        try:
            print("📤 Training logs uploaded to Comet ML")
            print("✅ Training completed successfully!")

        except Exception as e:
            print(f"⚠️ Error logging final results: {e}")

    print("🎉 Training finished! Check lightning_logs/ for checkpoints.")


if __name__ == "__main__":
    main()
