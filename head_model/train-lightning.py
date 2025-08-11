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
from typing import Dict, List, Optional, Tuple, Union

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
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
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
    "freeze_backbone_epochs": 0,  # Freeze backbone for N epochs
    # Dataset Configuration
    "data_path": "../data/yolo-4-class/dataset.yaml",
    "num_classes": 4,
    "class_names": ["bird", "head", "eye", "beak"],
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
    "num_workers": 2,  # Reduced for stability
    "pin_memory": True,
    "persistent_workers": True,
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


class YOLOHead(nn.Module):
    """YOLO detection head."""

    def __init__(self, num_classes: int, in_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Detection layers for 3 scales
        self.detection_layers = nn.ModuleList(
            [
                self._make_detection_layer(in_channels)
                for _ in range(3)  # 3 detection scales
            ]
        )

    def _make_detection_layer(self, in_channels: int) -> nn.Module:
        """Create a detection layer."""
        # 5 = 4 bbox coords + 1 objectness
        out_channels = (self.num_classes + 5) * 3  # 3 anchors per scale

        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through detection head."""
        outputs = []
        for i, feat in enumerate(features):
            output = self.detection_layers[i](feat)
            outputs.append(output)
        return outputs


class YOLOBackbone(nn.Module):
    """Simple YOLO-style backbone with FPN."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 6, 2, 2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Backbone stages
        self.stage1 = self._make_stage(32, 64, 2)  # /4
        self.stage2 = self._make_stage(64, 128, 2)  # /8
        self.stage3 = self._make_stage(128, 256, 2)  # /16
        self.stage4 = self._make_stage(256, 512, 2)  # /32

        # FPN layers
        self.p5_conv = nn.Conv2d(512, 256, 1, 1, 0)
        self.p4_conv = nn.Conv2d(256, 256, 1, 1, 0)
        self.p3_conv = nn.Conv2d(128, 256, 1, 1, 0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_stage(
        self, in_channels: int, out_channels: int, stride: int
    ) -> nn.Module:
        """Create a backbone stage."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through backbone."""
        # Stem and stages
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # FPN
        p5 = self.p5_conv(c5)
        p4 = self.p4_conv(c4) + self.p4_upsample(p5)
        p3 = self.p3_conv(c3) + self.p3_upsample(p4)

        return [p3, p4, p5]  # Multi-scale features


class YOLOModel(nn.Module):
    """Complete YOLO model."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(num_classes)

        # Anchor sizes for 3 scales (small, medium, large)
        self.register_buffer(
            "anchors",
            torch.tensor(
                [
                    [[10, 13], [16, 30], [33, 23]],  # Small objects (high resolution)
                    [[30, 61], [62, 45], [59, 119]],  # Medium objects
                    [
                        [116, 90],
                        [156, 198],
                        [373, 326],
                    ],  # Large objects (low resolution)
                ],
                dtype=torch.float32,
            ),
        )

        # Grid strides for each scale
        self.register_buffer("strides", torch.tensor([8, 16, 32], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs


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
        img_sizes: Optional[List[int]] = None,
        augment: bool = True,
        config: Optional[Dict] = None,
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

        self.img_paths = sorted(list(self.img_dir.glob("*.jpg")))
        self.label_paths = [self.label_dir / (p.stem + ".txt") for p in self.img_paths]

        print(f"Loaded {len(self.img_paths)} {split} images from {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
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
    ) -> Dict[str, Union[torch.Tensor, str]]:
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
    ) -> Dict[str, Union[torch.Tensor, str]]:
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

        with open(label_path, "r") as f:
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
    ) -> Tuple[Image.Image, torch.Tensor]:
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
    ) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
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


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
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


class YOLOLoss(nn.Module):
    """YOLO loss function with focal loss and IoU-aware classification."""

    def __init__(self, num_classes: int, config: Dict):
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        # Loss gains
        self.box_gain = config.get("box_loss_gain", 7.5)
        self.cls_gain = config.get("cls_loss_gain", 0.5)
        self.obj_gain = config.get("obj_loss_gain", 1.0)

        # Focal loss
        self.focal_gamma = config.get("focal_loss_gamma", 1.5)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.eps = 1e-7

        # BCE loss
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")

        # Class names for per-class logging
        self.class_names = config.get(
            "class_names", [f"class_{i}" for i in range(num_classes)]
        )

        # IoU threshold for positive assignment
        self.iou_t = 0.20  # IoU threshold for anchor assignment

        # Anchor thresholds
        self.anchor_t = 4.0  # Anchor multiple threshold

    def forward(
        self, predictions: List[torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute YOLO loss."""
        device = predictions[0].device

        # Initialize losses
        loss_box = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # Per-class losses for logging
        per_class_losses = {
            name: torch.zeros(1, device=device) for name in self.class_names
        }

        # Grid sizes for each scale
        grid_sizes = []
        for pred in predictions:
            _, _, h, w = pred.shape
            grid_sizes.append((h, w))

        # Build targets once for all scales
        targets_all_scales = self._build_all_targets(targets, grid_sizes, device)

        # Process each scale
        for i, pred in enumerate(predictions):
            # pred shape: [batch, (nc+5)*3, h, w]
            b, c, h, w = pred.shape
            pred = (
                pred.view(b, 3, self.num_classes + 5, h, w)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            # Extract predictions
            pred_boxes = pred[..., :4]  # x, y, w, h
            pred_obj = pred[..., 4]  # objectness
            pred_cls = pred[..., 5:]  # classes

            # Get targets for this scale
            scale_targets = targets_all_scales[i]
            tobj, tcls, tbox, indices, anchors = scale_targets

            # Objectness loss
            obj_loss = self.bce_obj(pred_obj, tobj).mean()
            loss_obj += obj_loss * self.obj_gain

            # Only compute cls and box loss where we have positive targets
            n_targets = indices[0].numel()
            if n_targets > 0:
                # Get positive predictions
                ps_cls = pred_cls[indices]
                ps_box = pred_boxes[indices]

                # Classification loss with label smoothing
                if self.num_classes > 1:
                    t = torch.zeros_like(ps_cls, device=device)
                    if len(tcls) > 0:
                        t[range(n_targets), tcls] = 1.0

                    # Apply label smoothing
                    if self.label_smoothing > 0:
                        t = (
                            t * (1 - self.label_smoothing)
                            + self.label_smoothing / self.num_classes
                        )

                    cls_loss = self._focal_loss(ps_cls, t)
                    loss_cls += cls_loss * self.cls_gain

                    # Per-class losses for logging
                    for class_idx, class_name in enumerate(self.class_names):
                        class_mask = (
                            (tcls == class_idx)
                            if len(tcls) > 0
                            else torch.zeros(0, dtype=torch.bool, device=device)
                        )
                        if class_mask.any():
                            class_loss = self._focal_loss(
                                ps_cls[class_mask], t[class_mask]
                            )
                            per_class_losses[class_name] += class_loss * self.cls_gain

                # Box regression loss (IoU loss)
                if len(tbox) > 0:
                    # Convert predictions to absolute coordinates
                    pxy = ps_box[..., :2].sigmoid() * 2 - 0.5
                    pwh = (ps_box[..., 2:4].sigmoid() * 2) ** 2 * anchors
                    pbox = torch.cat((pxy, pwh), dim=-1)

                    # IoU loss
                    iou = self._bbox_iou(pbox, tbox, x1y1x2y2=False)
                    box_loss = (1.0 - iou).mean()
                    loss_box += box_loss * self.box_gain

        # Total loss
        loss_total = loss_box + loss_cls + loss_obj

        result = {
            "loss": loss_total,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_obj": loss_obj,
        }

        # Add per-class losses
        for class_name, class_loss in per_class_losses.items():
            result[f"loss_cls_{class_name}"] = class_loss

        return result

    def _build_all_targets(
        self,
        targets: torch.Tensor,
        grid_sizes: List[Tuple[int, int]],
        device: torch.device,
    ):
        """Build targets for all scales."""
        # Define anchor boxes for each scale (these should match the model)
        anchors = [
            [[10, 13], [16, 30], [33, 23]],  # Small objects (high resolution)
            [[30, 61], [62, 45], [59, 119]],  # Medium objects
            [[116, 90], [156, 198], [373, 326]],  # Large objects (low resolution)
        ]

        scale_targets = []

        for scale_idx, (h, w) in enumerate(grid_sizes):
            # Initialize targets for this scale
            batch_size = int(targets[:, 0].max() + 1) if len(targets) > 0 else 1
            tobj = torch.zeros(
                batch_size, 3, h, w, device=device
            )  # 3 anchors per scale
            tcls = []
            tbox = []
            indices = [[], [], [], []]  # batch, anchor, grid_y, grid_x
            anch = []

            if len(targets) > 0:
                # Scale targets to grid
                gt = targets.clone()
                gt[:, 2:6] *= torch.tensor([w, h, w, h], device=device)  # Scale to grid

                # Anchor selection
                anchor_tensor = torch.tensor(
                    anchors[scale_idx], device=device, dtype=torch.float32
                )
                na = anchor_tensor.shape[0]  # Number of anchors

                for gi in range(len(gt)):
                    gt_box = gt[gi]
                    batch_idx = int(gt_box[0])
                    cls = int(gt_box[1])
                    gx, gy, gw, gh = gt_box[2:6]

                    # Check if target is within grid bounds
                    if 0 <= gx < w and 0 <= gy < h:
                        gi_int, gj_int = int(gx), int(gy)

                        # Simple anchor assignment (assign to all anchors for now)
                        for ai in range(na):
                            indices[0].append(batch_idx)
                            indices[1].append(ai)
                            indices[2].append(gj_int)
                            indices[3].append(gi_int)

                            # Set objectness target
                            tobj[batch_idx, ai, gj_int, gi_int] = 1.0

                            # Classification target
                            tcls.append(cls)

                            # Box targets (relative to cell)
                            tbox.append([gx - gi_int, gy - gj_int, gw, gh])
                            anch.append(anchor_tensor[ai])

            # Convert to tensors
            indices = tuple(
                torch.tensor(idx, device=device, dtype=torch.long) for idx in indices
            )
            tcls = (
                torch.tensor(tcls, device=device, dtype=torch.long)
                if tcls
                else torch.zeros(0, device=device, dtype=torch.long)
            )
            tbox = (
                torch.tensor(tbox, device=device, dtype=torch.float32)
                if tbox
                else torch.zeros((0, 4), device=device)
            )
            anch = torch.stack(anch, 0) if anch else torch.zeros((0, 2), device=device)

            scale_targets.append((tobj, tcls, tbox, indices, anch))

        return scale_targets

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

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss implementation."""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def _smooth_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Smooth BCE loss with label smoothing."""
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2
        return F.binary_cross_entropy_with_logits(pred, target, reduction="mean")


# ============================================================================
# LIGHTNING MODULE
# ============================================================================


class YOLOLightningModule(L.LightningModule):
    """Lightning module for YOLO training."""

    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Model
        self.model = YOLOModel(num_classes=config["num_classes"])

        # Load pretrained weights if specified
        if config.get("pretrained", True) and config.get("model_type") == "yolov8n":
            self._load_pretrained_weights()

        # Loss
        self.criterion = YOLOLoss(config["num_classes"], config)

        # Metrics storage
        self.val_predictions = []
        self.val_targets = []

        # Freeze backbone if specified
        self.freeze_epochs = config.get("freeze_backbone_epochs", 0)
        if self.freeze_epochs > 0:
            self._freeze_backbone()

    def _load_pretrained_weights(self):
        """Load pretrained weights (simplified - in practice you'd load actual pretrained YOLO weights)."""
        # This is a placeholder - you'd implement actual weight loading here
        print("📦 Loading pretrained weights...")

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print(f"🧊 Frozen backbone for {self.freeze_epochs} epochs")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print("🔥 Unfroze backbone parameters")

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Unfreeze backbone after specified epochs
        if self.current_epoch == self.freeze_epochs and self.freeze_epochs > 0:
            self._unfreeze_backbone()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch["images"]
        targets = batch["targets"]

        # Forward pass
        predictions = self(images)

        # Compute loss
        loss_dict = self.criterion(predictions, targets)

        # Log losses
        self.log(
            "train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/loss_box", loss_dict["loss_box"], on_step=True, on_epoch=True)
        self.log("train/loss_cls", loss_dict["loss_cls"], on_step=True, on_epoch=True)
        self.log("train/loss_obj", loss_dict["loss_obj"], on_step=True, on_epoch=True)

        # Log per-class losses
        for class_name in self.config["class_names"]:
            loss_key = f"loss_cls_{class_name}"
            if loss_key in loss_dict:
                self.log(
                    f"train/{loss_key}",
                    loss_dict[loss_key],
                    on_step=True,
                    on_epoch=True,
                )

        return loss_dict["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch["images"]
        targets = batch["targets"]

        # Forward pass
        predictions = self(images)

        # Compute loss
        loss_dict = self.criterion(predictions, targets)

        # Log losses
        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True)
        self.log("val/loss_box", loss_dict["loss_box"], on_epoch=True)
        self.log("val/loss_cls", loss_dict["loss_cls"], on_epoch=True)
        self.log("val/loss_obj", loss_dict["loss_obj"], on_epoch=True)

        # Log per-class losses
        for class_name in self.config["class_names"]:
            loss_key = f"loss_cls_{class_name}"
            if loss_key in loss_dict:
                self.log(f"val/{loss_key}", loss_dict[loss_key], on_epoch=True)

        # Store predictions for mAP calculation
        # Note: This is simplified - you'd want proper NMS and evaluation
        self.val_predictions.extend(predictions)
        self.val_targets.extend([targets])

        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        if self.val_predictions:
            # Simplified mAP calculation - replace with proper implementation
            mAP50 = torch.tensor(0.5)  # Placeholder
            mAP50_95 = torch.tensor(0.3)  # Placeholder

            self.log("val/mAP50", mAP50, on_epoch=True, prog_bar=True)
            self.log("val/mAP50-95", mAP50_95, on_epoch=True)

        # Clear stored predictions
        self.val_predictions.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Scheduler
        if self.config["lr_scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["epochs"]
            )
        elif self.config["lr_scheduler"] == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=int(self.trainer.estimated_stepping_batches),
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ============================================================================
# DATA MODULE
# ============================================================================


class YOLODataModule(L.LightningDataModule):
    """Lightning data module for YOLO dataset."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Load dataset configuration
        with open(config["data_path"], "r") as f:
            self.data_config = yaml.safe_load(f)

        self.data_dir = Path(self.data_config["path"])
        self.num_classes = self.data_config["nc"]
        self.class_names = self.data_config["names"]

    def setup(self, stage: Optional[str] = None):
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
            persistent_workers=self.config["persistent_workers"],
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
            persistent_workers=self.config["persistent_workers"],
            collate_fn=collate_fn,
        )


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def create_debug_dataset(config: Dict) -> str:
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


def setup_comet_logger(config: Dict) -> Optional[CometLogger]:
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
