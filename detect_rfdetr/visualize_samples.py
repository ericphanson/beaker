#!/usr/bin/env python3
"""
Visualize samples directly from the data loader to see what goes into training
Supports both train/test datasets and ONNX model inference for predictions
"""

import torch
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import onnxruntime as ort

from rfdetr.models.lwdetr import PostProcess
from rfdetr.util import box_ops


sys.path.append("rfdetr")

from rfdetr.main import populate_args
from rfdetr.datasets import build_dataset
import rfdetr.util.misc as utils
from torch.utils.data import DataLoader


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image with mean and std.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    Returns:
        Tensor: Denormalized image.
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in (cx, cy, w, h) format
    """

    # Convert to (x1, y1, x2, y2) format
    def cxcywh_to_xyxy(box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = cxcywh_to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = cxcywh_to_xyxy(box2)

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def find_class_mapping(gt_boxes, gt_labels, pred_boxes, pred_classes):
    """
    Find the mapping between predicted classes and ground truth classes
    by matching bounding boxes with highest IoU
    """
    mapping = {}
    unmatched_predictions = []

    print("\nAnalyzing class mapping:")
    print(f"GT boxes: {len(gt_boxes)}, GT labels: {gt_labels}")
    print(f"Pred boxes: {len(pred_boxes)}, Pred classes: {pred_classes}")

    for i, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        pred_class_name = CLASS_NAMES[pred_class]

        if best_iou > 0.1:  # Minimum IoU threshold
            gt_label = gt_labels[best_gt_idx]
            gt_class_name = CLASS_NAMES[gt_label]

            print(
                f"Pred class {pred_class} ({pred_class_name}) -> GT class {gt_label} ({gt_class_name}), IoU: {best_iou:.3f}"
            )

            if pred_class not in mapping:
                mapping[pred_class] = []
            mapping[pred_class].append((gt_label, best_iou))
        else:
            print(
                f"Pred class {pred_class} ({pred_class_name}) -> NO MATCH (best IoU: {best_iou:.3f})"
            )
            unmatched_predictions.append((pred_class, pred_class_name, best_iou))

    # Determine the most likely mapping
    final_mapping = {}
    for pred_class, matches in mapping.items():
        # Sort by IoU and take the most common GT class
        matches.sort(key=lambda x: x[1], reverse=True)
        most_common_gt = matches[0][0]
        final_mapping[pred_class] = most_common_gt

    print(f"Inferred mapping: {final_mapping}")
    if unmatched_predictions:
        first_10 = unmatched_predictions[:10]
        print(
            f"Unmatched predictions (possibly background): {first_10} and {len(unmatched_predictions) - 10} more"
        )

    return final_mapping


class ONNXInferenceModel:
    """ONNX model wrapper for inference"""

    def __init__(self, model_path, confidence_threshold=0.5, remap=True):
        """
        Initialize ONNX model

        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence score for predictions
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        self.remap = remap
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

        # Get expected input shape from the model
        input_shape = self.session.get_inputs()[0].shape
        self.expected_height = input_shape[2]
        self.expected_width = input_shape[3]
        self.post_process = PostProcess(num_select=20)

        print(f"Loaded ONNX model from {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Expected input shape: {input_shape}")
        print(f"Expected resolution: {self.expected_height}x{self.expected_width}")

    def preprocess_image(self, image_tensor):
        """
        Preprocess image tensor for ONNX inference

        Args:
            image_tensor: Image tensor of shape (C, H, W)

        Returns:
            Preprocessed tensor ready for ONNX inference
        """
        # Add batch dimension if not present
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Resize to expected dimensions if needed
        current_height, current_width = image_tensor.shape[2], image_tensor.shape[3]
        if (
            current_height != self.expected_height
            or current_width != self.expected_width
        ):
            import torch.nn.functional as F

            print(
                f"Resizing from {current_height}x{current_width} to {self.expected_height}x{self.expected_width}"
            )
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.expected_height, self.expected_width),
                mode="bilinear",
                align_corners=False,
            )

        return image_tensor.cpu().numpy()

    def postprocess_predictions(self, onnx_outputs, original_image_size):
        """
        Post-process ONNX model outputs

        Args:
            onnx_outputs: Raw ONNX model outputs
            original_image_size: (height, width) of the original input image

        Returns:
            Dictionary with processed predictions
        """

        outputs = {
            "pred_boxes": torch.tensor(onnx_outputs[0]),
            "pred_logits": torch.tensor(onnx_outputs[1]),
            "pred_orient": torch.tensor(onnx_outputs[2]),
        }
        # Repeated batch-size many times
        target_sizes = torch.tensor(
            [original_image_size for _ in range(onnx_outputs[0].shape[0])]
        )
        results = self.post_process(outputs, target_sizes)
        # print(results)
        # we expect batch size to be 1:
        assert len(results) == 1
        r = results[0]

        # Here we undo some of the postprocessing to match the visualization outputs
        # Rescale outputs to be in 0-1 range with cxcywh format
        boxes = box_ops.box_xyxy_to_cxcywh(r["boxes"])
        r["boxes"] = [
            [
                box[0] / original_image_size[0],
                box[1] / original_image_size[1],
                box[2] / original_image_size[0],
                box[3] / original_image_size[1],
            ]
            for box in boxes
        ]

        # Empirical -1.. not sure why this is necessary!
        # This makes the classes -1, 0, 1, 2, 3...
        # background is supposed to be the final class, but maybe somehow it is the
        # first class, so by doing this we filter it out?
        # r["labels"] = torch.tensor([r - 1 for r in r["labels"]])

        # Now we keep the top-k predictions per class
        selected_boxes = []
        selected_scores = []
        selected_labels = []
        top_k = 1

        for c in r["labels"].unique():
            idx = (r["labels"] == c).nonzero(as_tuple=True)[0]
            if len(idx) > top_k:
                idx = idx[torch.topk(r["scores"][idx], top_k).indices]

            # Convert tensor indices to list for proper indexing
            idx_list = idx.tolist()

            # Extract the corresponding boxes, scores, and labels
            class_boxes = [r["boxes"][i] for i in idx_list]
            class_scores = r["scores"][idx]
            class_labels = r["labels"][idx]

            selected_boxes.extend(class_boxes)
            selected_scores.append(class_scores)
            selected_labels.append(class_labels)

        # Concatenate all tensors
        if selected_scores:
            final_scores = torch.cat(selected_scores)
            final_labels = torch.cat(selected_labels)
        else:
            final_scores = torch.tensor([])
            final_labels = torch.tensor([])

        out = {
            "boxes": selected_boxes,
            "scores": final_scores,
            "labels": final_labels,
            "num_predictions": len(selected_boxes),
            "orients": r.get("orients", None),
        }

        return out

    def predict(self, image_tensor):
        """
        Run inference on an image tensor

        Args:
            image_tensor: Image tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Dictionary with predictions
        """
        # Store original image size for coordinate scaling
        original_size = (image_tensor.shape[-2], image_tensor.shape[-1])

        # Preprocess image
        input_tensor = self.preprocess_image(image_tensor)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Post-process predictions
        predictions = self.postprocess_predictions(outputs, original_size)

        return predictions


def visualize_sample(
    images,
    targets,
    sample_idx=0,
    class_names=None,
    save_path=None,
    show=True,
    predictions=None,
    onnx_model=None,
    show_eye=False,
    show_beak=False,
):
    """
    Visualize a single sample from the data loader

    Args:
        images: NestedTensor containing the batch of images
        targets: List of target dictionaries
        sample_idx: Index of the sample to visualize (default: 0)
        class_names: List of class names for labeling (optional)
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot
        predictions: Dictionary with prediction results (optional)
        onnx_model: ONNXInferenceModel instance for inference (optional)
        show_eye: Whether to show eye boxes (default: False)
        show_beak: Whether to show beak boxes (default: False)
    """
    # Extract the image and target for the specified sample
    if sample_idx >= images.tensors.shape[0]:
        print(
            f"Sample index {sample_idx} is out of range. Batch size: {images.tensors.shape[0]}"
        )
        return

    # Get the image tensor and denormalize it
    image_tensor = images.tensors[sample_idx]  # Shape: (C, H, W)
    image_np = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()

    # Get the target information
    target = targets[sample_idx]
    boxes = target["boxes"].cpu().numpy()  # Normalized coordinates (cx, cy, w, h)
    labels = target["labels"].cpu().numpy()
    image_id = target["image_id"].item()
    orig_size = target["orig_size"].cpu().numpy()  # (height, width)
    size = target["size"].cpu().numpy()  # (height, width) of resized image

    # Get orientation information if present
    has_orient = target.get("has_orient", None)
    orient = target.get("orient", None)
    if has_orient is not None:
        has_orient = has_orient.cpu().numpy()
    if orient is not None:
        orient = orient.cpu().numpy()

    print(f"Sample {sample_idx}:")
    print(f"  Image ID: {image_id}")
    print(f"  Original size: {orig_size}")
    print(f"  Resized size: {size}")
    print(f"  Image tensor shape: {image_tensor.shape}")
    print(f"  Number of objects: {len(boxes)}")
    print(f"  Labels: {labels}")
    if has_orient is not None:
        print(f"  Has orientation: {has_orient}")
        print(f"  Orientations (rad): {orient}")
        print(f"  Orientations (deg): {np.degrees(orient)}")

    # Run inference if ONNX model is provided
    if onnx_model is not None:
        print("Running ONNX inference...")
        predictions = onnx_model.predict(image_tensor)
        print(f"  Predictions: {predictions['num_predictions']} objects detected")
        if predictions["num_predictions"] > 0:
            print(f"  Predicted scores: {predictions['scores']}")
            print(f"  Predicted labels: {predictions['labels']}")
            print(f"  Predicted orientations: {predictions['orients']}")

            # Analyze class mapping by comparing bounding boxes
            if predictions["num_predictions"] > 0:
                # Handle boxes - convert list to numpy array
                if isinstance(predictions["boxes"], list):
                    pred_boxes_np = np.array(predictions["boxes"])
                else:
                    pred_boxes_np = (
                        predictions["boxes"].cpu().numpy()
                        if hasattr(predictions["boxes"], "cpu")
                        else predictions["boxes"]
                    )
                pred_labels_np = (
                    predictions["labels"].cpu().numpy()
                    if hasattr(predictions["labels"], "cpu")
                    else predictions["labels"]
                )

                # Find class mapping based on IoU for debugging/analysis
                # find_class_mapping(
                # boxes, labels, pred_boxes_np, pred_labels_np
                # )  # Both GT and pred labels are now 0-indexed

    # Create the plot
    print(f"Creating plot with {len(boxes)} GT boxes...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    title = f"Sample {sample_idx} - Image ID: {image_id}"
    if predictions and predictions["num_predictions"] > 0:
        title += f" (GT: {len(boxes)}, Pred: {predictions['num_predictions']})"
    ax.set_title(title)

    # Convert normalized boxes to pixel coordinates
    # Boxes are in format (center_x, center_y, width, height) normalized to [0, 1]
    h, w = image_tensor.shape[1], image_tensor.shape[2]

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Filter boxes and labels based on show_eye and show_beak flags
    # Keep bird (0) and head (1) always, filter eye (2) and beak (3) based on flags
    filtered_indices = []
    for i, label in enumerate(labels):
        if label == 0 or label == 1:  # bird and head - always show
            filtered_indices.append(i)
        elif label == 2 and show_eye:  # eye - show only if flag set
            filtered_indices.append(i)
        elif label == 3 and show_beak:  # beak - show only if flag set
            filtered_indices.append(i)

    filtered_boxes = boxes[filtered_indices]
    filtered_labels = labels[filtered_indices]
    if has_orient is not None:
        filtered_has_orient = has_orient[filtered_indices]
    else:
        filtered_has_orient = None
    if orient is not None:
        filtered_orient = orient[filtered_indices]
    else:
        filtered_orient = None

    for i, (box, label) in enumerate(zip(filtered_boxes, filtered_labels)):
        print(f"Processing box {box}")
        cx, cy, bw, bh = box

        # Convert to pixel coordinates
        cx_px = cx * w
        cy_px = cy * h
        bw_px = bw * w
        bh_px = bh * h

        # Convert to corner format (x, y, width, height) for matplotlib
        x = cx_px - bw_px / 2
        y = cy_px - bh_px / 2

        # Create rectangle patch
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x, y), bw_px, bh_px, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Draw orientation line if orientation data is present
        if (
            filtered_has_orient is not None
            and filtered_has_orient[i]
            and filtered_orient is not None
        ):
            angle = filtered_orient[i]
            # Draw a line through the center of the bbox at the given angle
            # Line length is proportional to the bbox size
            line_length = min(bw_px, bh_px) * 0.4

            # Calculate line endpoints
            x1 = cx_px - line_length * np.cos(angle)
            y1 = cy_px - line_length * np.sin(angle)
            x2 = cx_px + line_length * np.cos(angle)
            y2 = cy_px + line_length * np.sin(angle)

            # Draw the orientation line
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)

            # Add a small circle at the center
            ax.plot(cx_px, cy_px, "o", color=color, markersize=4)

        # Add label
        label_text = f"Class {label}"
        if class_names and label < len(class_names):
            label_text = f"{class_names[label]} ({label})"

        # Add orientation info to label if present
        if (
            filtered_has_orient is not None
            and filtered_has_orient[i]
            and filtered_orient is not None
        ):
            angle_deg = np.degrees(filtered_orient[i])
            label_text += f" ({angle_deg:.1f}°)"

        ax.text(
            x,
            y - 5,
            label_text,
            color=color,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
        )

    # Visualize predictions if available
    if predictions and predictions["num_predictions"] > 0:
        print(f"Visualizing {predictions['num_predictions']} predictions...")
        # Handle boxes - convert list to numpy array
        if isinstance(predictions["boxes"], list):
            pred_boxes = np.array(predictions["boxes"])
        else:
            pred_boxes = (
                predictions["boxes"].cpu().numpy()
                if hasattr(predictions["boxes"], "cpu")
                else predictions["boxes"]
            )
        pred_scores = (
            predictions["scores"].cpu().numpy()
            if hasattr(predictions["scores"], "cpu")
            else predictions["scores"]
        )
        pred_labels = (
            predictions["labels"].cpu().numpy()
            if hasattr(predictions["labels"], "cpu")
            else predictions["labels"]
        )
        pred_orients = None
        if predictions["orients"] is not None:
            pred_orients = (
                predictions["orients"].cpu().numpy()
                if hasattr(predictions["orients"], "cpu")
                else predictions["orients"]
            )

        pred_colors = [
            "yellow",
            "magenta",
            "lime",
            "turquoise",
            "gold",
            "lightcoral",
            "lightblue",
            "lightgreen",
        ]

        # Filter predictions based on show_eye and show_beak flags
        # Keep bird (0) and head (1) always, filter eye (2) and beak (3) based on flags
        filtered_pred_indices = []
        for i, pred_label in enumerate(pred_labels):
            if pred_label == 0 or pred_label == 1:  # bird and head - always show
                filtered_pred_indices.append(i)
            elif pred_label == 2 and show_eye:  # eye - show only if flag set
                filtered_pred_indices.append(i)
            elif pred_label == 3 and show_beak:  # beak - show only if flag set
                filtered_pred_indices.append(i)

        filtered_pred_boxes = pred_boxes[filtered_pred_indices]
        filtered_pred_scores = pred_scores[filtered_pred_indices]
        filtered_pred_labels = pred_labels[filtered_pred_indices]
        if pred_orients is not None:
            filtered_pred_orients = pred_orients[filtered_pred_indices]
        else:
            filtered_pred_orients = None

        for i, (pred_box, pred_score, pred_label) in enumerate(
            zip(filtered_pred_boxes, filtered_pred_scores, filtered_pred_labels)
        ):
            cx, cy, bw, bh = pred_box
            print(f"Processing prediction {pred_box}")

            # Convert to pixel coordinates
            cx_px = cx * w
            cy_px = cy * h
            bw_px = bw * w
            bh_px = bh * h

            # Convert to corner format
            x = cx_px - bw_px / 2
            y = cy_px - bh_px / 2

            # Create rectangle patch with dashed line for predictions
            pred_color = pred_colors[i % len(pred_colors)]
            rect = patches.Rectangle(
                (x, y),
                bw_px,
                bh_px,
                linewidth=2,
                edgecolor=pred_color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

            # Draw predicted orientation line if orientation data is available
            if filtered_pred_orients is not None and i < len(filtered_pred_orients):
                # filtered_pred_orients contains [cos(θ), sin(θ)] for each prediction
                cos_theta, sin_theta = filtered_pred_orients[i]

                # Draw a line through the center of the bbox at the predicted angle
                # (angle calculated directly from arctan2(sin_theta, cos_theta))
                # Line length is proportional to the bbox size
                line_length = min(bw_px, bh_px) * 0.4

                # Calculate line endpoints
                x1 = cx_px - line_length * cos_theta
                y1 = cy_px - line_length * sin_theta
                x2 = cx_px + line_length * cos_theta
                y2 = cy_px + line_length * sin_theta

                # Draw the predicted orientation line (dashed to match the bbox)
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=pred_color,
                    linewidth=3,
                    alpha=0.8,
                    linestyle="--",
                )

                # Add a small circle at the center
                ax.plot(cx_px, cy_px, "o", color=pred_color, markersize=4, alpha=0.8)

            # Add prediction label
            pred_label_text = f"Pred: Class {pred_label} ({pred_score:.2f})"
            if class_names and pred_label < len(class_names):
                pred_label_text = f"Pred: {class_names[pred_label]} ({pred_score:.2f})"

            # Add predicted orientation info to label if available
            if pred_orients is not None and i < len(pred_orients):
                cos_theta, sin_theta = pred_orients[i]
                pred_angle_deg = np.degrees(np.arctan2(sin_theta, cos_theta))
                pred_label_text += f" ({pred_angle_deg:.1f}°)"

            ax.text(
                x,
                y + bh_px + 5,
                pred_label_text,
                color=pred_color,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=pred_color, alpha=0.3),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    ax.axis("off")

    if save_path:
        print(f"Working on saving to {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Visualization saved to: {save_path}")
    else:
        print("No save path specified")

    plt.tight_layout()
    if show:
        plt.show()


def visualize_batch(
    images,
    targets,
    num_samples=None,
    class_names=None,
    save_dir=None,
    show=True,
    onnx_model=None,
):
    """
    Visualize multiple samples from a batch

    Args:
        images: NestedTensor containing the batch of images
        targets: List of target dictionaries
        num_samples: Number of samples to visualize (default: all in batch)
        class_names: List of class names for labeling (optional)
        save_dir: Directory to save visualizations (optional)
        onnx_model: ONNXInferenceModel instance for inference (optional)
    """
    batch_size = images.tensors.shape[0]
    if num_samples is None:
        num_samples = batch_size
    else:
        num_samples = min(num_samples, batch_size)

    print(f"Visualizing {num_samples} samples from batch of size {batch_size}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    for i in range(num_samples):
        save_path = None
        if save_dir:
            save_path = save_dir / f"sample_{i}.png"

        visualize_sample(
            images,
            targets,
            sample_idx=i,
            class_names=class_names,
            save_path=save_path,
            show=show,
            onnx_model=onnx_model,
            show_eye=args.eye,
            show_beak=args.beak,
        )


def create_dataloader(
    dataset_dir="../data/cub_coco_parts", batch_size=4, num_workers=0, image_set="train"
):
    """
    Create a data loader for visualization

    Args:
        dataset_dir: Path to the dataset directory
        batch_size: Batch size for the data loader
        num_workers: Number of worker processes
        image_set: Dataset split to use ("train" or "val")

    Returns:
        DataLoader: The created data loader
    """
    # Create minimal args for dataset building using your actual training config
    args = populate_args(
        dataset_file="roboflow",
        coco_path=dataset_dir,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        resolution=640,
        # Your actual training configuration
        square_resize_div_64=True,  # Using square resize transforms
        multi_scale=True,  # Multi-scale training enabled
        expanded_scales=True,  # Expanded scale variations
        do_random_resize_via_padding=False,  # No padding resize
    )

    # Add missing attributes that the dataset building expects
    args.patch_size = 16
    args.num_windows = 4

    print(f"Building {image_set} dataset...")
    print("Using training configuration:")
    print(f"  - square_resize_div_64: {args.square_resize_div_64}")
    print(f"  - multi_scale: {args.multi_scale}")
    print(f"  - expanded_scales: {args.expanded_scales}")
    print(f"  - do_random_resize_via_padding: {args.do_random_resize_via_padding}")

    dataset = build_dataset(image_set=image_set, args=args, resolution=args.resolution)
    print(f"Dataset built successfully. Length: {len(dataset)}")

    # Create data loader
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
    )

    return data_loader


CLASS_NAMES = [
    "bird",
    "head",
    "eye",
    "beak",
    "background",
]  # Adjust based on your dataset


def main(show=False, image_set="train", onnx_model_path=None):
    """
    Main function to demonstrate the visualization

    Args:
        show: Whether to display plots interactively
        image_set: Dataset split to use ("train" or "val")
        onnx_model_path: Path to ONNX model for inference (optional)
    """
    # Define class names for CUB dataset (you may need to adjust these)

    # Initialize ONNX model if provided
    onnx_model = None
    if onnx_model_path:
        try:
            onnx_model = ONNXInferenceModel(onnx_model_path)
            print(f"ONNX model loaded successfully from {onnx_model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            onnx_model = None

    # Create data loader
    data_loader = create_dataloader(batch_size=4, image_set=image_set)

    # Get a batch
    print(f"\nGetting first batch from {image_set} dataset...")
    batch = next(iter(data_loader))
    images, targets = batch

    print(f"Batch contains {len(targets)} samples")

    # Visualize the first few samples
    visualize_batch(
        images,
        targets,
        num_samples=2,
        class_names=CLASS_NAMES,
        save_dir="visualizations",
        show=show,
        onnx_model=onnx_model,
    )


def visualize_random_samples(
    num_batches=3,
    samples_per_batch=2,
    dataset_dir="../data/cub_coco_parts",
    save_dir="visualizations",
    class_names=CLASS_NAMES,
    show=False,
    image_set="train",
    onnx_model_path=None,
):
    """
    Visualize random samples from multiple batches

    Args:
        num_batches: Number of batches to sample from
        samples_per_batch: Number of samples to visualize per batch
        dataset_dir: Path to the dataset directory
        save_dir: Directory to save visualizations
        show: Whether to display plots interactively
        image_set: Dataset split to use ("train" or "val")
        onnx_model_path: Path to ONNX model for inference (optional)
    """

    # Initialize ONNX model if provided
    onnx_model = None
    if onnx_model_path:
        try:
            onnx_model = ONNXInferenceModel(onnx_model_path)
            print(f"ONNX model loaded successfully from {onnx_model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            onnx_model = None

    # Create data loader
    data_loader = create_dataloader(
        dataset_dir=dataset_dir, batch_size=4, image_set=image_set
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    sample_count = 0

    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

        try:
            batch = next(iter(data_loader))
            images, targets = batch

            for sample_idx in range(min(samples_per_batch, len(targets))):
                save_path = save_dir / f"batch_{batch_idx}_sample_{sample_idx}.png"
                visualize_sample(
                    images,
                    targets,
                    sample_idx=sample_idx,
                    class_names=class_names,
                    save_path=save_path,
                    show=show,
                    onnx_model=onnx_model,
                    show_eye=args.eye,
                    show_beak=args.beak,
                )
                sample_count += 1

        except StopIteration:
            print("No more batches available")
            break

    print(f"\nVisualized {sample_count} samples total")


def find_samples_with_orientation(
    dataset_dir="../data/cub_coco_parts", max_samples=10, save_dir="visualizations"
):
    """
    Search through the dataset to find and visualize samples that have orientation data

    Args:
        dataset_dir: Path to the dataset directory
        max_samples: Maximum number of oriented samples to find and visualize
        save_dir: Directory to save visualizations
    """
    class_names = CLASS_NAMES

    # Create data loader
    data_loader = create_dataloader(
        dataset_dir=dataset_dir, batch_size=8
    )  # Larger batch for faster search

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    found_count = 0
    total_checked = 0

    print(f"Searching for samples with orientation data (max {max_samples})...")

    for batch_idx, (images, targets) in enumerate(data_loader):
        for sample_idx in range(len(targets)):
            total_checked += 1
            target = targets[sample_idx]

            # Check if this sample has any orientation data
            if "has_orient" in target:
                has_orient = target["has_orient"].cpu().numpy()
                if any(has_orient):  # If any bbox in this sample has orientation
                    print(f"\nFound oriented sample! (checked {total_checked} samples)")
                    print(f"  Image ID: {target['image_id'].item()}")
                    print(
                        f"  Objects with orientation: {sum(has_orient)}/{len(has_orient)}"
                    )

                    # Visualize this sample
                    save_path = save_dir / f"oriented_sample_{found_count}.png"
                    visualize_sample(
                        images,
                        targets,
                        sample_idx=sample_idx,
                        class_names=class_names,
                        save_path=save_path,
                        show_eye=args.eye,
                        show_beak=args.beak,
                    )

                    found_count += 1
                    if found_count >= max_samples:
                        print(
                            f"\nFound {found_count} samples with orientation data after checking {total_checked} samples"
                        )
                        return

        if batch_idx % 10 == 0:
            print(
                f"Checked {total_checked} samples, found {found_count} with orientation..."
            )

    print(
        f"\nSearch complete: Found {found_count} samples with orientation data after checking {total_checked} samples"
    )
    if found_count == 0:
        print("No samples with orientation data found in the dataset")


def find_oriented_samples(
    max_batches=50,
    dataset_dir="../data/cub_coco_parts",
    save_dir="visualizations",
    show=False,
    image_set="train",
    onnx_model_path=None,
):
    """
    Find and visualize samples that have orientation data (has_orient=True)

    Args:
        max_batches: Maximum number of batches to search through
        dataset_dir: Path to the dataset directory
        save_dir: Directory to save visualizations
        show: Whether to display plots interactively
        image_set: Dataset split to use ("train" or "val")
        onnx_model_path: Path to ONNX model for inference (optional)
    """
    class_names = CLASS_NAMES

    # Initialize ONNX model if provided
    onnx_model = None
    if onnx_model_path:
        try:
            onnx_model = ONNXInferenceModel(onnx_model_path)
            print(f"ONNX model loaded successfully from {onnx_model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            onnx_model = None

    # Create data loader
    data_loader = create_dataloader(
        dataset_dir=dataset_dir, batch_size=4, image_set=image_set
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    sample_count = 0
    oriented_samples_found = 0

    print(f"Searching for samples with orientation data (max {max_batches} batches)...")

    for batch_idx in range(max_batches):
        try:
            batch = next(iter(data_loader))
            images, targets = batch

            # Check each sample in the batch
            for sample_idx, target in enumerate(targets):
                sample_count += 1

                # Check if this sample has any orientation data
                if "has_orient" in target:
                    has_orient = target["has_orient"].cpu().numpy()

                    # Check if any object in this sample has orientation
                    if has_orient.any():
                        print(
                            f"\nFound oriented sample! Batch {batch_idx}, Sample {sample_idx}"
                        )
                        print(
                            f"  Objects with orientation: {has_orient.sum()}/{len(has_orient)}"
                        )

                        if "orient" in target:
                            orient_angles = target["orient"].cpu().numpy()
                            print(f"  Orientation angles (rad): {orient_angles}")
                            print(
                                f"  Orientation angles (deg): {np.degrees(orient_angles)}"
                            )

                        # Visualize this sample
                        save_path = (
                            save_dir
                            / f"oriented_batch_{batch_idx}_sample_{sample_idx}.png"
                        )
                        visualize_sample(
                            images,
                            targets,
                            sample_idx=sample_idx,
                            class_names=class_names,
                            save_path=save_path,
                            show=show,
                            onnx_model=onnx_model,
                            show_eye=args.eye,
                            show_beak=args.beak,
                        )
                        oriented_samples_found += 1

                        # Stop after finding a few oriented samples
                        if oriented_samples_found >= 5:
                            print(
                                f"\nFound {oriented_samples_found} oriented samples, stopping search."
                            )
                            return

                if sample_count % 50 == 0:
                    print(f"Searched {sample_count} samples so far...")

        except StopIteration:
            print("No more batches available")
            break

    if oriented_samples_found == 0:
        print(
            f"\nNo oriented samples found after searching {sample_count} samples across {batch_idx + 1} batches."
        )
        print("This might mean:")
        print("- The dataset doesn't contain orientation annotations")
        print("- Orientation data is stored differently")
        print("- Need to search more samples")
    else:
        print(
            f"\nFound {oriented_samples_found} oriented samples out of {sample_count} total samples."
        )


def analyze_class_mappings_across_samples(
    num_samples=10, image_set="test", onnx_model_path=None
):
    """
    Analyze class mappings across multiple samples to find consistent patterns
    """
    if not onnx_model_path:
        onnx_model_path = "onnx_export/export_output/inference_model.sim.onnx"

    # Initialize ONNX model, no mapping here
    onnx_model = ONNXInferenceModel(onnx_model_path, remap=False)

    # Create data loader
    data_loader = create_dataloader(batch_size=4, image_set=image_set)

    all_mappings = []
    sample_count = 0

    print(f"Analyzing class mappings across {num_samples} samples...")
    print("=" * 60)

    for batch_idx, (images, targets) in enumerate(data_loader):
        for sample_idx in range(len(targets)):
            if sample_count >= num_samples:
                break

            # Get sample data
            image_tensor = images.tensors[sample_idx]
            target = targets[sample_idx]
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            image_id = target["image_id"].item()

            # Run inference
            predictions = onnx_model.predict(image_tensor)

            if predictions["num_predictions"] > 0:
                # Handle boxes - convert list to numpy array
                if isinstance(predictions["boxes"], list):
                    pred_boxes_np = np.array(predictions["boxes"])
                else:
                    pred_boxes_np = (
                        predictions["boxes"].cpu().numpy()
                        if hasattr(predictions["boxes"], "cpu")
                        else predictions["boxes"]
                    )

                # Handle labels - convert tensor to numpy
                pred_labels_np = (
                    predictions["labels"].cpu().numpy()
                    if hasattr(predictions["labels"], "cpu")
                    else predictions["labels"]
                )

                print(f"\nSample {sample_count + 1} (Image ID: {image_id}):")
                print(f"GT: {labels} -> {[CLASS_NAMES[label] for label in labels]}")
                print(
                    f"Pred raw: {pred_labels_np} -> {[CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Unknown({label})' for label in pred_labels_np]}"
                )

                # Find class mapping based on IoU
                class_mapping = find_class_mapping(
                    boxes, labels, pred_boxes_np, pred_labels_np
                )
                all_mappings.append(
                    {
                        "sample_id": sample_count + 1,
                        "image_id": image_id,
                        "mapping": class_mapping,
                        "gt_labels": labels.tolist(),
                        "pred_labels": pred_labels_np.tolist()
                        if hasattr(pred_labels_np, "tolist")
                        else list(pred_labels_np),
                    }
                )

            sample_count += 1
            if sample_count >= num_samples:
                break

        if sample_count >= num_samples:
            break

    # Analyze patterns across all mappings
    print("\n" + "=" * 60)
    print("SUMMARY OF CLASS MAPPINGS ACROSS ALL SAMPLES:")
    print("=" * 60)

    # Collect all mappings for each model class
    class_mapping_summary = {}
    for mapping_data in all_mappings:
        for model_class, gt_class in mapping_data["mapping"].items():
            model_class = int(model_class)
            gt_class = int(gt_class)

            if model_class not in class_mapping_summary:
                class_mapping_summary[model_class] = []
            class_mapping_summary[model_class].append(gt_class)

    # Print summary
    print(f"\nAnalyzed {len(all_mappings)} samples with predictions")
    print("\nMost common mappings for each model class:")

    final_mapping = {}
    for model_class in sorted(class_mapping_summary.keys()):
        gt_classes = class_mapping_summary[model_class]
        from collections import Counter

        counter = Counter(gt_classes)
        most_common = counter.most_common()

        model_class_name = (
            CLASS_NAMES[model_class]
            if model_class < len(CLASS_NAMES)
            else f"Unknown({model_class})"
        )
        print(f"\nModel class {model_class} ({model_class_name}):")
        for gt_class, count in most_common:
            gt_class_name = (
                CLASS_NAMES[gt_class]
                if gt_class < len(CLASS_NAMES)
                else f"Unknown({gt_class})"
            )
            percentage = count / len(gt_classes) * 100
            print(
                f"  -> GT class {gt_class} ({gt_class_name}): {count}/{len(gt_classes)} ({percentage:.1f}%)"
            )

        # Store the most common mapping
        if most_common:
            final_mapping[model_class] = most_common[0][0]

    print("\nFINAL RECOMMENDED MAPPING:")
    for model_class, gt_class in sorted(final_mapping.items()):
        model_class_name = (
            CLASS_NAMES[model_class]
            if model_class < len(CLASS_NAMES)
            else f"Unknown({model_class})"
        )
        gt_class_name = (
            CLASS_NAMES[gt_class]
            if gt_class < len(CLASS_NAMES)
            else f"Unknown({gt_class})"
        )
        print(
            f"Model class {model_class} ({model_class_name}) -> GT class {gt_class} ({gt_class_name})"
        )

    return final_mapping, all_mappings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize samples from the data loader"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "random", "oriented", "mapping"],
        default="single",
        help="Visualization mode: single batch, random samples, find oriented samples, or analyze class mappings",
    )
    parser.add_argument(
        "--num_batches", type=int, default=3, help="Number of batches for random mode"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Whether to show the visualizations interactively",
    )
    parser.add_argument(
        "--samples_per_batch",
        type=int,
        default=2,
        help="Samples per batch for random mode",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=50,
        help="Maximum batches to search for oriented samples",
    )
    parser.add_argument(
        "--dataset_dir", default="../data/cub_coco_parts", help="Dataset directory"
    )
    parser.add_argument(
        "--save_dir", default="visualizations", help="Directory to save visualizations"
    )
    parser.add_argument(
        "--image_set",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to use (train, val, or test)",
    )
    parser.add_argument(
        "--onnx_model", default=None, help="Path to ONNX model for inference (optional)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for ONNX model predictions",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to analyze for mapping mode",
    )
    parser.add_argument(
        "--eye",
        action="store_true",
        help="Show eye predictions and ground truth boxes",
    )
    parser.add_argument(
        "--beak",
        action="store_true",
        help="Show beak predictions and ground truth boxes",
    )

    args = parser.parse_args()

    # Setup ONNX model path if provided
    onnx_model_path = None
    if args.onnx_model:
        onnx_model_path = args.onnx_model
    elif Path("onnx_export/export_output/inference_model.sim.onnx").exists():
        # Use default ONNX model if available
        onnx_model_path = "onnx_export/export_output/inference_model.sim.onnx"
        print(f"Using default ONNX model: {onnx_model_path}")

    if args.mode == "single":
        main(show=args.show, image_set=args.image_set, onnx_model_path=onnx_model_path)
    elif args.mode == "random":
        visualize_random_samples(
            args.num_batches,
            args.samples_per_batch,
            args.dataset_dir,
            args.save_dir,
            show=args.show,
            image_set=args.image_set,
            onnx_model_path=onnx_model_path,
        )
    elif args.mode == "oriented":
        find_oriented_samples(
            args.max_batches,
            args.dataset_dir,
            args.save_dir,
            show=args.show,
            image_set=args.image_set,
            onnx_model_path=onnx_model_path,
        )
    elif args.mode == "mapping":
        if not onnx_model_path:
            print("Error: ONNX model path is required for mapping analysis")
            exit(1)
        final_mapping, all_mappings = analyze_class_mappings_across_samples(
            num_samples=args.num_samples,
            image_set=args.image_set,
            onnx_model_path=onnx_model_path,
        )
