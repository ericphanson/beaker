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


class ONNXInferenceModel:
    """ONNX model wrapper for inference"""

    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize ONNX model

        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence score for predictions
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

        # Get expected input shape from the model
        input_shape = self.session.get_inputs()[0].shape
        self.expected_height = input_shape[2]
        self.expected_width = input_shape[3]

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

    def postprocess_predictions(self, outputs, original_image_size):
        """
        Post-process ONNX model outputs

        Args:
            outputs: Raw ONNX model outputs
            original_image_size: (height, width) of the original input image

        Returns:
            Dictionary with processed predictions
        """
        # This will need to be adapted based on your ONNX model's output format
        # Based on the export script, outputs should be [dets, labels, orients]

        if len(outputs) >= 3:
            dets = outputs[0]  # Bounding boxes
            labels = outputs[1]  # Class logits
            orients = outputs[2]  # Orientations
        elif len(outputs) >= 2:
            dets = outputs[0]  # Bounding boxes
            labels = outputs[1]  # Class logits
        else:
            # Single output case - need to split based on your model
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")

        # Convert to torch tensors for easier processing
        dets = torch.from_numpy(dets).squeeze(0)  # Remove batch dimension
        labels = torch.from_numpy(labels).squeeze(0)  # Remove batch dimension

        # Apply softmax to get probabilities
        probs = torch.softmax(labels, dim=-1)

        # Get top 1 prediction per class (excluding background class if it's the last)
        # Assuming background is the last class
        num_classes = probs.shape[-1] - 1  # Exclude background

        selected_indices = []
        selected_probs = []
        selected_classes = []

        for class_idx in range(num_classes):
            # Get probabilities for this class across all queries
            class_probs = probs[:, class_idx]

            # Find the query with highest probability for this class
            best_query_idx = torch.argmax(class_probs)
            best_prob = class_probs[best_query_idx]

            # Only include if probability is reasonable (avoid very low confidence)
            if best_prob > 0.1:  # Much lower threshold than before
                selected_indices.append(best_query_idx)
                selected_probs.append(best_prob)
                selected_classes.append(class_idx)

        if len(selected_indices) > 0:
            # Convert to tensors
            selected_indices = torch.tensor(selected_indices)
            selected_probs = torch.tensor(selected_probs)
            selected_classes = torch.tensor(selected_classes)

            # Extract selected predictions
            confident_boxes = dets[selected_indices]
            confident_probs = selected_probs
            confident_classes = selected_classes

            # Handle orientation predictions if available
            confident_orients = None
            if len(outputs) >= 3 and orients is not None:
                orients_tensor = torch.from_numpy(orients).squeeze(
                    0
                )  # Remove batch dimension
                confident_orients = orients_tensor[selected_indices]

            return {
                "boxes": confident_boxes,
                "scores": confident_probs,
                "labels": confident_classes + 1,  # Add 1 if your classes are 1-indexed
                "num_predictions": len(confident_boxes),
                "orients": confident_orients,
            }
        else:
            return {
                "boxes": torch.empty(0, 4),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
                "orients": None,
                "num_predictions": 0,
            }

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

    # Create the plot
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

    for i, (box, label) in enumerate(zip(boxes, labels)):
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
        if has_orient is not None and has_orient[i] and orient is not None:
            angle = orient[i]
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
        if class_names and label - 1 < len(class_names):
            label_text = f"{class_names[label-1]} ({label})"

        # Add orientation info to label if present
        if has_orient is not None and has_orient[i] and orient is not None:
            angle_deg = np.degrees(orient[i])
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

        for i, (pred_box, pred_score, pred_label) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            cx, cy, bw, bh = pred_box

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
            if pred_orients is not None and i < len(pred_orients):
                # pred_orients contains [cos(θ), sin(θ)] for each prediction
                cos_theta, sin_theta = pred_orients[i]

                # Calculate the angle from cos and sin
                pred_angle = np.arctan2(sin_theta, cos_theta)

                # Draw a line through the center of the bbox at the predicted angle
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
            if class_names and pred_label - 1 < len(class_names):
                pred_label_text = (
                    f"Pred: {class_names[pred_label-1]} ({pred_score:.2f})"
                )

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
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Visualization saved to: {save_path}")

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
    class_names = ["background", "bird", "head", "beak", "eye"]

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize samples from the data loader"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "random", "oriented"],
        default="single",
        help="Visualization mode: single batch, random samples, or find oriented samples",
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
