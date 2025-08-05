#!/usr/bin/env python3
"""
Debug script to understand the VHR-BirdPose model output format and coordinate transformation.
"""

import sys
import os
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
from typing import Tuple


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Preprocess image for VHR-BirdPose model."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()

    # Get original dimensions
    original_height, original_width = image_rgb.shape[:2]
    target_height, target_width = target_size

    # Calculate scale factor (preserve aspect ratio)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale_factor = min(scale_x, scale_y)

    # Resize with aspect ratio preservation
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Pad to target size
    processed = (
        np.ones((target_height, target_width, 3), dtype=np.uint8) * 114
    )  # Gray padding
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    processed[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized
    )

    # Convert to tensor format: (1, 3, H, W)
    processed = processed.astype(np.float32) / 255.0  # Normalize to [0, 1]
    processed = np.transpose(processed, (2, 0, 1))  # HWC to CHW
    processed = np.expand_dims(processed, axis=0)  # Add batch dimension

    print(f"Original image size: {original_width} x {original_height}")
    print(f"Scale factor: {scale_factor}")
    print(f"Resized to: {new_width} x {new_height}")
    print(f"Padding offsets: x={x_offset}, y={y_offset}")
    print(f"Final tensor shape: {processed.shape}")

    return processed, original_image, scale_factor


def run_inference_debug(model_path: str, input_tensor: np.ndarray):
    """Run inference and examine outputs."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Input name: {input_info.name}")
    print(f"Input shape: {input_info.shape}")
    print(f"Output name: {output_info.name}")
    print(f"Output shape: {output_info.shape}")

    # Run inference
    outputs = session.run(None, {input_info.name: input_tensor})
    heatmaps = outputs[0]

    print(f"Actual output shape: {heatmaps.shape}")

    # Analyze heatmaps
    if len(heatmaps.shape) == 4:
        batch_size, num_joints, heatmap_h, heatmap_w = heatmaps.shape
        print(f"Batch size: {batch_size}")
        print(f"Number of joints: {num_joints}")
        print(f"Heatmap dimensions: {heatmap_h} x {heatmap_w}")

        # Remove batch dimension for analysis
        heatmaps = heatmaps[0]
    else:
        num_joints, heatmap_h, heatmap_w = heatmaps.shape
        print(f"Number of joints: {num_joints}")
        print(f"Heatmap dimensions: {heatmap_h} x {heatmap_w}")

    # Analyze each joint's heatmap
    print("\nJoint-wise analysis:")
    keypoint_names = [
        "Head_Mid_Top",
        "Eye_Left",
        "Eye_Right",
        "Mouth_Front_Top",
        "Mouth_Back_Left",
        "Mouth_Back_Right",
        "Mouth_Front_Bottom",
        "Shoulder_Left",
        "Shoulder_Right",
        "Elbow_Left",
        "Elbow_Right",
        "Wrist_Left",
        "Wrist_Right",
        "Torso_Mid_Back",
        "Hip_Left",
        "Hip_Right",
        "Knee_Left",
        "Knee_Right",
        "Ankle_Left",
        "Ankle_Right",
        "Tail_Top_Back",
        "Tail_Mid_Back",
        "Tail_End_Back",
    ]

    for joint_idx in range(min(5, num_joints)):  # Analyze first 5 joints
        heatmap = heatmaps[joint_idx]
        max_val = np.max(heatmap)
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        avg_val = np.mean(heatmap)

        joint_name = (
            keypoint_names[joint_idx]
            if joint_idx < len(keypoint_names)
            else f"Joint_{joint_idx}"
        )
        print(f"  {joint_name}:")
        print(f"    Max value: {max_val:.6f} at position {max_idx}")
        print(f"    Average value: {avg_val:.6f}")

    return heatmaps


def visualize_heatmaps(
    heatmaps: np.ndarray, original_image: np.ndarray, num_to_show: int = 6
):
    """Visualize individual heatmaps."""
    keypoint_names = [
        "Head_Mid_Top",
        "Eye_Left",
        "Eye_Right",
        "Mouth_Front_Top",
        "Mouth_Back_Left",
        "Mouth_Back_Right",
        "Mouth_Front_Bottom",
        "Shoulder_Left",
        "Shoulder_Right",
        "Elbow_Left",
        "Elbow_Right",
        "Wrist_Left",
        "Wrist_Right",
        "Torso_Mid_Back",
        "Hip_Left",
        "Hip_Right",
        "Knee_Left",
        "Knee_Right",
        "Ankle_Left",
        "Ankle_Right",
        "Tail_Top_Back",
        "Tail_Mid_Back",
        "Tail_End_Back",
    ]

    cols = 3
    rows = (num_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_to_show):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        if i < len(heatmaps):
            heatmap = heatmaps[i]
            im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear")
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            ax.plot(max_idx[1], max_idx[0], "bo", markersize=8)

            joint_name = keypoint_names[i] if i < len(keypoint_names) else f"Joint_{i}"
            ax.set_title(f"{joint_name}\nMax: {np.max(heatmap):.3f} at {max_idx}")
            plt.colorbar(im, ax=ax)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(num_to_show, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("heatmap_debug.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    model_path = "checkpoints/vhr_birdpose_s_add_int8.onnx"
    image_path = "example.jpeg"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 1

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1

    print("=" * 80)
    print("VHR-BirdPose Debug Analysis")
    print("=" * 80)

    # Preprocess image
    print("\n1. Preprocessing image...")
    input_tensor, original_image, scale_factor = preprocess_image(image_path)

    # Run inference
    print("\n2. Running inference...")
    heatmaps = run_inference_debug(model_path, input_tensor)

    # Visualize heatmaps
    print("\n3. Visualizing heatmaps...")
    visualize_heatmaps(heatmaps, original_image, num_to_show=9)

    print("\nDebug analysis completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
