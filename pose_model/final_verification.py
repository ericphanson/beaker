#!/usr/bin/env python3
"""
Quick verification of keypoint detection results with proper softmax normalization.
"""

import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
from typing import Tuple


def apply_softmax_to_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply softmax to normalize a single heatmap to probability distribution."""
    flat = heatmap.flatten()
    exp_vals = np.exp(flat - np.max(flat))
    softmax_vals = exp_vals / np.sum(exp_vals)
    return softmax_vals.reshape(heatmap.shape)


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)):
    """Preprocess image for VHR-BirdPose model."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()

    original_height, original_width = image_rgb.shape[:2]
    target_height, target_width = target_size

    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale_factor = min(scale_x, scale_y)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    processed = np.ones((target_height, target_width, 3), dtype=np.uint8) * 114
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    processed[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized
    )

    processed = processed.astype(np.float32) / 255.0
    processed = np.transpose(processed, (2, 0, 1))
    processed = np.expand_dims(processed, axis=0)

    return processed, original_image, scale_factor


def postprocess_heatmaps(
    heatmaps: np.ndarray, original_size: Tuple[int, int], scale_factor: float
):
    """Postprocess heatmaps with softmax normalization."""
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]

    num_joints, heatmap_h, heatmap_w = heatmaps.shape
    original_h, original_w = original_size

    keypoints = []
    for joint_idx in range(num_joints):
        raw_heatmap = heatmaps[joint_idx]

        # Apply softmax normalization
        heatmap = apply_softmax_to_heatmap(raw_heatmap)
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_val = heatmap[max_idx]

        heatmap_y = max_idx[0]
        heatmap_x = max_idx[1]

        # Coordinate transformation
        target_h, target_w = 256, 256
        scale_heatmap_to_input = target_w / heatmap_w

        input_x = heatmap_x * scale_heatmap_to_input
        input_y = heatmap_y * scale_heatmap_to_input

        new_width = int(original_w * scale_factor)
        new_height = int(original_h * scale_factor)
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2

        input_x -= x_offset
        input_y -= y_offset

        orig_x = int(input_x / scale_factor)
        orig_y = int(input_y / scale_factor)

        orig_x = max(0, min(orig_x, original_w - 1))
        orig_y = max(0, min(orig_y, original_h - 1))

        keypoints.append((orig_x, orig_y, float(max_val)))

    return keypoints


def test_image(image_path: str, model_path: str):
    """Test keypoint detection on a single image."""
    print(f"\n=== Testing {image_path} ===")

    # Preprocess
    input_tensor, original_image, scale_factor = preprocess_image(image_path)

    # Run inference
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"image": input_tensor})
    heatmaps = outputs[0]

    # Postprocess
    keypoints = postprocess_heatmaps(heatmaps, original_image.shape[:2], scale_factor)

    # Animal Kingdom keypoint names
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

    # Show confident keypoints
    print(f"Image size: {original_image.shape[:2]}")
    print(f"Scale factor: {scale_factor}")
    print("\nConfident keypoints (>0.001):")
    confident_keypoints = []
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.001:
            name = keypoint_names[i] if i < len(keypoint_names) else f"Joint_{i}"
            print(f"  {i:2d}: {name:20s} at ({x:4d}, {y:4d}) conf={conf:.6f}")
            confident_keypoints.append((i, x, y, conf, name))

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(original_image)
    ax.set_title(
        f"VHR-BirdPose: {image_path} ({len(confident_keypoints)} confident keypoints)"
    )

    colors = plt.cm.tab20(np.linspace(0, 1, len(keypoint_names)))

    for i, x, y, conf, name in confident_keypoints:
        color = colors[i]
        ax.plot(
            x,
            y,
            "o",
            color=color,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
        )
        ax.annotate(
            f"{name}\n({conf:.4f})",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

    ax.axis("off")
    plt.tight_layout()

    # Save with image name
    output_file = f"verified_{image_path.replace('.jpeg', '.png')}"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as '{output_file}'")
    plt.show()

    return confident_keypoints


def main():
    model_path = "checkpoints/vhr_birdpose_s_add_int8.onnx"

    for image_name in ["example.jpeg", "example2.jpeg"]:
        if os.path.exists(image_name):
            test_image(image_name, model_path)
        else:
            print(f"Skipping {image_name} - file not found")


if __name__ == "__main__":
    import os

    main()
