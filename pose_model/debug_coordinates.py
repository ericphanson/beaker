#!/usr/bin/env python3
"""
Debug script to examine confidence values and coordinate transformation issues.
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple


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

    print(f"Original size: {original_width} x {original_height}")
    print(f"Scale factor: {scale_factor}")
    print(f"Resized to: {new_width} x {new_height}")
    print(f"Padding: x_offset={x_offset}, y_offset={y_offset}")

    return processed, original_image, scale_factor, (x_offset, y_offset)


def analyze_heatmaps(heatmaps: np.ndarray, image_name: str):
    """Analyze heatmap values and statistics."""
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]

    print(f"\n=== Heatmap Analysis for {image_name} ===")
    print(f"Heatmap shape: {heatmaps.shape}")

    # Overall statistics
    print(f"Global min: {np.min(heatmaps):.6f}")
    print(f"Global max: {np.max(heatmaps):.6f}")
    print(f"Global mean: {np.mean(heatmaps):.6f}")
    print(f"Global std: {np.std(heatmaps):.6f}")

    # Per-joint analysis
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

    print("\nPer-joint statistics:")
    confident_joints = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        max_val = np.max(heatmap)
        mean_val = np.mean(heatmap)
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        name = keypoint_names[i] if i < len(keypoint_names) else f"Joint_{i}"
        print(
            f"  {i:2d}: {name:20s} max={max_val:.6f} mean={mean_val:.6f} at {max_idx}"
        )

        if max_val > 0.3:
            confident_joints.append((i, name, max_val, max_idx))

    return confident_joints


def postprocess_with_debug(
    heatmaps: np.ndarray,
    original_size: Tuple[int, int],
    scale_factor: float,
    padding_offset: Tuple[int, int],
):
    """Postprocess with detailed debug information."""
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]

    num_joints, heatmap_h, heatmap_w = heatmaps.shape
    original_h, original_w = original_size
    x_offset, y_offset = padding_offset

    print("\n=== Coordinate Transformation Debug ===")
    print(f"Original image size: {original_w} x {original_h}")
    print(f"Heatmap size: {heatmap_w} x {heatmap_h}")
    print(f"Scale factor: {scale_factor}")
    print(f"Padding offsets: x={x_offset}, y={y_offset}")

    keypoints = []
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

    for joint_idx in range(min(5, num_joints)):  # Debug first 5 joints
        heatmap = heatmaps[joint_idx]
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_val = heatmap[max_idx]

        if max_val < 0.3:
            continue

        heatmap_y = max_idx[0]
        heatmap_x = max_idx[1]

        # Step-by-step transformation
        target_h, target_w = 256, 256
        scale_heatmap_to_input = target_w / heatmap_w

        input_x = heatmap_x * scale_heatmap_to_input
        input_y = heatmap_y * scale_heatmap_to_input

        # Remove padding
        unpadded_x = input_x - x_offset
        unpadded_y = input_y - y_offset

        # Scale to original
        orig_x = int(unpadded_x / scale_factor)
        orig_y = int(unpadded_y / scale_factor)

        # Clamp to bounds
        final_x = max(0, min(orig_x, original_w - 1))
        final_y = max(0, min(orig_y, original_h - 1))

        name = keypoint_names[joint_idx]
        print(f"\n{name}:")
        print(f"  Heatmap coords: ({heatmap_x}, {heatmap_y}) confidence: {max_val:.3f}")
        print(f"  Input coords:   ({input_x:.1f}, {input_y:.1f})")
        print(f"  Unpadded:       ({unpadded_x:.1f}, {unpadded_y:.1f})")
        print(f"  Original:       ({orig_x}, {orig_y})")
        print(f"  Final:          ({final_x}, {final_y})")

        keypoints.append((final_x, final_y, float(max_val)))

    return keypoints


def apply_softmax_to_heatmaps(heatmaps: np.ndarray, temperature: float = 1.0):
    """Apply softmax to each heatmap to normalize confidence values."""
    if len(heatmaps.shape) == 4:
        batch_size, num_joints, h, w = heatmaps.shape
        normalized = np.zeros_like(heatmaps)

        for b in range(batch_size):
            for j in range(num_joints):
                heatmap = heatmaps[b, j] / temperature
                # Flatten, apply softmax, reshape
                flat = heatmap.flatten()
                exp_vals = np.exp(
                    flat - np.max(flat)
                )  # Subtract max for numerical stability
                softmax_vals = exp_vals / np.sum(exp_vals)
                normalized[b, j] = softmax_vals.reshape(h, w)

        return normalized
    else:
        # Already squeezed
        num_joints, h, w = heatmaps.shape
        normalized = np.zeros_like(heatmaps)

        for j in range(num_joints):
            heatmap = heatmaps[j] / temperature
            flat = heatmap.flatten()
            exp_vals = np.exp(flat - np.max(flat))
            softmax_vals = exp_vals / np.sum(exp_vals)
            normalized[j] = softmax_vals.reshape(h, w)

        return normalized


def main():
    model_path = "checkpoints/vhr_birdpose_s_add_int8.onnx"

    for image_name in ["example.jpeg", "example2.jpeg"]:
        if not os.path.exists(image_name):
            print(f"Skipping {image_name} - file not found")
            continue

        print(f"\n{'='*80}")
        print(f"ANALYZING {image_name.upper()}")
        print(f"{'='*80}")

        # Preprocess
        input_tensor, original_image, scale_factor, padding_offset = preprocess_image(
            image_name
        )

        # Run inference
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"image": input_tensor})
        raw_heatmaps = outputs[0]

        # Analyze raw heatmaps
        confident_joints = analyze_heatmaps(raw_heatmaps, image_name)

        # Apply softmax normalization
        normalized_heatmaps = apply_softmax_to_heatmaps(raw_heatmaps)
        print("\nAfter softmax normalization:")
        analyze_heatmaps(normalized_heatmaps, f"{image_name} (softmax)")

        # Debug coordinate transformation
        keypoints = postprocess_with_debug(
            raw_heatmaps, original_image.shape[:2], scale_factor, padding_offset
        )


if __name__ == "__main__":
    import os

    main()
