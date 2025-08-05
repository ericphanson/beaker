#!/usr/bin/env python3
"""
Debug the test_model function to see actual confidence values
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple


def apply_softmax_to_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply softmax to normalize a single heatmap to probability distribution."""
    flat = heatmap.flatten()
    exp_vals = np.exp(flat - np.max(flat))
    softmax_vals = exp_vals / np.sum(exp_vals)
    return softmax_vals.reshape(heatmap.shape)


def postprocess_heatmaps_debug(
    heatmaps: np.ndarray, original_size: Tuple[int, int], scale_factor: float
):
    """Debug version of postprocess_heatmaps"""
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

        # Debug output for first few joints
        if joint_idx < 5:
            print(f"Joint {joint_idx}: confidence {max_val:.6f}")

    return keypoints


def main():
    # Load and preprocess image
    image = cv2.imread("example2.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()

    # Simplified preprocessing
    original_height, original_width = image_rgb.shape[:2]
    scale_factor = 0.2
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized = cv2.resize(image_rgb, (new_width, new_height))

    processed = np.ones((256, 256, 3), dtype=np.uint8) * 114
    y_offset = (256 - new_height) // 2
    x_offset = (256 - new_width) // 2
    processed[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized
    )

    processed = processed.astype(np.float32) / 255.0
    processed = np.transpose(processed, (2, 0, 1))
    input_tensor = np.expand_dims(processed, axis=0)

    # Run inference
    session = ort.InferenceSession(
        "checkpoints/vhr_birdpose_s_add_int8.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"image": input_tensor})
    heatmaps = outputs[0]

    # Postprocess with debug
    print("Processing keypoints with debug output:")
    keypoints = postprocess_heatmaps_debug(
        heatmaps, original_image.shape[:2], scale_factor
    )

    # Calculate statistics like the original script
    confidences = [conf for _, _, conf in keypoints]
    avg_confidence = np.mean(confidences)
    valid_keypoints = sum(1 for conf in confidences if conf > 0.001)

    print("\nStatistics:")
    print(f"Average confidence: {avg_confidence:.6f}")
    print(f"Valid keypoints (>0.001): {valid_keypoints}/23")
    print(f"Max confidence: {max(confidences):.6f}")
    print(f"Min confidence: {min(confidences):.6f}")

    # Show all confidence values
    print("\nAll confidence values:")
    for i, conf in enumerate(confidences):
        print(f"  Joint {i:2d}: {conf:.6f}")


if __name__ == "__main__":
    main()
