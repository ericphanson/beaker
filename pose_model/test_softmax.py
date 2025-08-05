#!/usr/bin/env python3
"""
Quick test to check softmax confidence values
"""

import numpy as np
import cv2
import onnxruntime as ort


def apply_softmax_to_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply softmax to normalize a single heatmap to probability distribution."""
    flat = heatmap.flatten()
    exp_vals = np.exp(flat - np.max(flat))
    softmax_vals = exp_vals / np.sum(exp_vals)
    return softmax_vals.reshape(heatmap.shape)


def test_softmax_confidence():
    # Load image and preprocess (simplified)
    image = cv2.imread("example2.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Quick preprocessing
    resized = cv2.resize(image_rgb, (256, 192))
    processed = np.ones((256, 256, 3), dtype=np.uint8) * 114
    processed[32:224, 0:256] = resized
    processed = processed.astype(np.float32) / 255.0
    processed = np.transpose(processed, (2, 0, 1))
    processed = np.expand_dims(processed, axis=0)

    # Run inference
    session = ort.InferenceSession(
        "checkpoints/vhr_birdpose_s_add_int8.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"image": processed})
    heatmaps = outputs[0][0]  # Remove batch dimension

    print("Raw heatmap statistics:")
    print(f"  Shape: {heatmaps.shape}")
    print(f"  Min: {np.min(heatmaps):.6f}")
    print(f"  Max: {np.max(heatmaps):.6f}")
    print(f"  Mean: {np.mean(heatmaps):.6f}")

    # Check highest confidence keypoints before softmax
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

    print("\nTop raw confidence values:")
    max_vals = [np.max(heatmaps[i]) for i in range(heatmaps.shape[0])]
    sorted_indices = np.argsort(max_vals)[::-1][:5]

    for idx in sorted_indices:
        name = keypoint_names[idx]
        val = max_vals[idx]
        print(f"  {name}: {val:.6f}")

    # Apply softmax and check confidence
    print("\nAfter softmax normalization:")
    softmax_max_vals = []
    for i in range(heatmaps.shape[0]):
        softmax_heatmap = apply_softmax_to_heatmap(heatmaps[i])
        max_val = np.max(softmax_heatmap)
        softmax_max_vals.append(max_val)

    print(f"  Max softmax confidence: {np.max(softmax_max_vals):.6f}")
    print(f"  Min softmax confidence: {np.min(softmax_max_vals):.6f}")
    print(f"  Average softmax confidence: {np.mean(softmax_max_vals):.6f}")

    # Find reasonable threshold
    thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    for thresh in thresholds:
        count = sum(1 for val in softmax_max_vals if val > thresh)
        print(f"  Keypoints above {thresh}: {count}/23")


if __name__ == "__main__":
    test_softmax_confidence()
