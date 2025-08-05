#!/usr/bin/env python3
"""
Analyze model outputs for occlusion information and implement better confidence methods.
"""

import numpy as np
import cv2
import onnxruntime as ort
import torch
import torch.nn.functional as F
from typing import Tuple, List


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


def analyze_model_outputs(model_path: str, image_path: str):
    """Analyze what the model actually outputs."""
    print("=== Analyzing Model Outputs ===")

    # Load model and get info
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Run inference
    input_tensor, _, _ = preprocess_image(image_path)
    outputs = session.run(None, {"image": input_tensor})

    print("\nActual output shapes:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: {output.shape}")
        if len(output.shape) == 4:
            B, J, H, W = output.shape
            print(f"    Batch={B}, Joints={J}, Height={H}, Width={W}")
            print(f"    Min={np.min(output):.6f}, Max={np.max(output):.6f}")
            print(f"    Mean={np.mean(output):.6f}, Std={np.std(output):.6f}")

    return outputs


def method1_peak_value(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Method 1: Peak value confidence"""
    if len(heatmaps.shape) == 4:
        B, J, H, W = heatmaps.shape
        # Reshape to (B, J, H*W) and find max
        reshaped = heatmaps.reshape(B, J, -1)
        peak_vals = np.max(reshaped, axis=-1)  # (B, J)
        peak_indices = np.argmax(reshaped, axis=-1)  # (B, J)
        return peak_vals, peak_indices
    else:
        # Already squeezed
        J, H, W = heatmaps.shape
        reshaped = heatmaps.reshape(J, -1)
        peak_vals = np.max(reshaped, axis=-1)  # (J,)
        peak_indices = np.argmax(reshaped, axis=-1)  # (J,)
        return peak_vals, peak_indices


def method2_peak_mass(heatmaps: np.ndarray, T: float = 1.0, k: int = 5) -> np.ndarray:
    """Method 2: Peak mass confidence using average pooling after softmax"""
    # Convert to torch tensor for easier manipulation
    if len(heatmaps.shape) == 4:
        H_torch = torch.from_numpy(heatmaps).float()
        B, J, Hs, Ws = H_torch.shape
    else:
        # Add batch dimension
        H_torch = torch.from_numpy(heatmaps).float().unsqueeze(0)
        B, J, Hs, Ws = H_torch.shape

    # Apply softmax with temperature
    soft = torch.softmax(H_torch.view(B * J, -1) / T, dim=-1).view(B, J, Hs, Ws)

    # Average pooling to get summed probability in k×k patches
    patch = F.avg_pool2d(soft, k, stride=1, padding=k // 2)  # summed prob in k×k
    conf_mass = patch.view(B, J, -1).max(-1).values

    return conf_mass.numpy()


def method3_entropy(heatmaps: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Method 3: Entropy-based confidence"""
    # Convert to torch tensor
    if len(heatmaps.shape) == 4:
        H_torch = torch.from_numpy(heatmaps).float()
        B, J, Hs, Ws = H_torch.shape
    else:
        H_torch = torch.from_numpy(heatmaps).float().unsqueeze(0)
        B, J, Hs, Ws = H_torch.shape

    # Apply softmax with temperature
    soft = torch.softmax(H_torch.view(B * J, -1) / T, dim=-1).view(B, J, Hs, Ws)

    # Calculate entropy
    logp = torch.log(soft + 1e-12)
    entropy = -(soft * logp).view(B, J, -1).sum(-1)

    # Normalize entropy (lower entropy = higher confidence)
    max_entropy = np.log(Hs * Ws)
    conf_entropy = 1 - entropy / max_entropy

    return conf_entropy.numpy()


def detect_occlusion(
    heatmaps: np.ndarray,
    peak_indices: np.ndarray,
    confidence: np.ndarray,
    image_shape: Tuple[int, int],
    scale_factor: float,
) -> List[bool]:
    """Detect potentially occluded keypoints based on various heuristics."""
    if len(heatmaps.shape) == 4:
        _, J, H, W = heatmaps.shape
        heatmaps = heatmaps[0]  # Remove batch dimension
        confidence = confidence[0] if len(confidence.shape) > 1 else confidence
        peak_indices = peak_indices[0] if len(peak_indices.shape) > 1 else peak_indices
    else:
        J, H, W = heatmaps.shape

    original_h, original_w = image_shape
    occlusion_flags = []

    for j in range(J):
        is_occluded = False

        # Convert peak index to 2D coordinates
        peak_idx = peak_indices[j]
        peak_y = peak_idx // W
        peak_x = peak_idx % W

        # Heuristic 1: Very low confidence
        if confidence[j] < 0.1:  # Adjust threshold as needed
            is_occluded = True

        # Heuristic 2: Peak at image boundaries (in heatmap space)
        if peak_x <= 1 or peak_x >= W - 2 or peak_y <= 1 or peak_y >= H - 2:
            is_occluded = True

        # Heuristic 3: Check if transformed coordinates are at image boundaries
        # Transform to original image coordinates
        target_h, target_w = 256, 256
        scale_heatmap_to_input = target_w / W

        input_x = peak_x * scale_heatmap_to_input
        input_y = peak_y * scale_heatmap_to_input

        new_width = int(original_w * scale_factor)
        new_height = int(original_h * scale_factor)
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2

        input_x -= x_offset
        input_y -= y_offset

        orig_x = input_x / scale_factor
        orig_y = input_y / scale_factor

        # Check if near image boundaries
        margin = 10  # pixels
        if (
            orig_x < margin
            or orig_x > original_w - margin
            or orig_y < margin
            or orig_y > original_h - margin
        ):
            is_occluded = True

        occlusion_flags.append(is_occluded)

    return occlusion_flags


def test_confidence_methods(image_path: str, model_path: str):
    """Test all three confidence methods and occlusion detection."""
    print(f"\n=== Testing Confidence Methods on {image_path} ===")

    # Get model outputs
    outputs = analyze_model_outputs(model_path, image_path)
    heatmaps = outputs[0]  # Assuming first output is heatmaps

    input_tensor, original_image, scale_factor = preprocess_image(image_path)

    print("\n--- Method Comparison ---")

    # Method 1: Peak value
    peak_vals, peak_indices = method1_peak_value(heatmaps)
    print("Method 1 (Peak Value):")
    print(f"  Range: {np.min(peak_vals):.6f} to {np.max(peak_vals):.6f}")
    print(f"  Mean: {np.mean(peak_vals):.6f}")

    # Method 2: Peak mass
    conf_mass = method2_peak_mass(heatmaps, T=1.0, k=5)
    print("Method 2 (Peak Mass, k=5):")
    print(f"  Range: {np.min(conf_mass):.6f} to {np.max(conf_mass):.6f}")
    print(f"  Mean: {np.mean(conf_mass):.6f}")

    # Method 3: Entropy
    conf_entropy = method3_entropy(heatmaps, T=1.0)
    print("Method 3 (Entropy):")
    print(f"  Range: {np.min(conf_entropy):.6f} to {np.max(conf_entropy):.6f}")
    print(f"  Mean: {np.mean(conf_entropy):.6f}")

    # Occlusion detection
    occlusion_flags = detect_occlusion(
        heatmaps, peak_indices, peak_vals, original_image.shape[:2], scale_factor
    )

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

    print("\n--- Detailed Results ---")
    print(
        f"{'Joint':<3} {'Name':<20} {'Peak':<8} {'Mass':<8} {'Entropy':<8} {'Occluded'}"
    )
    print("-" * 70)

    # Handle batch dimension
    if len(peak_vals.shape) > 1:
        peak_vals = peak_vals[0]
        conf_mass = conf_mass[0]
        conf_entropy = conf_entropy[0]

    for i in range(len(keypoint_names)):
        name = keypoint_names[i]
        occluded_str = "YES" if occlusion_flags[i] else "NO"
        print(
            f"{i:3d} {name:<20} {peak_vals[i]:8.4f} {conf_mass[i]:8.4f} {conf_entropy[i]:8.4f} {occluded_str}"
        )

    return {
        "peak_vals": peak_vals,
        "conf_mass": conf_mass,
        "conf_entropy": conf_entropy,
        "occlusion_flags": occlusion_flags,
        "heatmaps": heatmaps,
        "original_image": original_image,
        "scale_factor": scale_factor,
    }


def main():
    model_path = "checkpoints/vhr_birdpose_s_add_int8.onnx"

    for image_name in ["example.jpeg", "example2.jpeg"]:
        if os.path.exists(image_name):
            results = test_confidence_methods(image_name, model_path)
        else:
            print(f"Skipping {image_name} - file not found")


if __name__ == "__main__":
    import os

    main()
