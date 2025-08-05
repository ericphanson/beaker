#!/usr/bin/env python3
"""
Simple PyTorch vs ONNX comparison script for VHR-BirdPose
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import yaml
import onnxruntime as ort
import torch
from typing import Tuple, List

# Add the upstream directory to the path for importing model components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upstream.pose_vhr import get_pose_net


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Preprocess image for VHR-BirdPose model."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

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


def run_pytorch_inference(
    config_path: str, checkpoint_path: str, input_tensor: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run inference with PyTorch model."""
    cfg = load_config(config_path)
    model = get_pose_net(cfg, is_train=False)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Apply key mapping for fit_out -> att_fit_out
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("fit_out."):
            new_key = key.replace("fit_out.", "att_fit_out.")
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value

    model.load_state_dict(mapped_state_dict, strict=False)
    torch_input = torch.from_numpy(input_tensor).float()

    with torch.no_grad():
        start_time = time.time()
        outputs = model(torch_input)
        inference_time = time.time() - start_time

    if isinstance(outputs, (list, tuple)):
        heatmaps = outputs[0].numpy()
    else:
        heatmaps = outputs.numpy()

    return heatmaps, inference_time


def run_onnx_inference(
    model_path: str, input_tensor: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run inference with ONNX model."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_time

    return outputs[0], inference_time


def postprocess_heatmaps(
    heatmaps: np.ndarray, original_size: Tuple[int, int], scale_factor: float
) -> List[Tuple[int, int, float]]:
    """Postprocess heatmaps to get keypoint coordinates."""
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]

    num_joints, heatmap_h, heatmap_w = heatmaps.shape
    original_h, original_w = original_size

    keypoints = []
    for joint_idx in range(num_joints):
        heatmap = heatmaps[joint_idx]
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_val = heatmap[max_idx]

        heatmap_x = max_idx[1]
        heatmap_y = max_idx[0]

        target_h, target_w = 256, 256
        x_offset = (target_w - int(original_w * scale_factor)) // 2
        y_offset = (target_h - int(original_h * scale_factor)) // 2

        input_x = (heatmap_x / heatmap_w) * target_w
        input_y = (heatmap_y / heatmap_h) * target_h

        input_x -= x_offset
        input_y -= y_offset

        orig_x = int(input_x / scale_factor)
        orig_y = int(input_y / scale_factor)

        orig_x = max(0, min(orig_x, original_w - 1))
        orig_y = max(0, min(orig_y, original_h - 1))

        keypoints.append((orig_x, orig_y, float(max_val)))

    return keypoints


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNX VHR-BirdPose models"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="example.jpeg",
        help="Path to input image (default: example.jpeg)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="w32_256x256_adam_lr1e-3_ak_vhr_s.yaml",
        help="Path to YAML config file (default: w32_256x256_adam_lr1e-3_ak_vhr_s.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="vhr_birdpose_s_add.pth",
        help="Path to PyTorch checkpoint file (default: vhr_birdpose_s_add.pth)",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="checkpoints/vhr_birdpose_s_add_raw.onnx",
        help="Path to ONNX model (default: checkpoints/vhr_birdpose_s_add_raw.onnx)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VHR-BirdPose: PyTorch vs ONNX Comparison")
    print("=" * 80)

    # Load and preprocess image
    try:
        input_tensor, original_image, scale_factor = preprocess_image(args.image)
        print(f"✓ Image loaded: {args.image}")
        print(f"  Original size: {original_image.shape[:2]}")
        print(f"  Scale factor: {scale_factor:.3f}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return 1

    results = []

    # Test PyTorch model
    print("\nTesting PyTorch model...")
    try:
        if not os.path.exists(args.checkpoint):
            print(f"  ✗ Checkpoint not found: {args.checkpoint}")
        elif not os.path.exists(args.config):
            print(f"  ✗ Config not found: {args.config}")
        else:
            heatmaps_pt, time_pt = run_pytorch_inference(
                args.config, args.checkpoint, input_tensor
            )
            keypoints_pt = postprocess_heatmaps(
                heatmaps_pt, original_image.shape[:2], scale_factor
            )
            size_pt = os.path.getsize(args.checkpoint) / (1024 * 1024)

            confidences_pt = [conf for _, _, conf in keypoints_pt]
            valid_pt = sum(1 for conf in confidences_pt if conf > 0.3)

            results.append(
                {
                    "name": "PyTorch .pth",
                    "size_mb": size_pt,
                    "time": time_pt,
                    "keypoints": keypoints_pt,
                    "valid_kpts": valid_pt,
                    "avg_conf": np.mean(confidences_pt),
                }
            )

            print(f"  ✓ Size: {size_pt:.1f} MB")
            print(f"  ✓ Inference time: {time_pt:.3f}s")
            print(f"  ✓ Valid keypoints: {valid_pt}/23")
            print(f"  ✓ Average confidence: {np.mean(confidences_pt):.3f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test ONNX model
    print("\nTesting ONNX model...")
    try:
        if not os.path.exists(args.onnx_model):
            print(f"  ✗ ONNX model not found: {args.onnx_model}")
        else:
            heatmaps_onnx, time_onnx = run_onnx_inference(args.onnx_model, input_tensor)
            keypoints_onnx = postprocess_heatmaps(
                heatmaps_onnx, original_image.shape[:2], scale_factor
            )
            size_onnx = os.path.getsize(args.onnx_model) / (1024 * 1024)

            confidences_onnx = [conf for _, _, conf in keypoints_onnx]
            valid_onnx = sum(1 for conf in confidences_onnx if conf > 0.3)

            results.append(
                {
                    "name": "ONNX",
                    "size_mb": size_onnx,
                    "time": time_onnx,
                    "keypoints": keypoints_onnx,
                    "valid_kpts": valid_onnx,
                    "avg_conf": np.mean(confidences_onnx),
                }
            )

            print(f"  ✓ Size: {size_onnx:.1f} MB")
            print(f"  ✓ Inference time: {time_onnx:.3f}s")
            print(f"  ✓ Valid keypoints: {valid_onnx}/23")
            print(f"  ✓ Average confidence: {np.mean(confidences_onnx):.3f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Comparison summary
    if len(results) == 2:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(
            f"{'Model':<15} {'Size (MB)':<12} {'Time (s)':<10} {'Keypoints':<12} {'Avg Conf':<10}"
        )
        print("-" * 80)

        for result in results:
            print(
                f"{result['name']:<15} {result['size_mb']:<12.1f} {result['time']:<10.3f} {result['valid_kpts']}/23{'':<7} {result['avg_conf']:<10.3f}"
            )

        # Speed comparison
        if len(results) == 2:
            speedup = results[0]["time"] / results[1]["time"]
            faster_model = results[0]["name"] if speedup < 1 else results[1]["name"]
            speedup = max(speedup, 1 / speedup)
            print(f"\n{faster_model} is {speedup:.2f}x faster")

            # Check if results are identical
            pt_kpts = results[0]["keypoints"]
            onnx_kpts = results[1]["keypoints"]

            # Compare keypoint coordinates (allowing small floating point differences)
            identical = True
            max_diff = 0
            for i, ((x1, y1, c1), (x2, y2, c2)) in enumerate(zip(pt_kpts, onnx_kpts)):
                diff = abs(x1 - x2) + abs(y1 - y2) + abs(c1 - c2)
                max_diff = max(max_diff, diff)
                if diff > 0.01:  # Allow small floating point differences
                    identical = False
                    break

            if identical:
                print("✓ Results are identical (max diff: {:.6f})".format(max_diff))
            else:
                print(f"⚠ Results differ (max diff: {max_diff:.6f})")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
