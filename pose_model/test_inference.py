#!/usr/bin/env python3
"""
VHR-BirdPose Inference Testing Script

This script tests inference with multiple ONNX model variants:
- Raw ONNX model
- Optimized ONNX model
- INT8 quantized model

It loads an example image, runs pose estimation, and visualizes results.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import yaml
import onnxruntime as ort
from typing import Tuple, List
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Preprocess image for VHR-BirdPose model.

    Args:
        image_path: Path to input image
        target_size: Target size (height, width) for model input

    Returns:
        Tuple of (processed_tensor, original_image, scale_factor)
    """
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

    return processed, original_image, scale_factor


def confidence_method_peak_value(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Method 1: Peak value confidence - most interpretable"""
    if len(heatmaps.shape) == 4:
        B, J, H, W = heatmaps.shape
        reshaped = heatmaps.reshape(B, J, -1)
        peak_vals = np.max(reshaped, axis=-1)  # (B, J)
        peak_indices = np.argmax(reshaped, axis=-1)  # (B, J)
        return peak_vals, peak_indices
    else:
        J, H, W = heatmaps.shape
        reshaped = heatmaps.reshape(J, -1)
        peak_vals = np.max(reshaped, axis=-1)  # (J,)
        peak_indices = np.argmax(reshaped, axis=-1)  # (J,)
        return peak_vals, peak_indices


def postprocess_heatmaps(
    heatmaps: np.ndarray, original_size: Tuple[int, int], scale_factor: float
) -> List[Tuple[int, int, float]]:
    """
    Postprocess heatmaps to get keypoint coordinates with peak value confidence.

    Args:
        heatmaps: Model output heatmaps (1, num_joints, H, W)
        original_size: Original image size (height, width)
        scale_factor: Scale factor used in preprocessing

    Returns:
        List of (x, y, confidence) tuples for each keypoint
    """
    if len(heatmaps.shape) == 4:
        heatmaps_work = heatmaps[0]  # Remove batch dimension
    else:
        heatmaps_work = heatmaps

    num_joints, heatmap_h, heatmap_w = heatmaps_work.shape
    original_h, original_w = original_size

    # Get confidence using peak value method
    peak_vals, peak_indices = confidence_method_peak_value(heatmaps)

    # Handle batch dimension
    if len(peak_vals.shape) > 1:
        peak_vals = peak_vals[0]
        peak_indices = peak_indices[0]

    keypoints = []
    for joint_idx in range(num_joints):
        # Convert peak index to 2D coordinates
        peak_idx = peak_indices[joint_idx]
        heatmap_y = peak_idx // heatmap_w
        heatmap_x = peak_idx % heatmap_w

        confidence = peak_vals[joint_idx]

        # Scale heatmap coordinates to input image size (256x256)
        target_h, target_w = 256, 256
        scale_heatmap_to_input = target_w / heatmap_w  # Should be 4.0 for 64->256

        input_x = heatmap_x * scale_heatmap_to_input
        input_y = heatmap_y * scale_heatmap_to_input

        # Account for padding that was added during preprocessing
        new_width = int(original_w * scale_factor)
        new_height = int(original_h * scale_factor)
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2

        # Remove padding offset to get coordinates in the resized (but unpadded) image
        input_x -= x_offset
        input_y -= y_offset

        # Scale back to original image size
        orig_x = int(input_x / scale_factor)
        orig_y = int(input_y / scale_factor)

        # Ensure coordinates are within image bounds
        orig_x = max(0, min(orig_x, original_w - 1))
        orig_y = max(0, min(orig_y, original_h - 1))

        keypoints.append((orig_x, orig_y, float(confidence)))

    return keypoints


def visualize_pose(
    image: np.ndarray,
    keypoints: List[Tuple[int, int, float]],
    title: str = "Pose Estimation",
    confidence_threshold: float = 0.3,
) -> plt.Figure:
    """
    Visualize pose estimation results.

    Args:
        image: Original image (RGB)
        keypoints: List of (x, y, confidence) tuples
        title: Plot title
        confidence_threshold: Minimum confidence to display keypoint

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    # Animal Kingdom bird pose keypoint names (23 keypoints)
    # Official keypoint order from Animal Kingdom dataset
    keypoint_names = [
        "Head_Mid_Top",  # 0
        "Eye_Left",  # 1
        "Eye_Right",  # 2
        "Mouth_Front_Top",  # 3
        "Mouth_Back_Left",  # 4
        "Mouth_Back_Right",  # 5
        "Mouth_Front_Bottom",  # 6
        "Shoulder_Left",  # 7
        "Shoulder_Right",  # 8
        "Elbow_Left",  # 9
        "Elbow_Right",  # 10
        "Wrist_Left",  # 11
        "Wrist_Right",  # 12
        "Torso_Mid_Back",  # 13
        "Hip_Left",  # 14
        "Hip_Right",  # 15
        "Knee_Left",  # 16
        "Knee_Right",  # 17
        "Ankle_Left",  # 18
        "Ankle_Right",  # 19
        "Tail_Top_Back",  # 20
        "Tail_Mid_Back",  # 21
        "Tail_End_Back",  # 22
    ]

    # Color map for different keypoints
    colors = plt.cm.tab20(np.linspace(0, 1, len(keypoint_names)))

    # Plot keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            color = colors[i]
            ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=2,
            )

            # Add keypoint label
            keypoint_name = keypoint_names[i] if i < len(keypoint_names) else f"kp_{i}"
            ax.annotate(
                f"{keypoint_name}\n({conf:.2f})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            )

    # Add confidence statistics
    valid_confs = [conf for _, _, conf in keypoints if conf > confidence_threshold]
    if valid_confs:
        stats_text = f"Valid keypoints: {len(valid_confs)}/{len(keypoints)}\n"
        stats_text += f"Avg confidence: {np.mean(valid_confs):.3f}\n"
        stats_text += f"Min confidence: {np.min(valid_confs):.3f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    return fig


def run_inference(
    model_path: str, input_tensor: np.ndarray, providers: List[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Run inference with ONNX model.

    Args:
        model_path: Path to ONNX model
        input_tensor: Input tensor (1, 3, H, W)
        providers: ONNX Runtime providers

    Returns:
        Tuple of (output_heatmaps, inference_time)
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Create inference session
    session = ort.InferenceSession(model_path, providers=providers)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_time

    # Extract heatmaps (assuming first output)
    heatmaps = outputs[0]

    return heatmaps, inference_time


def test_model(
    model_path: str,
    model_name: str,
    input_tensor: np.ndarray,
    original_image: np.ndarray,
    scale_factor: float,
    confidence_threshold: float = 0.3,
) -> dict:
    """Test a single model and return results."""
    if not os.path.exists(model_path):
        return {
            "name": model_name,
            "path": model_path,
            "status": "missing",
            "error": f"Model file not found: {model_path}",
        }

    try:
        # Get model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # Run inference
        heatmaps, inference_time = run_inference(model_path, input_tensor)

        # Postprocess results
        keypoints = postprocess_heatmaps(
            heatmaps, original_image.shape[:2], scale_factor
        )

        # Calculate statistics
        confidences = [conf for _, _, conf in keypoints]
        avg_confidence = np.mean(confidences)
        valid_keypoints = sum(1 for conf in confidences if conf > confidence_threshold)

        return {
            "name": model_name,
            "path": model_path,
            "status": "success",
            "size_mb": model_size_mb,
            "inference_time": inference_time,
            "keypoints": keypoints,
            "avg_confidence": avg_confidence,
            "valid_keypoints": valid_keypoints,
            "total_keypoints": len(keypoints),
        }

    except Exception as e:
        return {
            "name": model_name,
            "path": model_path,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Test VHR-BirdPose ONNX models")
    parser.add_argument(
        "--image",
        type=str,
        default="example.jpeg",
        help="Path to input image (default: example.jpeg)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="checkpoints",
        help="Working directory containing ONNX models (default: checkpoints)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for visualization (default: 0.3)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save visualization results to files",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VHR-BirdPose ONNX Model Inference Testing")
    print("=" * 80)

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"✗ Error: Image file not found: {args.image}")
        return 1

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    try:
        input_tensor, original_image, scale_factor = preprocess_image(args.image)
        print("✓ Image loaded and preprocessed")
        print(f"  Original size: {original_image.shape[:2]}")
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  Input tensor shape: {input_tensor.shape}")
    except Exception as e:
        print(f"✗ Error preprocessing image: {e}")
        return 1

    # Define models to test
    models = [
        ("Raw ONNX", os.path.join(args.workdir, "vhr_birdpose_s_add_raw.onnx")),
        ("Simplified", os.path.join(args.workdir, "vhr_birdpose_s_add_sim.onnx")),
        ("Optimized", os.path.join(args.workdir, "vhr_birdpose_s_add_opt.onnx")),
        ("Preprocessed", os.path.join(args.workdir, "vhr_birdpose_s_add_preproc.onnx")),
        ("INT8 Quantized", os.path.join(args.workdir, "vhr_birdpose_s_add_int8.onnx")),
    ]

    print(f"\nTesting {len(models)} model variants...")
    print("-" * 80)

    results = []
    for model_name, model_path in models:
        print(f"Testing {model_name}...")
        result = test_model(
            model_path,
            model_name,
            input_tensor,
            original_image,
            scale_factor,
            args.confidence_threshold,
        )
        results.append(result)

        if result["status"] == "success":
            print(f"  ✓ Size: {result['size_mb']:.1f} MB")
            print(f"  ✓ Inference time: {result['inference_time']:.3f}s")
            print(
                f"  ✓ Valid keypoints: {result['valid_keypoints']}/{result['total_keypoints']}"
            )
            print(f"  ✓ Average confidence: {result['avg_confidence']:.3f}")
        elif result["status"] == "missing":
            print(f"  ⚠ Skipped: {result['error']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
        print()

    # Summary table
    print("Summary Results:")
    print("-" * 100)
    print(
        f"{'Model':<15} {'Status':<10} {'Size (MB)':<12} {'Time (s)':<10} {'Keypoints':<12} {'Avg Conf':<10}"
    )
    print("-" * 100)

    for result in results:
        if result["status"] == "success":
            keypoints_str = f"{result['valid_keypoints']}/{result['total_keypoints']}"
            print(
                f"{result['name']:<15} {'✓':<10} {result['size_mb']:<12.1f} {result['inference_time']:<10.3f} {keypoints_str:<12} {result['avg_confidence']:<10.3f}"
            )
        else:
            status_symbol = "⚠" if result["status"] == "missing" else "✗"
            print(
                f"{result['name']:<15} {status_symbol:<10} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<10}"
            )

    # Generate visualizations
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        print(
            f"\nGenerating visualizations for {len(successful_results)} successful models..."
        )

        # Create subplots
        n_models = len(successful_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, result in enumerate(successful_results):
            ax = axes[i]
            ax.imshow(original_image)
            ax.set_title(
                f"{result['name']}\n{result['size_mb']:.1f}MB, {result['inference_time']:.3f}s"
            )
            ax.axis("off")

            # Plot keypoints
            keypoints = result["keypoints"]
            colors = plt.cm.tab20(np.linspace(0, 1, len(keypoints)))

            # Animal Kingdom keypoint names for labels
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

            for j, (x, y, conf) in enumerate(keypoints):
                if conf > args.confidence_threshold:
                    ax.plot(
                        x,
                        y,
                        "o",
                        color=colors[j],
                        markersize=6,
                        markeredgecolor="white",
                        markeredgewidth=1,
                    )

                    # Add keypoint label with confidence
                    keypoint_name = (
                        keypoint_names[j] if j < len(keypoint_names) else f"kp_{j}"
                    )
                    ax.annotate(
                        f"{keypoint_name} ({conf:.2f})",
                        (x, y),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=6,
                        color="white",
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor=colors[j], alpha=0.7
                        ),
                    )

        # Hide unused subplots
        for i in range(len(successful_results), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        if args.save_results:
            output_path = "inference_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✓ Comparison visualization saved to: {output_path}")

        # plt.show()

        # Create detailed visualization for best model
        best_result = min(successful_results, key=lambda x: x["inference_time"])
        print(
            f"\nGenerating detailed visualization for fastest model: {best_result['name']}"
        )

        detailed_fig = visualize_pose(
            original_image,
            best_result["keypoints"],
            f"VHR-BirdPose: {best_result['name']}",
            args.confidence_threshold,
        )

        if args.save_results:
            output_path = (
                f"detailed_{best_result['name'].lower().replace(' ', '_')}.png"
            )
            detailed_fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✓ Detailed visualization saved to: {output_path}")

        # plt.show()

    print("\n" + "=" * 80)
    print("Inference testing completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
