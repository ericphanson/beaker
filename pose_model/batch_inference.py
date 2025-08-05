#!/usr/bin/env python3
"""
Batch VHR-BirdPose Inference Script

This script performs batch inference on a directory of images using the VHR-BirdPose model.
It supports both ONNX and CoreML backends for fast inference on 500+ images.

Usage:
    python batch_inference.py --input-dir /path/to/images --output-dir /path    # Setup model session
    print(f"\nInitializing {args.backend.upper()} session...")
    if args.backend == 'onnx':
        session_info = setup_onnx_session(str(args.model_path), use_coreml=False)
    elif args.backend == 'coreml':
        session_info = setup_onnx_session(str(args.model_path), use_coreml=True)t
    python batch_inference.py --input-dir /path/to/images --output-dir /path/to/output --backend coreml
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from adjustText import adjust_text

# Import functions from test_inference.py
from test_inference import preprocess_image, postprocess_heatmaps, visualize_pose


def load_images_from_directory(
    input_dir: Path, extensions: List[str] = None
) -> List[Path]:
    """
    Load all image files from a directory.

    Args:
        input_dir: Path to input directory
        extensions: List of allowed extensions (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def setup_onnx_session(model_path: str, use_coreml: bool = False):
    """Setup ONNX Runtime session with optional CoreML execution provider."""
    try:
        import onnxruntime as ort

        # Configure session options for performance
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 0  # Use all available cores
        sess_options.intra_op_num_threads = 0  # Use all available cores
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Choose providers based on backend
        providers = []
        if use_coreml:
            # CoreML execution provider for Apple Silicon
            providers.append(
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "0",
                        "EnableOnSubgraphs": "0",
                    },
                )
            )

        # Add fallback providers
        if ort.get_device() == "GPU":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        input_name = session.get_inputs()[0].name

        active_providers = session.get_providers()
        print(f"✓ ONNX session initialized with providers: {active_providers}")

        if use_coreml and "CoreMLExecutionProvider" in active_providers:
            print("  ✓ CoreML acceleration enabled for Apple Silicon")
        elif use_coreml:
            print("  ⚠ CoreML provider requested but not available, using fallback")

        return session, input_name

    except ImportError:
        print("✗ Error: onnxruntime not available")
        sys.exit(1)


def run_onnx_inference(
    session, input_name: str, input_tensor: np.ndarray
) -> np.ndarray:
    """Run inference with ONNX model."""
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0]


def process_single_image(
    image_path: Path,
    session_info: Tuple,
    backend: str,
    confidence_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Process a single image and return results.

    Args:
        image_path: Path to image file
        session_info: Model session information (differs by backend)
        backend: 'onnx' or 'coreml'
        confidence_threshold: Minimum confidence threshold

    Returns:
        Dictionary with results
    """
    try:
        # Load and preprocess image
        start_time = time.time()
        input_tensor, original_image, scale_factor = preprocess_image(str(image_path))
        preprocess_time = time.time() - start_time

        # Run inference
        start_time = time.time()
        if backend == "onnx" or backend == "coreml":
            # Both backends now use ONNX Runtime (coreml uses optimized ONNX Runtime)
            session, input_name = session_info
            heatmaps = run_onnx_inference(session, input_name, input_tensor)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        inference_time = time.time() - start_time

        # Postprocess results
        start_time = time.time()
        keypoints = postprocess_heatmaps(
            heatmaps, original_image.shape[:2], scale_factor
        )
        postprocess_time = time.time() - start_time

        # Calculate statistics
        confidences = [conf for _, _, conf in keypoints]
        valid_keypoints = sum(1 for conf in confidences if conf > confidence_threshold)

        # Convert keypoints to dictionary format for JSON output
        keypoints_dict = convert_keypoints_to_dict(keypoints)

        return {
            "status": "success",
            "image_path": str(image_path),
            "image_shape": original_image.shape[:2],
            "keypoints": keypoints_dict,
            "keypoints_raw": keypoints,  # Keep original for visualization
            "statistics": {
                "total_keypoints": len(keypoints),
                "valid_keypoints": valid_keypoints,
                "avg_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
            },
            "timing": {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "postprocess_time": postprocess_time,
                "total_time": preprocess_time + inference_time + postprocess_time,
            },
        }

    except Exception as e:
        return {"status": "error", "image_path": str(image_path), "error": str(e)}


def filter_keypoints_for_visualization(
    keypoints: List[Tuple[int, int, float]],
) -> List[Tuple[int, int, float, str]]:
    """
    Filter keypoints to show only the most important ones for cleaner visualization.

    Returns:
        List of (x, y, confidence, name) tuples for the selected keypoints
    """
    # Animal Kingdom keypoint names
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

    # Add names to keypoints
    named_keypoints = []
    for i, (x, y, conf) in enumerate(keypoints):
        name = keypoint_names[i] if i < len(keypoint_names) else f"kp_{i}"
        named_keypoints.append((x, y, conf, name, i))

    selected_keypoints = []

    # 1. Top 4 confidence keypoints overall
    sorted_by_conf = sorted(named_keypoints, key=lambda x: x[2], reverse=True)
    top_4 = sorted_by_conf[:4]
    selected_keypoints.extend(top_4)

    # 2. Highest confidence mouth keypoint
    mouth_keypoints = [kp for kp in named_keypoints if "Mouth" in kp[3]]
    if mouth_keypoints:
        best_mouth = max(mouth_keypoints, key=lambda x: x[2])
        if best_mouth not in selected_keypoints:
            selected_keypoints.append(best_mouth)

    # 3. Highest confidence eye keypoint
    eye_keypoints = [kp for kp in named_keypoints if "Eye" in kp[3]]
    if eye_keypoints:
        best_eye = max(eye_keypoints, key=lambda x: x[2])
        if best_eye not in selected_keypoints:
            selected_keypoints.append(best_eye)

    # 4. Highest confidence tail keypoint
    tail_keypoints = [kp for kp in named_keypoints if "Tail" in kp[3]]
    if tail_keypoints:
        best_tail = max(tail_keypoints, key=lambda x: x[2])
        if best_tail not in selected_keypoints:
            selected_keypoints.append(best_tail)

    # Remove duplicates and return (x, y, confidence, name)
    unique_keypoints = []
    seen_indices = set()
    for x, y, conf, name, idx in selected_keypoints:
        if idx not in seen_indices:
            unique_keypoints.append((x, y, conf, name))
            seen_indices.add(idx)

    return unique_keypoints


def save_visualization(
    image_path: Path,
    keypoints: List[Tuple[int, int, float]],
    output_dir: Path,
    confidence_threshold: float = 0.3,
):
    """Save detailed visualization for an image with filtered keypoints."""
    try:
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Filter keypoints for cleaner visualization
        filtered_keypoints = filter_keypoints_for_visualization(keypoints)

        # Create visualization with filtered keypoints
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        ax.set_title(f"VHR-BirdPose: {image_path.name}")
        ax.axis("off")

        # Color map for keypoints
        colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_keypoints)))

        max_confidence = max(conf for _, _, conf in keypoints) if keypoints else 0

        # Plot filtered keypoints and collect text objects for adjustText
        texts = []
        for i, (x, y, conf, name) in enumerate(filtered_keypoints):
            if conf > confidence_threshold:
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

                # Create text object for adjustText
                text = ax.text(
                    x,
                    y,
                    f"{name}\n({conf:.2f})",
                    fontsize=9,
                    color="white",
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                )
                texts.append(text)

        # Use adjustText to prevent label overlap
        if texts:
            # Extract x, y coordinates for repulsion from points
            x_coords = [
                x
                for x, y, conf, name in filtered_keypoints
                if conf > confidence_threshold
            ]
            y_coords = [
                y
                for x, y, conf, name in filtered_keypoints
                if conf > confidence_threshold
            ]

            adjust_text(
                texts,
                x=x_coords,
                y=y_coords,
                # arrowprops=dict(arrowstyle='->', color='white', alpha=0.7, lw=1),
                expand_points=(1.3, 1.3),
                expand_text=(1.3, 1.3),
                max_move=(30.0, 30.0),
                force_points=0.5,
                force_text=0.5,
            )

        # Add statistics with max confidence
        stats_text = f"Showing {len([kp for kp in filtered_keypoints if kp[2] > confidence_threshold])}/{len(filtered_keypoints)} selected keypoints\n"
        stats_text += f"Max confidence: {max_confidence:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=11,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
        )

        # Save visualization
        output_path = output_dir / f"{image_path.stem}_pose.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Clean up memory

        return str(output_path)

    except Exception as e:
        print(f"⚠ Warning: Failed to save visualization for {image_path}: {e}")
        return None
    """Save detailed visualization for an image."""
    try:
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create visualization
        fig = visualize_pose(
            image,
            keypoints,
            title=f"VHR-BirdPose: {image_path.name}",
            confidence_threshold=confidence_threshold,
        )

        # Save visualization
        output_path = output_dir / f"{image_path.stem}_pose.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Clean up memory

        return str(output_path)

    except Exception as e:
        print(f"⚠ Warning: Failed to save visualization for {image_path}: {e}")
        return None


def convert_keypoints_to_dict(
    keypoints: List[Tuple[int, int, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Convert keypoints list to dictionary format with keypoint names as keys.

    Returns:
        Dictionary with keypoint names as keys and {x, y, confidence} as values
    """
    # Animal Kingdom keypoint names
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

    keypoints_dict = {}
    for i, (x, y, conf) in enumerate(keypoints):
        name = keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}"
        keypoints_dict[name] = {"x": int(x), "y": int(y), "confidence": float(conf)}

    return keypoints_dict


def main():
    parser = argparse.ArgumentParser(description="Batch VHR-BirdPose inference")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("checkpoints/vhr_birdpose_s_add_int8.onnx"),
        help="Path to model file (default: INT8 quantized ONNX)",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "coreml"],
        default="onnx",
        help="Inference backend: 'onnx' (standard) or 'coreml' (optimized for Apple Silicon) (default: onnx)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for visualization (default: 0.3)",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save detailed visualization images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        print(f"✗ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"✗ Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VHR-BirdPose Batch Inference")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Load image files
    print("\nLoading images...")
    image_files = load_images_from_directory(args.input_dir)
    print(f"✓ Found {len(image_files)} images")

    if len(image_files) == 0:
        print("✗ No images found in input directory")
        sys.exit(1)

    # Setup model session
    print(f"\nInitializing {args.backend.upper()} session...")
    if args.backend == "onnx":
        session_info = setup_onnx_session(str(args.model_path), use_coreml=False)
    elif args.backend == "coreml":
        session_info = setup_onnx_session(str(args.model_path), use_coreml=True)

    # Process images
    print(f"\nProcessing {len(image_files)} images...")
    results = []
    total_inference_time = 0
    successful_count = 0

    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_single_image(
            image_path, session_info, args.backend, args.confidence_threshold
        )

        results.append(result)

        if result["status"] == "success":
            successful_count += 1
            total_inference_time += result["timing"]["total_time"]

            # Save visualization if requested (use raw keypoints for visualization)
            if args.save_visualizations:
                save_visualization(
                    image_path,
                    result["keypoints_raw"],
                    args.output_dir,
                    args.confidence_threshold,
                )

    # Save JSON results
    json_output_path = args.output_dir / "batch_inference_results.json"

    # Clean results for JSON output (remove raw keypoints used for visualization)
    clean_results = []
    for result in results:
        clean_result = result.copy()
        if "keypoints_raw" in clean_result:
            del clean_result["keypoints_raw"]
        clean_results.append(clean_result)

    # Prepare summary
    summary = {
        "configuration": {
            "input_directory": str(args.input_dir),
            "output_directory": str(args.output_dir),
            "model_path": str(args.model_path),
            "backend": args.backend,
            "confidence_threshold": args.confidence_threshold,
            "save_visualizations": args.save_visualizations,
        },
        "summary": {
            "total_images": len(image_files),
            "successful_images": successful_count,
            "failed_images": len(image_files) - successful_count,
            "avg_inference_time": total_inference_time / successful_count
            if successful_count > 0
            else 0,
            "total_processing_time": total_inference_time,
        },
        "results": clean_results,
    }

    with open(json_output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {json_output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(image_files) - successful_count}")
    if successful_count > 0:
        print(
            f"Average inference time: {total_inference_time / successful_count:.3f}s per image"
        )
        print(f"Total processing time: {total_inference_time:.2f}s")

        # Calculate aggregate statistics
        all_valid_keypoints = []
        all_avg_confidences = []
        for result in results:
            if result["status"] == "success":
                all_valid_keypoints.append(result["statistics"]["valid_keypoints"])
                all_avg_confidences.append(result["statistics"]["avg_confidence"])

        if all_valid_keypoints:
            print(f"Average valid keypoints: {np.mean(all_valid_keypoints):.1f}/23")
            print(f"Average confidence: {np.mean(all_avg_confidences):.3f}")

    if args.save_visualizations:
        print(f"✓ Visualizations saved to: {args.output_dir}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
