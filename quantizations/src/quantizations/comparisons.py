"""
Generate comparison images and figures for quantized models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def load_and_preprocess_image(
    image_path: Path, target_size: tuple = (640, 640)
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load and preprocess an image for model inference."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original size for scaling
    original_height, original_width = image.shape[:2]

    # Resize to target size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] and convert to CHW format
    input_image = resized_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    return input_image, resized_image, (original_width, original_height)


def run_model_inference(model_path: Path, input_data: np.ndarray) -> np.ndarray:
    """Run inference on a model."""
    try:
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        return outputs[0] if outputs else np.array([])  # type: ignore
    except Exception as e:
        logger.error(f"Inference failed for {model_path}: {e}")
        raise


def parse_detections(
    output: np.ndarray, confidence_threshold: float = 0.5
) -> List[Dict[str, float]]:
    """Parse detection output into bounding boxes."""
    detections = []

    # Handle different output formats
    if len(output.shape) == 3 and output.shape[0] == 1:
        output = output[0]  # Remove batch dimension

    if len(output.shape) == 2:
        # Format: [num_detections, 6] where each detection is [x1, y1, x2, y2, confidence, class]
        for detection in output:
            if len(detection) >= 5:
                x1, y1, x2, y2, conf = detection[:5]
                if conf >= confidence_threshold:
                    detections.append(
                        {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "confidence": float(conf),
                        }
                    )

    return detections


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Calculate intersection
    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def match_detections(
    baseline_detections: List[Dict], quantized_detections: List[Dict]
) -> List[Tuple[Dict, Dict, float]]:
    """Match detections between baseline and quantized models and calculate IoU."""
    matches = []

    for baseline_det in baseline_detections:
        best_match = None
        best_iou = 0.0

        for quant_det in quantized_detections:
            iou = calculate_iou(baseline_det, quant_det)
            if iou > best_iou:
                best_iou = iou
                best_match = quant_det

        if best_match is not None:
            matches.append((baseline_det, best_match, best_iou))
        else:
            # No match found
            matches.append((baseline_det, None, 0.0))

    return matches


def create_detection_comparison_figure(
    image_path: Path,
    model_results: Dict[str, Tuple[Path, List[Dict]]],
    output_path: Path,
    confidence_threshold: float = 0.5,
) -> None:
    """Create a comparison figure showing detections from multiple models."""
    try:
        # Load the original image
        input_data, display_image, original_size = load_and_preprocess_image(image_path)

        # Create subplot figure
        num_models = len(model_results)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        if num_models == 1:
            axes = [axes]

        baseline_detections = None

        for idx, (model_name, (model_path, detections)) in enumerate(
            model_results.items()
        ):
            ax = axes[idx]
            ax.imshow(display_image)
            ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
            ax.axis("off")

            # Store baseline detections (first model)
            if idx == 0:
                baseline_detections = detections

            # Draw bounding boxes
            colors = ["red", "blue", "green", "orange", "purple"]
            for det_idx, detection in enumerate(detections):
                color = colors[det_idx % len(colors)]
                x1, y1, x2, y2 = (
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
                )
                conf = detection["confidence"]

                # Create rectangle
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add confidence text
                ax.text(
                    x1,
                    y1 - 5,
                    f"{conf:.2f}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )

            # Calculate and display IoU if not baseline
            if idx > 0 and baseline_detections:
                matches = match_detections(baseline_detections, detections)
                avg_iou = np.mean([match[2] for match in matches]) if matches else 0.0
                ax.text(
                    0.02,
                    0.98,
                    f"Avg IoU: {avg_iou:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment="top",
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Comparison figure saved: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create comparison figure: {e}")
        raise


def generate_model_comparison_images(
    models: Dict[str, Path],
    test_images: List[Path],
    output_dir: Path,
    confidence_threshold: float = 0.5,
) -> List[Path]:
    """Generate comparison images for multiple models on test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_files = []

    # Determine model type from the output directory or model names
    model_type_prefix = ""
    if output_dir.name in ["head", "cutout"]:
        model_type_prefix = f"{output_dir.name}-"
    else:
        # Try to infer from model paths
        for model_name, model_path in models.items():
            if "head" in str(model_path).lower():
                model_type_prefix = "head-"
                break
            elif "cutout" in str(model_path).lower():
                model_type_prefix = "cutout-"
                break

    for image_path in test_images:
        logger.info(f"Processing comparison for {image_path.name}")

        # Run inference on all models
        model_results = {}

        for model_name, model_path in models.items():
            try:
                input_data, _, _ = load_and_preprocess_image(image_path)
                output = run_model_inference(model_path, input_data)
                detections = parse_detections(output, confidence_threshold)
                model_results[model_name] = (model_path, detections)
                logger.debug(f"{model_name}: Found {len(detections)} detections")
            except Exception as e:
                logger.warning(f"Failed to run {model_name} on {image_path.name}: {e}")
                model_results[model_name] = (model_path, [])

        if model_results:
            # Create comparison figure with model-type prefix
            output_filename = f"{model_type_prefix}comparison_{image_path.stem}.png"
            output_path = output_dir / output_filename

            create_detection_comparison_figure(
                image_path, model_results, output_path, confidence_threshold
            )
            comparison_files.append(output_path)

    return comparison_files


def create_performance_table(
    models: Dict[str, Path],
    test_images: List[Path],
    timing_results: Dict[str, Dict[str, float]],
    validation_results: Dict[str, Dict[str, float]],
) -> str:
    """Create a markdown table with performance metrics."""

    table_lines = [
        "| Model | Size (MB) | Inference (ms) | Cosine Similarity | RMSE | Max Diff | Size Reduction |",
        "|-------|-----------|----------------|-------------------|------|----------|----------------|",
    ]

    baseline_size = None
    for model_name, model_path in models.items():
        # Get model size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if baseline_size is None:
            baseline_size = size_mb

        # Get timing
        timing = timing_results.get(model_name, {})
        inference_ms = timing.get("mean_ms", 0)

        # Get validation metrics
        validation = validation_results.get(model_name, {})
        cosine_sim = validation.get("avg_cosine_similarity", 1.0)
        rmse = validation.get("avg_rmse", 0.0)
        max_diff = validation.get("max_max_absolute_diff", 0.0)

        # Calculate size reduction
        size_reduction = (
            ((baseline_size - size_mb) / baseline_size * 100) if baseline_size else 0
        )

        table_lines.append(
            f"| {model_name} | {size_mb:.1f} | {inference_ms:.1f} ± {timing.get('std_ms', 0):.1f} | "
            f"{cosine_sim:.6f} | {rmse:.2f} | {max_diff:.2f} | {size_reduction:.1f}% |"
        )

    return "\n".join(table_lines)
