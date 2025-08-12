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

# Try relative import first, fall back to direct import
try:
    from .yolov8 import YOLOv8
except ImportError:
    try:
        from yolov8 import YOLOv8
    except ImportError:
        # If both fail, try importing from the same directory
        import sys
        import os

        sys.path.insert(0, os.path.dirname(__file__))
        from yolov8 import YOLOv8

logger = logging.getLogger(__name__)


def load_and_preprocess_image(
    image_path: Path, model_type: str = "detect"
) -> np.ndarray:
    """Load and preprocess an image for model inference."""
    if model_type == "detect":
        target_size = (960, 960)  # Use default size for detect models
    else:
        target_size = (1024, 1024)  # Use default size for other models
    if model_type == "detect":
        # For detect models, we'll return the image path and let YOLOv8 handle preprocessing
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        # For detect models, return the original image path and display image
        return str(image_path), display_image, (original_width, original_height)
    else:
        # Keep original preprocessing for other model types (like "cutout")
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


def run_model_inference(
    model_path: Path,
    input_data,
    model_type: str = "detect",
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
):
    """Run inference on a model."""
    try:
        if model_type == "detect":
            # Use YOLOv8 class for detect models
            detector = YOLOv8(
                str(model_path), input_data, confidence_threshold, iou_threshold
            )
            # Run the full inference pipeline and return the detections in our format
            output_image = detector.main()

            # Extract detections from the YOLOv8 detector
            # We need to re-run the inference to get the raw detections before drawing
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            model_inputs = session.get_inputs()
            input_shape = model_inputs[0].shape
            detector.input_width = input_shape[2]
            detector.input_height = input_shape[3]

            # Preprocess using YOLOv8 methods
            img_data, pad = detector.preprocess()

            # Run inference
            outputs = session.run(None, {model_inputs[0].name: img_data})

            # Parse the raw outputs to get detections in our format
            detections = parse_yolov8_detections(
                outputs[0], detector, pad, confidence_threshold
            )

            return detections
        else:
            # Original inference for other model types
            providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(model_path), providers=providers)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            return outputs[0] if outputs else np.array([])
    except Exception as e:
        logger.error(f"Inference failed for {model_path}: {e}")
        raise


def parse_yolov8_detections(
    output: np.ndarray, detector: YOLOv8, pad: tuple, confidence_threshold: float = 0.5
) -> List[Dict[str, float]]:
    """Parse YOLOv8 detection output into bounding boxes using YOLOv8's postprocessing logic."""
    detections = []

    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Calculate the scaling factors for the bounding box coordinates
    gain = min(
        detector.input_height / detector.img_height,
        detector.input_width / detector.img_width,
    )
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_threshold:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, confidence_threshold, detector.iou_thres
        )

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Convert to x1, y1, x2, y2 format
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h

            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(score),
                    "class_id": int(class_id),
                }
            )

    return detections


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
    model_type: str = "detect",
) -> None:
    if model_type == "detect":
        imgsz = (960, 960)  # Use default size for detect models
    else:
        imgsz = (1024, 1024)  # Use default size for other models
    """Create a comparison figure showing detections from multiple models."""
    try:
        # Load the original image
        input_data, display_image, original_size = load_and_preprocess_image(
            image_path, imgsz, model_type
        )

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
    iou_threshold: float = 0.5,
    model_type: str = "detect",
) -> List[Path]:
    """Generate comparison images for multiple models on test images."""

    if model_type == "detect":
        imgsz = (960, 960)  # Use default size for detect models
    else:
        imgsz = (1024, 1024)  # Use default size for other models
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_files = []

    for image_path in test_images:
        logger.info(f"Processing comparison for {image_path.name}")

        # Run inference on all models
        model_results = {}

        for model_name, model_path in models.items():
            try:
                if model_type == "detect":
                    # For detect models, pass the image path directly
                    input_data, _, _ = load_and_preprocess_image(image_path, model_type)
                    detections = run_model_inference(
                        model_path,
                        input_data,
                        model_type,
                        confidence_threshold,
                        iou_threshold,
                    )
                else:
                    # For other models, use original preprocessing
                    input_data, _, _ = load_and_preprocess_image(image_path, model_type)
                    output = run_model_inference(model_path, input_data, model_type)
                    detections = parse_detections(output, confidence_threshold)

                model_results[model_name] = (model_path, detections)
                logger.debug(f"{model_name}: Found {len(detections)} detections")
            except Exception as e:
                logger.warning(f"Failed to run {model_name} on {image_path.name}: {e}")
                model_results[model_name] = (model_path, [])

        if model_results:
            # Create comparison figure
            output_filename = f"comparison_{image_path.stem}.png"
            output_path = output_dir / output_filename

            create_detection_comparison_figure(
                image_path,
                model_results,
                output_path,
                confidence_threshold,
                imgsz,
                model_type,
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
