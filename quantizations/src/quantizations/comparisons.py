"""
Generate comparison images and figures for quantized models using Beaker CLI outputs.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

logger = logging.getLogger(__name__)


def run_beaker_inference(
    model_path: Path,
    model_type: str,
    test_images: List[Path],
    output_dir: Path,
    beaker_cargo_path: Path | None = None,
) -> bool:
    """Run Beaker CLI inference on test images with the specified model."""
    try:
        # Default to relative path if not specified
        if beaker_cargo_path is None:
            beaker_cargo_path = Path("../beaker/Cargo.toml")

        # Set environment variable for the model
        env = os.environ.copy()
        if model_type == "head":
            env["BEAKER_HEAD_MODEL_PATH"] = str(model_path)
            command_type = "head"
            extra_args = ["--crop", "--bounding-box", "--metadata"]
        elif model_type == "cutout":
            env["BEAKER_CUTOUT_MODEL_PATH"] = str(model_path)
            command_type = "cutout"
            extra_args = ["--metadata"]
        else:
            logger.error(f"Unknown model type: {model_type}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each test image
        for test_image in test_images:
            try:
                logger.info(
                    f"Running Beaker {command_type} inference on {test_image.name}"
                )

                # Build command
                cmd = [
                    "cargo",
                    "run",
                    "--manifest-path",
                    str(beaker_cargo_path),
                    command_type,
                    str(test_image),
                    "--output-dir",
                    str(output_dir),
                ] + extra_args

                # Run the command
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 second timeout
                )

                if result.returncode != 0:
                    logger.warning(
                        f"Beaker CLI failed for {test_image.name}: {result.stderr}"
                    )
                    continue

                logger.debug(f"Successfully processed {test_image.name}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Beaker CLI timed out for {test_image.name}")
                continue
            except Exception as e:
                logger.warning(f"Error processing {test_image.name}: {e}")
                continue

        return True

    except Exception as e:
        logger.error(f"Failed to run Beaker inference: {e}")
        return False


def collect_beaker_outputs(
    temp_dir: Path, model_name: str, final_output_dir: Path
) -> List[Path]:
    """Collect and rename Beaker outputs with model name prefix."""
    collected_files = []

    try:
        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Find all output files in temp directory
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                # Create new filename with model prefix
                new_name = f"{model_name}_{file_path.name}"
                final_path = final_output_dir / new_name

                # Copy file to final location
                shutil.copy2(file_path, final_path)
                collected_files.append(final_path)
                logger.debug(f"Collected {file_path.name} as {new_name}")

        return collected_files

    except Exception as e:
        logger.error(f"Error collecting outputs for {model_name}: {e}")
        return []


def load_image_safely(image_path: Path) -> np.ndarray | None:
    """Load an image file safely, returning None if it fails."""
    try:
        if image_path.exists() and image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ]:
            return mpimg.imread(str(image_path))
        return None
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def create_side_by_side_comparison(
    image_groups: Dict[str, List[Path]], output_path: Path, test_image_name: str
) -> bool:
    """Create a side-by-side comparison of images from different models."""
    try:
        # Filter out groups with no images
        valid_groups = {
            name: paths
            for name, paths in image_groups.items()
            if paths and any(load_image_safely(p) is not None for p in paths)
        }

        if len(valid_groups) < 2:
            logger.warning(
                f"Not enough valid image groups for comparison of {test_image_name}"
            )
            return False

        # Determine the layout - try to make it roughly square
        num_groups = len(valid_groups)
        cols = int(np.ceil(np.sqrt(num_groups)))
        rows = int(np.ceil(num_groups / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        # Handle different axes configurations
        if num_groups == 1:
            axes_flat = [axes]
        elif isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for idx, (model_name, image_paths) in enumerate(valid_groups.items()):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # Find the first valid image in this group
            image_data = None
            used_path = None
            for img_path in image_paths:
                image_data = load_image_safely(img_path)
                if image_data is not None:
                    used_path = img_path
                    break

            if image_data is not None:
                ax.imshow(image_data)

                # Create title from model name and filename
                title = f"{model_name}"
                if used_path:
                    # Add the specific filename if it's different from the model name
                    filename = used_path.stem.replace(
                        f"{model_name.lower().replace(' ', '-')}_", ""
                    )
                    if filename != test_image_name:
                        title += f"\n({filename})"

                ax.set_title(title, fontsize=10, fontweight="bold")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{model_name}\n(No image)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(model_name, fontsize=10, fontweight="bold")

            ax.axis("off")

        # Hide unused subplots
        for idx in range(num_groups, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.suptitle(
            f"Model Comparison: {test_image_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Created comparison figure: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create side-by-side comparison: {e}")
        return False


def parse_beaker_metadata(metadata_file: Path) -> Dict:
    """Parse Beaker metadata file (.beaker.toml) for comparison metrics."""
    try:
        import tomllib  # Python 3.11+

        with open(metadata_file, "rb") as f:
            metadata = tomllib.load(f)
        return metadata
    except ImportError:
        # For older Python versions, try to read as text and warn
        logger.warning(
            f"TOML parsing not available for Python < 3.11, skipping metadata: {metadata_file}"
        )
        return {}
    except Exception as e:
        logger.error(f"Error parsing metadata file {metadata_file}: {e}")
        return {}


def compare_head_detections(
    original_metadata: Dict, quantized_metadata: Dict
) -> Dict[str, float]:
    """Compare head detection results between original and quantized models."""
    try:
        # Extract bounding boxes from metadata
        original_detections = original_metadata.get("head", {}).get("detections", [])
        quantized_detections = quantized_metadata.get("head", {}).get("detections", [])

        if not original_detections and not quantized_detections:
            # Both models found no detections - perfect match
            return {
                "detection_count_diff": 0.0,
                "avg_confidence_diff": 0.0,
                "avg_bbox_iou": 1.0,
                "avg_bbox_center_distance": 0.0,
                "max_bbox_center_distance": 0.0,
            }

        if not original_detections or not quantized_detections:
            # One model found detections, the other didn't
            return {
                "detection_count_diff": abs(
                    len(original_detections) - len(quantized_detections)
                ),
                "avg_confidence_diff": 1.0,  # Maximum difference
                "avg_bbox_iou": 0.0,  # No overlap
                "avg_bbox_center_distance": float("inf"),
                "max_bbox_center_distance": float("inf"),
            }

        # Compare detection counts
        count_diff = abs(len(original_detections) - len(quantized_detections))

        # Compare detections by matching closest bounding boxes
        bbox_ious = []
        confidence_diffs = []
        center_distances = []

        for orig_det in original_detections:
            best_iou = 0.0
            best_conf_diff = 1.0
            best_center_dist = float("inf")

            orig_bbox = [
                orig_det.get("x1", 0),
                orig_det.get("y1", 0),
                orig_det.get("x2", 0),
                orig_det.get("y2", 0),
            ]
            orig_center = [
                (orig_bbox[0] + orig_bbox[2]) / 2,
                (orig_bbox[1] + orig_bbox[3]) / 2,
            ]

            for quant_det in quantized_detections:
                quant_bbox = [
                    quant_det.get("x1", 0),
                    quant_det.get("y1", 0),
                    quant_det.get("x2", 0),
                    quant_det.get("y2", 0),
                ]
                quant_center = [
                    (quant_bbox[0] + quant_bbox[2]) / 2,
                    (quant_bbox[1] + quant_bbox[3]) / 2,
                ]

                # Calculate IoU
                iou = calculate_bbox_iou(orig_bbox, quant_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_conf_diff = abs(
                        orig_det.get("confidence", 0) - quant_det.get("confidence", 0)
                    )
                    # Calculate center distance
                    best_center_dist = np.sqrt(
                        (orig_center[0] - quant_center[0]) ** 2
                        + (orig_center[1] - quant_center[1]) ** 2
                    )

            bbox_ious.append(best_iou)
            confidence_diffs.append(best_conf_diff)
            center_distances.append(best_center_dist)

        return {
            "detection_count_diff": float(count_diff),
            "avg_confidence_diff": float(np.mean(confidence_diffs))
            if confidence_diffs
            else 0.0,
            "avg_bbox_iou": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
            "avg_bbox_center_distance": float(np.mean(center_distances))
            if center_distances
            else 0.0,
            "max_bbox_center_distance": float(np.max(center_distances))
            if center_distances
            else 0.0,
        }

    except Exception as e:
        logger.error(f"Error comparing head detections: {e}")
        return {
            "detection_count_diff": 0.0,
            "avg_confidence_diff": 0.0,
            "avg_bbox_iou": 0.0,
            "avg_bbox_center_distance": 0.0,
            "max_bbox_center_distance": 0.0,
        }


def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    try:
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    except Exception as e:
        logger.error(f"Error calculating bbox IoU: {e}")
        return 0.0


def compare_cutout_masks(
    original_mask_path: Path, quantized_mask_path: Path
) -> Dict[str, float]:
    """Compare cutout mask images between original and quantized models."""
    try:
        from PIL import Image
        import numpy as np

        # Load mask images
        orig_mask = np.array(Image.open(original_mask_path).convert("L"))
        quant_mask = np.array(Image.open(quantized_mask_path).convert("L"))

        if orig_mask.shape != quant_mask.shape:
            logger.warning(
                f"Mask shape mismatch: {orig_mask.shape} vs {quant_mask.shape}"
            )
            # Resize to match
            from PIL import Image

            quant_pil = Image.fromarray(quant_mask).resize(
                (orig_mask.shape[1], orig_mask.shape[0])
            )
            quant_mask = np.array(quant_pil)

        # Convert to binary masks (assuming non-zero pixels are foreground)
        orig_binary = (orig_mask > 0).astype(np.uint8)
        quant_binary = (quant_mask > 0).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(orig_binary, quant_binary).sum()
        union = np.logical_or(orig_binary, quant_binary).sum()

        # Calculate IoU
        iou = intersection / union if union > 0 else 1.0

        # Calculate pixel-wise differences
        pixel_diff = np.abs(
            orig_mask.astype(np.float32) - quant_mask.astype(np.float32)
        )
        mean_pixel_diff = np.mean(pixel_diff)
        max_pixel_diff = np.max(pixel_diff)

        # Calculate Dice coefficient (another common mask similarity metric)
        dice = (
            (2 * intersection) / (orig_binary.sum() + quant_binary.sum())
            if (orig_binary.sum() + quant_binary.sum()) > 0
            else 1.0
        )

        return {
            "mask_iou": float(iou),
            "mask_dice": float(dice),
            "mean_pixel_diff": float(mean_pixel_diff),
            "max_pixel_diff": float(max_pixel_diff),
            "intersection_pixels": int(intersection),
            "union_pixels": int(union),
        }

    except Exception as e:
        logger.error(f"Error comparing cutout masks: {e}")
        return {
            "mask_iou": 0.0,
            "mask_dice": 0.0,
            "mean_pixel_diff": 255.0,  # Maximum possible difference
            "max_pixel_diff": 255.0,
            "intersection_pixels": 0,
            "union_pixels": 0,
        }


def analyze_beaker_outputs(
    output_dir: Path, test_images: List[Path], models: Dict[str, Path], model_type: str
) -> Dict[str, Dict]:
    """
    Analyze Beaker CLI outputs programmatically to compare model performance.

    Args:
        output_dir: Directory containing Beaker outputs
        test_images: List of test image paths
        models: Dict of {model_name: model_path}
        model_type: 'head' or 'cutout'

    Returns:
        Dict with detailed comparison metrics
    """
    try:
        comparison_results = {}

        for test_image in test_images:
            image_name = test_image.stem
            image_results = {}

            # Get paths for original and quantized outputs
            original_outputs = {}
            quantized_outputs = {}

            for model_name in models.keys():
                if model_name == "original":
                    # Find original outputs
                    toml_pattern = f"{model_name}_{image_name}.beaker.toml"
                    toml_files = list(output_dir.glob(f"**/{toml_pattern}"))
                    if toml_files:
                        original_outputs["metadata"] = toml_files[0]

                    if model_type == "cutout":
                        # Find mask files
                        mask_pattern = f"{model_name}_{image_name}_cutout.png"
                        mask_files = list(output_dir.glob(f"**/{mask_pattern}"))
                        if mask_files:
                            original_outputs["mask"] = mask_files[0]
                else:
                    # Find quantized outputs
                    toml_pattern = f"{model_name}_{image_name}.beaker.toml"
                    toml_files = list(output_dir.glob(f"**/{toml_pattern}"))
                    if toml_files:
                        quantized_outputs[model_name] = {"metadata": toml_files[0]}

                    if model_type == "cutout":
                        mask_pattern = f"{model_name}_{image_name}_cutout.png"
                        mask_files = list(output_dir.glob(f"**/{mask_pattern}"))
                        if mask_files:
                            if model_name not in quantized_outputs:
                                quantized_outputs[model_name] = {}
                            quantized_outputs[model_name]["mask"] = mask_files[0]

            # Compare each quantized model against original
            for quant_model_name, quant_outputs in quantized_outputs.items():
                if (
                    "metadata" not in original_outputs
                    or "metadata" not in quant_outputs
                ):
                    logger.warning(
                        f"Missing metadata files for {image_name} - {quant_model_name}"
                    )
                    continue

                # Parse metadata
                orig_metadata = parse_beaker_metadata(original_outputs["metadata"])
                quant_metadata = parse_beaker_metadata(quant_outputs["metadata"])

                if not orig_metadata or not quant_metadata:
                    logger.warning(
                        f"Failed to parse metadata for {image_name} - {quant_model_name}"
                    )
                    continue

                # Model-specific comparisons
                if model_type == "head":
                    # Compare bounding box detections
                    detection_metrics = compare_head_detections(
                        orig_metadata, quant_metadata
                    )

                    # Extract timing and model info
                    orig_timing = (
                        orig_metadata.get("head", {})
                        .get("execution", {})
                        .get("model_processing_time_ms", 0)
                    )
                    quant_timing = (
                        quant_metadata.get("head", {})
                        .get("execution", {})
                        .get("model_processing_time_ms", 0)
                    )

                    orig_size = (
                        orig_metadata.get("head", {})
                        .get("system", {})
                        .get("model_size_bytes", 0)
                    )
                    quant_size = (
                        quant_metadata.get("head", {})
                        .get("system", {})
                        .get("model_size_bytes", 0)
                    )

                    image_results[quant_model_name] = {
                        **detection_metrics,
                        "original_timing_ms": orig_timing,
                        "quantized_timing_ms": quant_timing,
                        "timing_speedup": orig_timing / quant_timing
                        if quant_timing > 0
                        else 0,
                        "original_size_bytes": orig_size,
                        "quantized_size_bytes": quant_size,
                        "size_reduction_percent": (
                            (orig_size - quant_size) / orig_size * 100
                        )
                        if orig_size > 0
                        else 0,
                    }

                elif model_type == "cutout":
                    # Compare cutout masks if available
                    mask_metrics = {}
                    if "mask" in original_outputs and "mask" in quant_outputs:
                        mask_metrics = compare_cutout_masks(
                            original_outputs["mask"], quant_outputs["mask"]
                        )

                    # Extract timing and model info
                    orig_timing = (
                        orig_metadata.get("cutout", {})
                        .get("execution", {})
                        .get("model_processing_time_ms", 0)
                    )
                    quant_timing = (
                        quant_metadata.get("cutout", {})
                        .get("execution", {})
                        .get("model_processing_time_ms", 0)
                    )

                    orig_size = (
                        orig_metadata.get("cutout", {})
                        .get("system", {})
                        .get("model_size_bytes", 0)
                    )
                    quant_size = (
                        quant_metadata.get("cutout", {})
                        .get("system", {})
                        .get("model_size_bytes", 0)
                    )

                    image_results[quant_model_name] = {
                        **mask_metrics,
                        "original_timing_ms": orig_timing,
                        "quantized_timing_ms": quant_timing,
                        "timing_speedup": orig_timing / quant_timing
                        if quant_timing > 0
                        else 0,
                        "original_size_bytes": orig_size,
                        "quantized_size_bytes": quant_size,
                        "size_reduction_percent": (
                            (orig_size - quant_size) / orig_size * 100
                        )
                        if orig_size > 0
                        else 0,
                    }

            if image_results:
                comparison_results[image_name] = image_results

        return comparison_results

    except Exception as e:
        logger.error(f"Error analyzing Beaker outputs: {e}")
        return {}


def generate_analysis_report(analysis_results: Dict[str, Dict], model_type: str) -> str:
    """Generate a detailed report from Beaker output analysis."""
    try:
        if not analysis_results:
            return "No analysis results available.\n"

        report = [f"\n## Beaker Output Analysis Report - {model_type.upper()} Model\n"]

        # Aggregate metrics across all test images
        all_metrics = {}
        model_names = set()

        for image_name, image_results in analysis_results.items():
            for model_name, metrics in image_results.items():
                model_names.add(model_name)
                if model_name not in all_metrics:
                    all_metrics[model_name] = []
                all_metrics[model_name].append(metrics)

        # Summary table
        report.append("### Summary")
        if model_type == "head":
            report.append(
                "| Model | Avg IoU | Avg Confidence Diff | Detection Count Match | Timing Speedup | Size Reduction |"
            )
            report.append(
                "|-------|---------|---------------------|----------------------|----------------|----------------|"
            )

            for model_name in sorted(model_names):
                metrics_list = all_metrics[model_name]
                avg_iou = np.mean([m.get("avg_bbox_iou", 0) for m in metrics_list])
                avg_conf_diff = np.mean(
                    [m.get("avg_confidence_diff", 0) for m in metrics_list]
                )
                avg_detection_match = np.mean(
                    [
                        1.0 if m.get("detection_count_diff", 1) == 0 else 0.0
                        for m in metrics_list
                    ]
                )
                avg_speedup = np.mean(
                    [m.get("timing_speedup", 0) for m in metrics_list]
                )
                avg_size_reduction = np.mean(
                    [m.get("size_reduction_percent", 0) for m in metrics_list]
                )

                report.append(
                    f"| {model_name} | {avg_iou:.3f} | {avg_conf_diff:.3f} | {avg_detection_match:.1%} | "
                    f"{avg_speedup:.2f}x | {avg_size_reduction:.1f}% |"
                )

        elif model_type == "cutout":
            report.append(
                "| Model | Mask IoU | Mask Dice | Pixel Diff (avg) | Timing Speedup | Size Reduction |"
            )
            report.append(
                "|-------|----------|-----------|------------------|----------------|----------------|"
            )

            for model_name in sorted(model_names):
                metrics_list = all_metrics[model_name]
                avg_iou = np.mean([m.get("mask_iou", 0) for m in metrics_list])
                avg_dice = np.mean([m.get("mask_dice", 0) for m in metrics_list])
                avg_pixel_diff = np.mean(
                    [m.get("mean_pixel_diff", 0) for m in metrics_list]
                )
                avg_speedup = np.mean(
                    [m.get("timing_speedup", 0) for m in metrics_list]
                )
                avg_size_reduction = np.mean(
                    [m.get("size_reduction_percent", 0) for m in metrics_list]
                )

                report.append(
                    f"| {model_name} | {avg_iou:.3f} | {avg_dice:.3f} | {avg_pixel_diff:.1f} | "
                    f"{avg_speedup:.2f}x | {avg_size_reduction:.1f}% |"
                )

        # Detailed per-image results
        report.append("\n### Detailed Results by Test Image")

        for image_name, image_results in analysis_results.items():
            report.append(f"\n#### {image_name}")

            for model_name, metrics in image_results.items():
                report.append(f"\n**{model_name}:**")

                if model_type == "head":
                    report.append(
                        f"- Detection Count Difference: {metrics.get('detection_count_diff', 0)}"
                    )
                    report.append(
                        f"- Average Bbox IoU: {metrics.get('avg_bbox_iou', 0):.3f}"
                    )
                    report.append(
                        f"- Average Confidence Difference: {metrics.get('avg_confidence_diff', 0):.3f}"
                    )
                    report.append(
                        f"- Average Center Distance: {metrics.get('avg_bbox_center_distance', 0):.1f} pixels"
                    )

                elif model_type == "cutout":
                    report.append(f"- Mask IoU: {metrics.get('mask_iou', 0):.3f}")
                    report.append(
                        f"- Mask Dice Coefficient: {metrics.get('mask_dice', 0):.3f}"
                    )
                    report.append(
                        f"- Mean Pixel Difference: {metrics.get('mean_pixel_diff', 0):.1f}"
                    )
                    report.append(
                        f"- Max Pixel Difference: {metrics.get('max_pixel_diff', 0):.1f}"
                    )

                # Common metrics
                report.append(
                    f"- Processing Time: {metrics.get('quantized_timing_ms', 0):.1f}ms vs {metrics.get('original_timing_ms', 0):.1f}ms (original)"
                )
                report.append(
                    f"- Timing Speedup: {metrics.get('timing_speedup', 0):.2f}x"
                )
                report.append(
                    f"- Model Size: {metrics.get('quantized_size_bytes', 0) / 1024 / 1024:.1f}MB vs {metrics.get('original_size_bytes', 0) / 1024 / 1024:.1f}MB (original)"
                )
                report.append(
                    f"- Size Reduction: {metrics.get('size_reduction_percent', 0):.1f}%"
                )

        return "\n".join(report)

    except Exception as e:
        logger.error(f"Error generating analysis report: {e}")
        return f"Error generating report: {e}\n"


def organize_outputs_by_test_image(
    model_results: Dict[str, List[Path]], test_images: List[Path]
) -> Dict[str, Dict[str, List[Path]]]:
    """Organize model outputs by test image for easier comparison."""
    organized = {}

    for test_image in test_images:
        test_name = test_image.stem
        organized[test_name] = {}

        for model_name, output_files in model_results.items():
            # Find files that belong to this test image
            matching_files = []
            for output_file in output_files:
                if test_name in output_file.name:
                    matching_files.append(output_file)

            if matching_files:
                organized[test_name][model_name] = matching_files

    return organized


def generate_model_comparison_images(
    models: Dict[str, Path],
    test_images: List[Path],
    output_dir: Path,
    confidence_threshold: float = 0.5,  # Kept for compatibility but not used
    beaker_cargo_path: Path | None = None,
) -> List[Path]:
    """Generate comparison results using Beaker CLI for multiple models on test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_files = []

    # Determine model type from the output directory or model names
    model_type = "head"  # Default
    if output_dir.name in ["head", "cutout"]:
        model_type = output_dir.name
    else:
        # Try to infer from model paths
        for model_name, model_path in models.items():
            if "cutout" in str(model_path).lower():
                model_type = "cutout"
                break

    logger.info(f"Running {model_type} model comparisons using Beaker CLI")

    # Process each model
    model_results = {}

    for model_name, model_path in models.items():
        logger.info(f"Processing {model_name} model: {model_path.name}")

        # Create temporary directory for this model's outputs
        with tempfile.TemporaryDirectory(
            prefix=f"beaker_{model_name}_"
        ) as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            # Run Beaker inference
            success = run_beaker_inference(
                model_path, model_type, test_images, temp_dir, beaker_cargo_path
            )

            if success:
                # Collect outputs with model name prefix
                model_outputs = collect_beaker_outputs(
                    temp_dir, model_name.lower().replace(" ", "-"), output_dir
                )
                model_results[model_name] = model_outputs

                logger.info(
                    f"Collected {len(model_outputs)} output files for {model_name}"
                )
            else:
                logger.warning(f"Failed to process {model_name}")
                model_results[model_name] = []

    # Organize outputs by test image for side-by-side comparisons
    organized_outputs = organize_outputs_by_test_image(model_results, test_images)

    # Generate side-by-side comparison images
    for test_name, model_outputs in organized_outputs.items():
        if len(model_outputs) > 1:  # Need at least 2 models to compare
            # Group image files by model
            image_groups = {}
            for model_name, files in model_outputs.items():
                # Filter for image files only
                image_files = [
                    f
                    for f in files
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                ]
                if image_files:
                    image_groups[model_name] = image_files

            if len(image_groups) > 1:
                comparison_filename = f"{model_type}_comparison_{test_name}.png"
                comparison_path = output_dir / comparison_filename

                success = create_side_by_side_comparison(
                    image_groups, comparison_path, test_name
                )

                if success:
                    comparison_files.append(comparison_path)

    # Run programmatic Beaker output analysis
    logger.info("Running programmatic analysis of Beaker outputs...")
    try:
        analysis_results = analyze_beaker_outputs(
            output_dir, test_images, models, model_type
        )
        if analysis_results:
            # Generate analysis report
            analysis_report = generate_analysis_report(analysis_results, model_type)

            # Save analysis report
            report_path = output_dir / f"{model_type}_beaker_analysis_report.md"
            with open(report_path, "w") as f:
                f.write(analysis_report)

            comparison_files.append(report_path)
            logger.info(f"Saved Beaker analysis report: {report_path}")

            # Also save raw analysis data as JSON for further processing
            import json

            json_path = output_dir / f"{model_type}_beaker_analysis_data.json"
            with open(json_path, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)

            comparison_files.append(json_path)
            logger.info(f"Saved Beaker analysis data: {json_path}")
        else:
            logger.warning("No Beaker output analysis results generated")
    except Exception as e:
        logger.error(f"Failed to run Beaker output analysis: {e}")

    # Generate comparison summary if we have head detection models
    if model_type == "head" and len(model_results) > 1:
        try:
            comparison_summary = generate_head_detection_summary(
                model_results, test_images, output_dir
            )
            if comparison_summary:
                comparison_files.append(comparison_summary)
        except Exception as e:
            logger.warning(f"Failed to generate comparison summary: {e}")

    logger.info(f"Generated {len(comparison_files)} comparison files")
    return comparison_files


def generate_head_detection_summary(
    model_results: Dict[str, List[Path]], test_images: List[Path], output_dir: Path
) -> Path | None:
    """Generate a summary comparison for head detection models."""
    try:
        summary_data = {
            "comparison_type": "head_detection",
            "models": list(model_results.keys()),
            "test_images": [img.name for img in test_images],
            "comparisons": [],
        }

        # Find baseline model (typically "Original")
        baseline_model = None
        for model_name in model_results.keys():
            if "original" in model_name.lower():
                baseline_model = model_name
                break

        if not baseline_model:
            baseline_model = list(model_results.keys())[0]

        # Compare each model against baseline
        for model_name, output_files in model_results.items():
            if model_name == baseline_model:
                continue

            model_comparison = {
                "model_name": model_name,
                "baseline": baseline_model,
                "image_comparisons": [],
            }

            # Find metadata files for comparison
            for test_image in test_images:
                baseline_metadata_file = None
                model_metadata_file = None

                # Look for corresponding metadata files
                for file_path in model_results[baseline_model]:
                    if (
                        test_image.stem in file_path.name
                        and file_path.suffix == ".toml"
                    ):
                        baseline_metadata_file = file_path
                        break

                for file_path in output_files:
                    if (
                        test_image.stem in file_path.name
                        and file_path.suffix == ".toml"
                    ):
                        model_metadata_file = file_path
                        break

                if baseline_metadata_file and model_metadata_file:
                    baseline_metadata = parse_beaker_metadata(baseline_metadata_file)
                    model_metadata = parse_beaker_metadata(model_metadata_file)

                    comparison_metrics = compare_head_detections(
                        baseline_metadata, model_metadata
                    )

                    model_comparison["image_comparisons"].append(
                        {"image": test_image.name, "metrics": comparison_metrics}
                    )

            summary_data["comparisons"].append(model_comparison)

        # Save summary
        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Generated comparison summary: {summary_file}")
        return summary_file

    except Exception as e:
        logger.error(f"Error generating head detection summary: {e}")
        return None


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
