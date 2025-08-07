"""
Validate quantized models against original models.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def load_and_preprocess_image(
    image_path: Path, target_size: tuple = (640, 640)
) -> np.ndarray:
    """Load and preprocess an image for model inference."""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target size
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and convert to CHW format
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise


def run_inference(model_path: Path, input_data: np.ndarray) -> np.ndarray:
    """Run inference on a model with the given input data."""
    try:
        # Create inference session
        providers = ["CPUExecutionProvider"]  # Use CPU for consistency
        session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: input_data})

        # Return the first output (assuming single output)
        return outputs[0] if outputs else np.array([])

    except Exception as e:
        logger.error(f"Inference failed for {model_path}: {e}")
        raise


def calculate_difference_metrics(
    original_output: np.ndarray, quantized_output: np.ndarray
) -> dict[str, float]:
    """Calculate various difference metrics between model outputs."""
    try:
        # Ensure arrays have the same shape
        if original_output.shape != quantized_output.shape:
            logger.warning(
                f"Output shape mismatch: {original_output.shape} vs {quantized_output.shape}"
            )
            # Try to reshape if possible
            if original_output.size == quantized_output.size:
                quantized_output = quantized_output.reshape(original_output.shape)
            else:
                raise ValueError("Output arrays have incompatible shapes and sizes")

        # Calculate various metrics
        absolute_diff = np.abs(original_output - quantized_output)

        metrics = {
            "max_absolute_diff": float(np.max(absolute_diff)),
            "mean_absolute_diff": float(np.mean(absolute_diff)),
            "rmse": float(np.sqrt(np.mean((original_output - quantized_output) ** 2))),
            "relative_error": float(
                np.max(absolute_diff / (np.abs(original_output) + 1e-8))
            ),
            "cosine_similarity": float(
                np.dot(original_output.flatten(), quantized_output.flatten())
                / (
                    np.linalg.norm(original_output.flatten())
                    * np.linalg.norm(quantized_output.flatten())
                    + 1e-8
                )
            ),
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating difference metrics: {e}")
        raise


def validate_single_image(
    original_model: Path, quantized_model: Path, image_path: Path
) -> dict[str, float]:
    """Validate models on a single image."""
    try:
        logger.debug(f"Validating on image: {image_path}")

        # Preprocess image
        input_data = load_and_preprocess_image(image_path)

        # Run inference on both models
        original_output = run_inference(original_model, input_data)
        quantized_output = run_inference(quantized_model, input_data)

        # Calculate difference metrics
        metrics = calculate_difference_metrics(original_output, quantized_output)

        logger.debug(f"Metrics for {image_path.name}: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Validation failed for image {image_path}: {e}")
        raise


def collect_test_images(test_images_dir: Path) -> list[Path]:
    """Collect test images from the specified directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    if test_images_dir.is_file():
        # Single image file
        if test_images_dir.suffix.lower() in image_extensions:
            image_paths.append(test_images_dir)
    else:
        # Directory of images
        for ext in image_extensions:
            image_paths.extend(test_images_dir.glob(f"*{ext}"))
            image_paths.extend(test_images_dir.glob(f"*{ext.upper()}"))

    if not image_paths:
        logger.warning(f"No test images found in {test_images_dir}")
    else:
        logger.info(f"Found {len(image_paths)} test images")

    return image_paths


def validate_models(
    original_model: Path,
    quantized_model: Path,
    test_images_dir: Path,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """
    Validate quantized model against original model.

    Returns:
        Tuple of (is_valid, max_difference)
    """
    try:
        logger.info(f"Validating {quantized_model.name} against {original_model.name}")

        # Collect test images
        test_images = collect_test_images(test_images_dir)
        if not test_images:
            # If no test images found, look for default images in parent directories
            parent_dir = test_images_dir
            for _ in range(3):  # Look up to 3 levels up
                parent_dir = parent_dir.parent
                if parent_dir.name == "beaker":
                    break
                default_images = list(parent_dir.glob("example*.jpg"))
                if default_images:
                    test_images = default_images
                    logger.info(
                        f"Using default test images: {[img.name for img in test_images]}"
                    )
                    break

        if not test_images:
            raise ValueError("No test images found for validation")

        # Validate on each test image
        all_metrics = []
        max_diff = 0.0

        for image_path in test_images:
            try:
                metrics = validate_single_image(
                    original_model, quantized_model, image_path
                )
                all_metrics.append(metrics)

                # Track maximum difference
                current_max = metrics["max_absolute_diff"]
                if current_max > max_diff:
                    max_diff = current_max

            except Exception as e:
                logger.warning(f"Skipping image {image_path.name} due to error: {e}")
                continue

        if not all_metrics:
            raise ValueError("No images could be validated")

        # Calculate aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f"avg_{key}"] = np.mean(values)
            avg_metrics[f"max_{key}"] = np.max(values)

        logger.info("Validation results:")
        logger.info(f"  Images tested: {len(all_metrics)}")
        logger.info(f"  Max absolute difference: {max_diff:.6f}")
        logger.info(
            f"  Avg absolute difference: {avg_metrics['avg_mean_absolute_diff']:.6f}"
        )
        logger.info(f"  RMSE: {avg_metrics['avg_rmse']:.6f}")
        logger.info(f"  Cosine similarity: {avg_metrics['avg_cosine_similarity']:.6f}")

        # Check if validation passes
        is_valid = max_diff <= tolerance

        return is_valid, max_diff

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def generate_validation_report(
    original_model: Path, quantized_model: Path, test_images_dir: Path
) -> dict[str, Any]:
    """Generate a detailed validation report."""
    try:
        # Run validation
        is_valid, max_diff = validate_models(
            original_model, quantized_model, test_images_dir
        )

        # Get model file sizes
        original_size = original_model.stat().st_size
        quantized_size = quantized_model.stat().st_size
        size_reduction = (original_size - quantized_size) / original_size * 100

        # Collect test images
        test_images = collect_test_images(test_images_dir)

        report = {
            "original_model": str(original_model),
            "quantized_model": str(quantized_model),
            "validation_passed": is_valid,
            "max_difference": max_diff,
            "model_sizes": {
                "original_bytes": original_size,
                "quantized_bytes": quantized_size,
                "size_reduction_percent": size_reduction,
            },
            "test_info": {
                "num_test_images": len(test_images),
                "test_images": [str(img) for img in test_images],
            },
        }

        return report

    except Exception as e:
        logger.error(f"Failed to generate validation report: {e}")
        raise
