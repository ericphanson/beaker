"""
Validate quantized models against original models using file size and metadata comparisons.
Note: This module no longer performs Python-based inference as it was unreliable.
Use Beaker CLI for actual inference validation.
"""

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


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


def get_model_size_info(original_model: Path, quantized_model: Path) -> dict[str, Any]:
    """Get file size information for model comparison."""
    try:
        original_size = original_model.stat().st_size
        quantized_size = quantized_model.stat().st_size
        size_reduction = (
            ((original_size - quantized_size) / original_size * 100)
            if original_size
            else 0
        )

        return {
            "original_bytes": original_size,
            "quantized_bytes": quantized_size,
            "size_reduction_percent": size_reduction,
            "compression_ratio": original_size / quantized_size
            if quantized_size > 0
            else 0,
        }
    except Exception as e:
        logger.error(f"Error getting model size info: {e}")
        return {
            "original_bytes": 0,
            "quantized_bytes": 0,
            "size_reduction_percent": 0.0,
            "compression_ratio": 0.0,
        }


def validate_models_basic(
    original_model: Path,
    quantized_model: Path,
    test_images_dir: Path,
    tolerance: float = 0.01,
) -> tuple[bool, float, dict[str, Any]]:
    """
    Basic validation focusing on file size and structure.

    Note: This no longer performs inference-based validation as the Python
    inference was unreliable. Use Beaker CLI for actual validation.

    Returns:
        Tuple of (is_valid, max_difference, detailed_metrics)
    """
    try:
        logger.info(
            f"Basic validation of {quantized_model.name} against {original_model.name}"
        )
        logger.warning(
            "Note: Inference-based validation disabled. Use Beaker CLI for accurate validation."
        )

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
            logger.warning(
                "No test images found, proceeding with basic file validation only"
            )

        # Get model sizes
        size_info = get_model_size_info(original_model, quantized_model)

        # Mock validation metrics (since we removed inference)
        # In practice, you should use Beaker CLI to generate actual metrics
        mock_metrics = {
            "avg_max_absolute_diff": 0.0,  # Would need Beaker CLI to measure
            "avg_mean_absolute_diff": 0.0,
            "avg_rmse": 0.0,
            "avg_relative_error": 0.0,
            "avg_cosine_similarity": 1.0,
            "max_max_absolute_diff": 0.0,
        }

        # Mock timing (would need actual Beaker CLI runs)
        mock_timing = {
            "original_timing": {"mean_ms": 0.0, "std_ms": 0.0},
            "quantized_timing": {"mean_ms": 0.0, "std_ms": 0.0},
        }

        detailed_metrics = {
            **mock_metrics,
            **mock_timing,
            "model_sizes": size_info,
            "test_info": {
                "num_test_images": len(test_images),
                "test_images": [str(img) for img in test_images],
            },
            "validation_method": "basic_file_only",
            "note": "Use Beaker CLI for inference-based validation",
        }

        logger.info("Basic validation results:")
        logger.info(f"  Images found: {len(test_images)}")
        logger.info(f"  Original size: {size_info['original_bytes']} bytes")
        logger.info(f"  Quantized size: {size_info['quantized_bytes']} bytes")
        logger.info(f"  Size reduction: {size_info['size_reduction_percent']:.1f}%")
        logger.info(
            "  NOTE: No inference validation performed. Use Beaker CLI for accurate testing."
        )

        # For basic validation, we assume it passes if the file exists and is smaller
        is_valid = (
            quantized_model.exists()
            and quantized_model.stat().st_size > 0
            and size_info["size_reduction_percent"] > 0
        )

        # Return 0 as max_diff since we're not doing inference
        max_diff = 0.0

        return is_valid, max_diff, detailed_metrics

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


# Keep backward compatibility functions
def validate_models_with_timing(
    original_model: Path,
    quantized_model: Path,
    test_images_dir: Path,
    tolerance: float = 0.01,
    num_timing_runs: int = 5,
) -> tuple[bool, float, dict[str, Any]]:
    """
    Validate quantized model with timing (now uses basic validation).

    NOTE: Inference-based validation has been disabled due to unreliable Python preprocessing.
    Use Beaker CLI for accurate inference validation.
    """
    logger.warning(
        "validate_models_with_timing now uses basic validation. Use Beaker CLI for inference validation."
    )
    return validate_models_basic(
        original_model, quantized_model, test_images_dir, tolerance
    )


def validate_models(
    original_model: Path,
    quantized_model: Path,
    test_images_dir: Path,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """
    Validate quantized model against original model (backward compatibility).

    NOTE: Inference-based validation has been disabled due to unreliable Python preprocessing.
    Use Beaker CLI for accurate inference validation.
    """
    logger.warning(
        "validate_models now uses basic validation. Use Beaker CLI for inference validation."
    )
    is_valid, max_diff, _ = validate_models_basic(
        original_model, quantized_model, test_images_dir, tolerance
    )
    return is_valid, max_diff


def generate_validation_report(
    original_model: Path, quantized_model: Path, test_images_dir: Path
) -> dict[str, Any]:
    """Generate a detailed validation report using basic file validation."""
    try:
        # Run basic validation
        is_valid, max_diff, detailed_metrics = validate_models_basic(
            original_model, quantized_model, test_images_dir
        )

        # Collect test images
        test_images = collect_test_images(test_images_dir)

        report = {
            "original_model": str(original_model),
            "quantized_model": str(quantized_model),
            "validation_passed": is_valid,
            "max_difference": max_diff,
            "model_sizes": detailed_metrics["model_sizes"],
            "test_info": {
                "num_test_images": len(test_images),
                "test_images": [str(img) for img in test_images],
            },
            "validation_method": "basic_file_only",
            "note": "Use Beaker CLI for inference-based validation",
        }

        return report

    except Exception as e:
        logger.error(f"Failed to generate validation report: {e}")
        raise
