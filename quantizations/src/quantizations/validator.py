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
