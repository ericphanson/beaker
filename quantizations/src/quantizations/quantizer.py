"""
Quantize ONNX models using different techniques.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader

logger = logging.getLogger(__name__)


class ImageCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for image models."""

    def __init__(
        self, image_paths: list[Path], input_name: str, target_size: tuple = (640, 640)
    ):
        self.image_paths = image_paths
        self.input_name = input_name
        self.target_size = target_size
        self.data_index = 0
        logger.info(f"Created calibration reader with {len(image_paths)} images")

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get the next calibration data sample."""
        if self.data_index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.data_index]
        self.data_index += 1

        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return self.get_next()  # Try next image

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

            # Normalize to [0, 1] and convert to CHW format
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            return {self.input_name: image}

        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return self.get_next()  # Try next image


def get_model_input_name(model_path: Path) -> str:
    """Get the input tensor name from an ONNX model."""
    try:
        model = onnx.load(str(model_path))
        input_name = model.graph.input[0].name
        logger.debug(f"Model input name: {input_name}")
        return input_name
    except Exception as e:
        logger.error(f"Failed to get input name from {model_path}: {e}")
        # Default fallback
        return "images"


def collect_calibration_images(
    test_images_dir: Path, max_images: int = 10
) -> list[Path]:
    """Collect images for calibration."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(test_images_dir.glob(f"*{ext}"))
        image_paths.extend(test_images_dir.glob(f"*{ext.upper()}"))

    # Limit number of images
    image_paths = image_paths[:max_images]

    logger.info(f"Found {len(image_paths)} calibration images")
    return image_paths


def quantize_dynamic_model(model_path: Path, output_path: Path) -> bool:
    """Apply dynamic quantization to a model."""
    try:
        logger.info(f"Applying dynamic quantization to {model_path.name}")

        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )

        logger.info(f"Dynamic quantization complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Dynamic quantization failed: {e}")
        return False


def quantize_static_model(
    model_path: Path, output_path: Path, calibration_data_reader: CalibrationDataReader
) -> bool:
    """Apply static quantization to a model."""
    try:
        logger.info(f"Applying static quantization to {model_path.name}")

        # Create a temporary directory for augmented model
        with tempfile.TemporaryDirectory() as temp_dir:
            augmented_model_path = Path(temp_dir) / "augmented_model.onnx"

            quantize_static(
                model_input=str(model_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantType.QUInt8,
            )

        logger.info(f"Static quantization complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Static quantization failed: {e}")
        return False


def quantize_model(model_path: Path, output_dir: Path, quantization_level: str) -> Path:
    """Quantize a model at the specified level."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    stem = model_path.stem
    suffix = model_path.suffix
    output_name = f"{stem}-{quantization_level}{suffix}"
    output_path = output_dir / output_name

    if quantization_level == "dynamic":
        success = quantize_dynamic_model(model_path, output_path)

    elif quantization_level == "static":
        # Find calibration images
        test_images_dir = model_path.parent.parent.parent.parent.parent
        calibration_images = collect_calibration_images(test_images_dir)

        if not calibration_images:
            logger.warning(
                "No calibration images found, falling back to dynamic quantization"
            )
            output_name = f"{stem}-dynamic{suffix}"
            output_path = output_dir / output_name
            success = quantize_dynamic_model(model_path, output_path)
        else:
            input_name = get_model_input_name(model_path)
            calibration_reader = ImageCalibrationDataReader(
                calibration_images, input_name
            )
            success = quantize_static_model(model_path, output_path, calibration_reader)

    elif quantization_level == "int8":
        # For now, int8 quantization is the same as static quantization
        # In the future, this could use more aggressive quantization
        test_images_dir = model_path.parent.parent.parent.parent.parent
        calibration_images = collect_calibration_images(test_images_dir)

        if not calibration_images:
            logger.warning(
                "No calibration images found, falling back to dynamic quantization"
            )
            output_name = f"{stem}-dynamic{suffix}"
            output_path = output_dir / output_name
            success = quantize_dynamic_model(model_path, output_path)
        else:
            input_name = get_model_input_name(model_path)
            calibration_reader = ImageCalibrationDataReader(
                calibration_images, input_name
            )
            success = quantize_static_model(model_path, output_path, calibration_reader)

    else:
        raise ValueError(f"Unsupported quantization level: {quantization_level}")

    if not success:
        raise RuntimeError(f"Quantization failed for {model_path}")

    return output_path


def get_model_size(model_path: Path) -> int:
    """Get model file size in bytes."""
    return model_path.stat().st_size


def get_quantization_info(original_path: Path, quantized_path: Path) -> dict[str, Any]:
    """Get information about quantization results."""
    original_size = get_model_size(original_path)
    quantized_size = get_model_size(quantized_path)

    size_reduction = (original_size - quantized_size) / original_size * 100

    return {
        "original_size": original_size,
        "quantized_size": quantized_size,
        "size_reduction_percent": size_reduction,
        "compression_ratio": original_size / quantized_size
        if quantized_size > 0
        else 0,
    }
