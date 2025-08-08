"""
Quantize ONNX models using different techniques.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxsim
from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader

logger = logging.getLogger(__name__)


def get_model_input_name(model_path: Path) -> str:
    """Get the input name of an ONNX model."""
    try:
        model = onnx.load(str(model_path))
        return model.graph.input[0].name
    except Exception as e:
        logger.warning(f"Could not get input name for {model_path}: {e}")
        return "images"  # Default fallback


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


def optimize_model(model_path: Path, output_path: Path) -> bool:
    """Optimize and simplify an ONNX model using onnx-simplifier."""
    try:
        logger.info(f"Optimizing model {model_path.name}")

        # Load the model
        model = onnx.load(str(model_path))

        # Simplify the model
        model_simplified, check = onnxsim.simplify(model)

        if check:
            # Save the optimized model
            onnx.save(model_simplified, str(output_path))
            logger.info(f"Model optimized and saved: {output_path}")
            return True
        else:
            logger.warning(f"Model simplification check failed for {model_path.name}")
            # Copy original if simplification fails
            import shutil

            shutil.copy2(model_path, output_path)
            return False

    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        # Copy original if optimization fails
        import shutil

        shutil.copy2(model_path, output_path)
        return False


def quantize_fp16_model(model_path: Path, output_path: Path) -> bool:
    """Apply FP16 quantization to a model (currently disabled due to complexity)."""
    try:
        logger.warning(
            "FP16 quantization is currently disabled due to ONNX type compatibility issues"
        )
        # For now, just copy the original model as a placeholder
        import shutil

        shutil.copy2(model_path, output_path)
        logger.info(f"FP16 quantization skipped: {output_path}")
        return False  # Return False to indicate this quantization was skipped

    except Exception as e:
        logger.error(f"FP16 quantization failed: {e}")
        return False


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

        quantize_static(
            model_input=str(model_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_data_reader,
        )

        logger.info(f"Static quantization complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Static quantization failed: {e}")
        return False


def quantize_model(model_path: Path, output_dir: Path, quantization_level: str) -> Path:
    """Quantize a model at the specified level."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with better naming
    stem = model_path.stem
    suffix = model_path.suffix

    # Determine model type from path or filename
    model_type = "unknown"
    if "head" in str(model_path).lower() or "best" in stem.lower():
        model_type = "head"
    elif "cutout" in str(model_path).lower():
        model_type = "cutout"

    # Create descriptive filename
    if quantization_level == "fp16":
        output_name = f"{model_type}-fp16{suffix}"
    elif quantization_level == "dynamic":
        output_name = f"{model_type}-dynamic-int8{suffix}"
    elif quantization_level == "static":
        output_name = f"{model_type}-static-int8{suffix}"
    elif quantization_level == "int8":
        output_name = f"{model_type}-static-int8{suffix}"
    else:
        output_name = f"{model_type}-{quantization_level}{suffix}"

    output_path = output_dir / output_name

    # First optimize the model if not already optimized
    if "optimized" not in stem:
        optimized_path = output_dir / f"{stem}-optimized{suffix}"
        optimize_model(model_path, optimized_path)
        working_model = optimized_path
    else:
        working_model = model_path

    if quantization_level == "fp16":
        success = quantize_fp16_model(working_model, output_path)

    elif quantization_level == "dynamic":
        success = quantize_dynamic_model(working_model, output_path)

    elif quantization_level == "static":
        # Find calibration images
        test_images_dir = model_path.parent.parent.parent.parent.parent
        calibration_images = collect_calibration_images(test_images_dir)

        if not calibration_images:
            logger.warning(
                "No calibration images found, falling back to dynamic quantization"
            )
            output_name = f"{model_type}-dynamic-int8{suffix}"
            output_path = output_dir / output_name
            success = quantize_dynamic_model(working_model, output_path)
        else:
            input_name = get_model_input_name(working_model)
            calibration_reader = ImageCalibrationDataReader(
                calibration_images, input_name
            )
            success = quantize_static_model(
                working_model, output_path, calibration_reader
            )

    elif quantization_level == "int8":
        # For now, int8 quantization is the same as static quantization
        # In the future, this could use more aggressive quantization
        test_images_dir = model_path.parent.parent.parent.parent.parent
        calibration_images = collect_calibration_images(test_images_dir)

        if not calibration_images:
            logger.warning(
                "No calibration images found, falling back to dynamic quantization"
            )
            output_name = f"{model_type}-dynamic-int8{suffix}"
            output_path = output_dir / output_name
            success = quantize_dynamic_model(working_model, output_path)
        else:
            input_name = get_model_input_name(working_model)
            calibration_reader = ImageCalibrationDataReader(
                calibration_images, input_name
            )
            success = quantize_static_model(
                working_model, output_path, calibration_reader
            )

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


def measure_inference_time(
    model_path: Path, test_images_dir: Path, num_runs: int = 5
) -> dict[str, float]:
    """
    Mock inference time measurement (Python inference disabled).

    NOTE: Python inference timing was unreliable due to preprocessing issues.
    Use Beaker CLI for accurate performance measurements.

    Returns:
        Mock timing dict with zeros (for backward compatibility)
    """
    logger.warning(
        f"Skipping Python inference timing for {model_path.name}. "
        "Use Beaker CLI for accurate performance measurements."
    )

    return {
        "mean_ms": 0.0,
        "std_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
    }
