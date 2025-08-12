"""
Inference comparison utilities using the beaker CLI tool.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def run_beaker_inference(
    model_path: Path, image_path: Path, output_dir: Path, model_type: str = "detect"
) -> Path:
    """
    Run inference using the beaker CLI tool.

    Args:
        model_path: Path to the ONNX model
        image_path: Path to the input image
        output_dir: Directory to save output
        model_type: Type of model ("detect" or "cutout")

    Returns:
        Path to the output image
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the beaker command based on model type
    cmd = ["beaker", model_type]

    if model_type == "detect":
        cmd.append("--bounding-box")

    cmd.extend(
        [
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            str(image_path),
        ]
    )

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Beaker output: {result.stdout}")

        # Find the output image based on beaker's naming convention.
        if model_type == "detect":
            # Beaker detect creates: {input_name}_bounding-box.jpg
            output_name = image_path.stem + "_bounding-box" + ".jpg"
        elif model_type == "cutout":
            # Beaker cutout creates: {input_name}_cutout.png (assuming similar pattern)
            output_name = image_path.stem + "_cutout" + ".png"
        else:
            # Fallback to generic pattern
            output_name = image_path.stem + "_output" + image_path.suffix

        output_path = output_dir / output_name

        if not output_path.exists():
            raise FileNotFoundError(f"No output image found in {output_dir}")

        logger.debug(f"Found beaker output: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Beaker command failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("Beaker CLI not found. Make sure it's installed and in PATH.")
        raise


def create_side_by_side_comparison(
    original_output: Path,
    quantized_output: Path,
    comparison_output: Path,
    labels: Tuple[str, str] = ("Original", "Quantized"),
) -> Path:
    """
    Create a side-by-side comparison image.

    Args:
        original_output: Path to original model output
        quantized_output: Path to quantized model output
        comparison_output: Path to save comparison image
        labels: Labels for the images

    Returns:
        Path to the comparison image
    """
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    # Load images
    img1 = Image.open(original_output)
    img2 = Image.open(quantized_output)

    # Resize images to same height if needed
    if img1.height != img2.height:
        target_height = min(img1.height, img2.height)
        img1 = img1.resize(
            (int(img1.width * target_height / img1.height), target_height)
        )
        img2 = img2.resize(
            (int(img2.width * target_height / img2.height), target_height)
        )

    # Create comparison image
    total_width = img1.width + img2.width
    comparison = Image.new("RGB", (total_width, img1.height), color="white")

    # Paste images side by side
    comparison.paste(img1, (0, 0))
    comparison.paste(img2, (img1.width, 0))

    # Add labels
    logger.info(f"Adding labels: {labels}")

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default(20)

    # Draw labels at the top
    label0_bbox = font.getbbox(labels[0])
    label1_bbox = font.getbbox(labels[1])
    label0_width = label0_bbox[2] - label0_bbox[0]
    label1_width = label1_bbox[2] - label1_bbox[0]
    # White background for labels
    draw.rectangle(
        [
            (img1.width // 2 - label0_width // 2 - 5, 0),
            (img1.width // 2 + label0_width // 2 + 5, 30),
        ],
        fill="white",
    )
    draw.rectangle(
        [
            (img1.width + img2.width // 2 - label1_width // 2 - 5, 0),
            (img1.width + img2.width // 2 + label1_width // 2 + 5, 30),
        ],
        fill="white",
    )
    # Draw text
    draw.text(
        (img1.width // 2 - label0_width // 2, 5), labels[0], fill="black", font=font
    )
    draw.text(
        (img1.width + img2.width // 2 - label1_width // 2, 5),
        labels[1],
        fill="black",
        font=font,
    )
    # Save comparison
    comparison.save(comparison_output)
    logger.info(f"Created comparison image: {comparison_output}")

    return comparison_output


def compare_models(
    original_model: Path,
    quantized_model: Path,
    test_image: Path,
    output_dir: Path,
    model_type: str = "detect",
) -> Path:
    """
    Compare original and quantized models by running inference and creating side-by-side comparison.

    Args:
        original_model: Path to original ONNX model
        quantized_model: Path to quantized ONNX model
        test_image: Path to test image
        output_dir: Directory to save outputs
        model_type: Type of model ("detect" or "cutout")

    Returns:
        Path to the comparison image
    """
    # Create subdirectories for outputs
    original_dir = output_dir / "original"
    quantized_dir = output_dir / "quantized"
    comparisons_dir = output_dir / "comparisons"

    # Run inference with both models
    logger.info(f"Running inference with original model: {original_model.name}")
    original_output = run_beaker_inference(
        original_model, test_image, original_dir, model_type
    )

    logger.info(f"Running inference with quantized model: {quantized_model.name}")
    quantized_output = run_beaker_inference(
        quantized_model, test_image, quantized_dir, model_type
    )

    # Create comparison
    comparison_name = f"comparison_{test_image.stem}_{quantized_model.stem}.png"
    comparison_path = comparisons_dir / comparison_name

    return create_side_by_side_comparison(
        original_output,
        quantized_output,
        comparison_path,
        labels=(original_model.stem, quantized_model.stem),
    )


def generate_all_comparisons(
    original_model: Path,
    quantized_models: List[Path],
    test_images: List[Path],
    output_dir: Path,
    model_type: str = "detect",
) -> List[Path]:
    """
    Generate comparison images for multiple quantized models and test images.

    Args:
        original_model: Path to original ONNX model
        quantized_models: List of quantized model paths
        test_images: List of test image paths
        output_dir: Directory to save outputs
        model_type: Type of model ("detect" or "cutout")

    Returns:
        List of comparison image paths
    """
    comparison_images = []

    for quantized_model in quantized_models:
        for test_image in test_images:
            try:
                comparison_path = compare_models(
                    original_model, quantized_model, test_image, output_dir, model_type
                )
                comparison_images.append(comparison_path)
                logger.info(f"✓ Generated comparison: {comparison_path.name}")

            except Exception as e:
                logger.warning(
                    f"Failed to generate comparison for {quantized_model.name} with {test_image.name}: {e}"
                )

    return comparison_images
