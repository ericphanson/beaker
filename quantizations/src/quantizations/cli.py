"""
Command-line interface for model quantization.
"""

import logging
from pathlib import Path

import click

from . import downloader, quantizer, uploader, inference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """Beaker model quantization tools."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./models"),
    help="Directory to save downloaded models",
)
@click.option(
    "--model-type",
    type=click.Choice(["detect", "cutout", "all"]),
    default="all",
    help="Which models to download",
)
def download(output_dir: Path, model_type: str) -> None:
    """Download ONNX models from GitHub releases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type in ["detect", "all"]:
        logger.info("Downloading detect detection model...")
        downloader.download_detect_model(output_dir)

    if model_type in ["cutout", "all"]:
        logger.info("Downloading cutout model...")
        downloader.download_cutout_model(output_dir)

    logger.info("Download complete!")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./quantized"),
    help="Directory to save quantized models",
)
@click.option(
    "--levels",
    multiple=True,
    default=["dynamic", "static"],
    type=click.Choice(["dynamic", "static", "int8", "fp16"]),
    help="Quantization levels to apply",
)
def quantize(model_path: Path, output_dir: Path, levels: tuple[str, ...]) -> None:
    """Quantize an ONNX model at specified levels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        logger.info(f"Applying {level} quantization to {model_path.name}...")
        output_path = quantizer.quantize_model(model_path, output_dir, level)
        logger.info(f"Saved quantized model: {output_path}")


@cli.command()
@click.argument("original_model", type=click.Path(exists=True, path_type=Path))
@click.argument("quantized_model", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--test-image",
    type=click.Path(exists=True, path_type=Path),
    help="Test image for comparison",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./comparisons"),
    help="Directory to save comparison images",
)
@click.option(
    "--model-type",
    type=click.Choice(["detect", "cutout"]),
    default="detect",
    help="Type of model",
)
def compare(
    original_model: Path,
    quantized_model: Path,
    test_image: Path | None,
    output_dir: Path,
    model_type: str,
) -> None:
    """Compare original and quantized models by generating side-by-side inference results."""
    # Use default test image if not specified
    if test_image is None:
        # Try to find example images
        current_path = Path(__file__).parent
        repo_root = None

        # Search up the directory tree for example images
        for _ in range(6):  # Reasonable limit
            if (current_path / "example.jpg").exists() or current_path.name == "beaker":
                repo_root = current_path
                break
            current_path = current_path.parent

        if repo_root:
            # Look for example images
            for pattern in ["example*.jpg", "example*.png"]:
                examples = list(repo_root.glob(pattern))
                if examples:
                    test_image = examples[0]
                    break

        if test_image is None:
            raise click.ClickException(
                "No test image specified and no example images found"
            )

    logger.info(f"Comparing {quantized_model.name} against {original_model.name}...")
    logger.info(f"Using test image: {test_image}")

    try:
        comparison_path = inference.compare_models(
            original_model, quantized_model, test_image, output_dir, model_type
        )
        logger.info(f"✓ Comparison complete! Saved to: {comparison_path}")
    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
        raise click.ClickException(f"Comparison failed: {e}")


@cli.command()
@click.argument("quantized_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model-type",
    type=click.Choice(["detect", "cutout"]),
    required=True,
    help="Type of model being uploaded",
)
@click.option("--version", default="v1", help="Version tag for the release")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be uploaded without actually doing it",
)
@click.option(
    "--include-comparisons",
    is_flag=True,
    help="Generate and include comparison images in the release",
)
@click.option(
    "--test-image",
    type=click.Path(exists=True, path_type=Path),
    help="Test image for comparison generation",
)
def upload(
    quantized_dir: Path,
    model_type: str,
    version: str,
    dry_run: bool,
    include_comparisons: bool,
    test_image: Path | None,
) -> None:
    """Upload quantized models to GitHub releases."""
    logger.info(f"Uploading {model_type} quantizations (version: {version})...")

    # Prepare additional data for enhanced uploads
    performance_table = ""
    comparison_images = []
    base_model_info = f"Base model type: {model_type}"

    if include_comparisons:
        logger.info("Generating comparison images...")
        try:
            # Find original and quantized models
            original_models = list(quantized_dir.glob(f"{model_type}-optimized.onnx"))
            if not original_models:
                original_models = list(quantized_dir.parent.glob("models/*.onnx"))

            if original_models:
                original_model = original_models[0]
                quantized_models = uploader.collect_quantized_models(
                    quantized_dir, model_type
                )

                # Use provided test image or find one
                if test_image is None:
                    # Look for example images
                    current_path = Path(__file__).parent
                    for _ in range(6):
                        if (
                            current_path / "example.jpg"
                        ).exists() or current_path.name == "beaker":
                            for pattern in ["example*.jpg", "example*.png"]:
                                examples = list(current_path.glob(pattern))
                                if examples:
                                    test_image = examples[0]
                                    break
                            break
                        current_path = current_path.parent

                if test_image and quantized_models:
                    # Generate comparison images
                    comparison_output = quantized_dir / "comparisons"
                    comparison_images = inference.generate_all_comparisons(
                        original_model,
                        quantized_models[:3],  # Limit to 3 models
                        [test_image],
                        comparison_output,
                        model_type,
                    )

                    logger.info(f"Generated {len(comparison_images)} comparison images")
        except Exception as e:
            logger.warning(f"Failed to generate comparison images: {e}")

    uploader.upload_quantizations(
        quantized_dir,
        model_type,
        version,
        dry_run,
        performance_table,
        comparison_images,
        base_model_info,
    )

    if not dry_run:
        logger.info("Upload complete!")
    else:
        logger.info("Dry run complete - no actual upload performed")


@cli.command()
@click.option(
    "--model-type",
    type=click.Choice(["detect", "cutout", "all"]),
    default="all",
    help="Which models to process",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Base output directory",
)
@click.option("--version", default="v1", help="Version tag for releases")
@click.option("--dry-run", is_flag=True, help="Perform dry run without uploading")
@click.option(
    "--no-download", is_flag=True, help="Skip download step and use existing models"
)
@click.option(
    "--test-image",
    type=click.Path(exists=True, path_type=Path),
    help="Test image for comparison generation",
)
def full_pipeline(
    model_type: str,
    output_dir: Path,
    version: str,
    dry_run: bool,
    no_download: bool,
    test_image: Path | None,
) -> None:
    """Run the complete quantization pipeline: download -> quantize -> generate comparisons -> upload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    quantized_dir = output_dir / "quantized"
    comparisons_dir = output_dir / "comparisons"

    # Download models (unless --no-download is specified)
    if not no_download:
        logger.info("Step 1: Downloading models...")
        download.callback(models_dir, model_type)
    else:
        logger.info("Step 1: Skipping download (--no-download specified)")

    # Find downloaded models
    model_files = []
    if model_type in ["detect"]:
        detect_files = list(models_dir.glob("*detect*.onnx"))
        model_files.extend(detect_files)
    if model_type in ["cutout"]:
        cutout_files = list(models_dir.glob("*isnet*.onnx"))
        model_files.extend(cutout_files)

    if not model_files:
        if no_download:
            raise click.ClickException(
                f"No model files found in {models_dir}. Run without --no-download to download models first."
            )
        else:
            raise click.ClickException("No model files found after download")

    if len(model_files) > 1:
        raise click.ClickException(
            f"Multiple model files found: {', '.join([f.name for f in model_files])}. Please specify a single model."
        )

    original_model = model_files[0]
    # Quantize each model
    logger.info("Step 2: Quantizing models...")
    all_quantized = []
    for level in ["dynamic", "static"]:
        try:
            quantized_path = quantizer.quantize_model(
                original_model, quantized_dir, level
            )
            all_quantized.append((original_model, quantized_path))
            logger.info(f"✓ Created {quantized_path.name}")
        except Exception as e:
            logger.warning(
                f"Failed to create {level} quantization for {original_model.name}: {e}"
            )

    if not all_quantized:
        raise click.ClickException("No quantized models were created")

    # Find test image if not provided
    if test_image is None:
        current_path = Path(__file__).parent
        for _ in range(6):
            if (current_path / "example.jpg").exists() or current_path.name == "beaker":
                for pattern in ["example*.jpg", "example*.png"]:
                    examples = list(current_path.glob(pattern))
                    if examples:
                        test_image = examples[0]
                        break
                if test_image:
                    break
                current_path = current_path.parent

    # Generate comparisons and upload for each model type
    logger.info("Step 3: Generating comparisons...")

    # Create optimized version of original if it doesn't exist
    optimized_original = quantized_dir / f"{model_type}-optimized.onnx"
    if not optimized_original.exists():
        try:
            quantizer.optimize_model(original_model, optimized_original)
            logger.info(f"✓ Created optimized original: {optimized_original.name}")
        except Exception as e:
            logger.warning(f"Failed to optimize original model: {e}")
            optimized_original = original_model

    # Generate comparison images if test image is available
    comparison_images = []
    if test_image:
        logger.info(f"Generating comparison images for {model_type} models...")
        quantized_models = [quant for _, quant in all_quantized]

        try:
            comparison_images = inference.generate_all_comparisons(
                original_model,
                quantized_models,
                [test_image],
                comparisons_dir,
                model_type,
            )
            logger.info(
                f"Generated {len(comparison_images)} comparison images for {model_type}"
            )
        except Exception as e:
            logger.warning(f"Failed to generate comparisons for {model_type}: {e}")

        # Upload quantized models for this type
        logger.info(f"Step 4: Uploading {model_type} quantizations...")
        base_model_info = f"Base model: {original_model.name}"

        try:
            uploader.upload_quantizations(
                quantized_dir,
                model_type,
                version,
                dry_run,
                "",  # No performance table for now
                comparison_images,
                base_model_info,
            )
        except Exception as e:
            logger.warning(f"Failed to upload {model_type} models: {e}")

    logger.info("Full pipeline complete!")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
