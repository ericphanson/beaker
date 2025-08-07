"""
Command-line interface for model quantization.
"""

import logging
from pathlib import Path

import click

from . import downloader, quantizer, uploader, validator

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
    type=click.Choice(["head", "cutout", "all"]),
    default="all",
    help="Which models to download",
)
def download(output_dir: Path, model_type: str) -> None:
    """Download ONNX models from GitHub releases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type in ["head", "all"]:
        logger.info("Downloading head detection model...")
        downloader.download_head_model(output_dir)

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
    type=click.Choice(["dynamic", "static", "int8"]),
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
    "--test-images",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing test images",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.01,
    help="Maximum allowed difference in predictions",
)
def validate(
    original_model: Path,
    quantized_model: Path,
    test_images: Path | None,
    tolerance: float,
) -> None:
    """Validate quantized model against original."""
    # Use default test images if not specified
    if test_images is None:
        test_images = Path(__file__).parent.parent.parent.parent.parent / "test_images"
        if not test_images.exists():
            test_images = Path(__file__).parent.parent.parent.parent.parent

    logger.info(f"Validating {quantized_model.name} against {original_model.name}...")

    is_valid, max_diff = validator.validate_models(
        original_model, quantized_model, test_images, tolerance
    )

    if is_valid:
        logger.info(f"✓ Validation passed! Max difference: {max_diff:.6f}")
    else:
        logger.error(
            f"✗ Validation failed! Max difference: {max_diff:.6f} > {tolerance}"
        )
        raise click.ClickException("Validation failed")


@cli.command()
@click.argument("quantized_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model-type",
    type=click.Choice(["head", "cutout"]),
    required=True,
    help="Type of model being uploaded",
)
@click.option("--version", default="v1", help="Version tag for the release")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be uploaded without actually doing it",
)
def upload(quantized_dir: Path, model_type: str, version: str, dry_run: bool) -> None:
    """Upload quantized models to GitHub releases."""
    logger.info(f"Uploading {model_type} quantizations (version: {version})...")

    uploader.upload_quantizations(quantized_dir, model_type, version, dry_run)

    if not dry_run:
        logger.info("Upload complete!")
    else:
        logger.info("Dry run complete - no actual upload performed")


@cli.command()
@click.option(
    "--model-type",
    type=click.Choice(["head", "cutout", "all"]),
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
@click.option(
    "--tolerance",
    type=float,
    default=0.01,
    help="Maximum allowed difference in validation",
)
@click.option("--dry-run", is_flag=True, help="Perform dry run without uploading")
def full_pipeline(
    model_type: str, output_dir: Path, version: str, tolerance: float, dry_run: bool
) -> None:
    """Run the complete quantization pipeline: download -> quantize -> validate -> upload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    quantized_dir = output_dir / "quantized"

    # Download models
    logger.info("Step 1: Downloading models...")
    download.callback(models_dir, model_type)

    # Find downloaded models
    model_files = []
    if model_type in ["head", "all"]:
        head_files = list(models_dir.glob("*head*.onnx")) + list(
            models_dir.glob("best.onnx")
        )
        model_files.extend(head_files)
    if model_type in ["cutout", "all"]:
        cutout_files = list(models_dir.glob("*cutout*.onnx"))
        model_files.extend(cutout_files)

    if not model_files:
        raise click.ClickException("No model files found after download")

    # Quantize each model
    logger.info("Step 2: Quantizing models...")
    all_quantized = []
    for model_file in model_files:
        for level in ["dynamic", "static"]:
            quantized_path = quantizer.quantize_model(model_file, quantized_dir, level)
            all_quantized.append((model_file, quantized_path))

    # Validate quantized models
    logger.info("Step 3: Validating quantized models...")
    # Use relative path to find test images
    test_images_dir = Path("../")
    for original, quantized_path in all_quantized:
        is_valid, max_diff = validator.validate_models(
            original, quantized_path, test_images_dir, tolerance
        )
        if not is_valid:
            logger.error(f"Validation failed for {quantized_path.name}")
            raise click.ClickException(f"Validation failed for {quantized_path.name}")
        logger.info(f"✓ {quantized_path.name} validated (max diff: {max_diff:.6f})")

    # Upload quantized models
    logger.info("Step 4: Uploading quantized models...")
    if model_type in ["head", "all"]:
        head_quantized = quantized_dir.glob("*head*")
        if any(head_quantized):
            uploader.upload_quantizations(quantized_dir, "head", version, dry_run)

    if model_type in ["cutout", "all"]:
        cutout_quantized = list(quantized_dir.glob("*cutout*"))
        if cutout_quantized:
            uploader.upload_quantizations(quantized_dir, "cutout", version, dry_run)

    logger.info("Full pipeline complete!")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
