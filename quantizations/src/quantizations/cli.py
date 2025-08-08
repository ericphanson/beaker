"""
Command-line interface for model quantization.
"""

import logging
from pathlib import Path

import click

from . import downloader, quantizer, uploader, validator, comparisons

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
        head_dir = output_dir / "head"
        head_dir.mkdir(parents=True, exist_ok=True)
        downloader.download_head_model(head_dir)

    if model_type in ["cutout", "all"]:
        logger.info("Downloading cutout model...")
        cutout_dir = output_dir / "cutout"
        cutout_dir.mkdir(parents=True, exist_ok=True)
        downloader.download_cutout_model(cutout_dir)

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
@click.option(
    "--model-type",
    type=click.Choice(["head", "cutout"]),
    help="Model type (auto-detected if not specified)",
)
def quantize(
    model_path: Path,
    output_dir: Path,
    levels: tuple[str, ...],
    model_type: str | None = None,
) -> None:
    """Quantize an ONNX model at specified levels."""

    # Create model-type-specific output directory
    if model_type:
        quantized_output_dir = output_dir / model_type
    else:
        # Auto-detect model type from path/filename
        if "head" in str(model_path).lower() or "best" in model_path.stem.lower():
            quantized_output_dir = output_dir / "head"
        elif "cutout" in str(model_path).lower():
            quantized_output_dir = output_dir / "cutout"
        else:
            quantized_output_dir = output_dir / "unknown"

    quantized_output_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        logger.info(f"Applying {level} quantization to {model_path.name}...")
        output_path = quantizer.quantize_model(model_path, quantized_output_dir, level)
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
        # Try to find the beaker repository root
        current_path = Path(__file__).parent
        repo_root = None

        # Search up the directory tree for the beaker repository
        for _ in range(6):  # Reasonable limit
            if (current_path / "example.jpg").exists() or current_path.name == "beaker":
                repo_root = current_path
                break
            current_path = current_path.parent

        if repo_root and (repo_root / "example.jpg").exists():
            test_images = repo_root
        else:
            test_images = Path.cwd()  # Default to current directory

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
@click.option(
    "--include-comparisons",
    is_flag=True,
    help="Generate and include comparison images in the release",
)
@click.option(
    "--test-images",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing test images for comparison generation",
)
@click.option(
    "--beaker-cargo-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to Beaker Cargo.toml file (default: ../beaker/Cargo.toml)",
)
def upload(
    quantized_dir: Path,
    model_type: str,
    version: str,
    dry_run: bool,
    include_comparisons: bool,
    test_images: Path | None,
    beaker_cargo_path: Path | None,
) -> None:
    """Upload quantized models to GitHub releases."""
    logger.info(f"Uploading {model_type} quantizations (version: {version})...")

    # Use model-type-specific quantized directory
    model_quantized_dir = quantized_dir / model_type
    if not model_quantized_dir.exists():
        # Fallback to original directory structure
        model_quantized_dir = quantized_dir
        logger.warning(
            f"Model-specific directory {model_quantized_dir} not found, using {quantized_dir}"
        )

    # Prepare additional data for enhanced uploads
    performance_table = ""
    comparison_images = []
    base_model_info = f"Base model type: {model_type}"

    if include_comparisons and test_images:
        logger.info("Generating comparison images...")
        try:
            # Find original and quantized models
            original_models = list(
                model_quantized_dir.glob(f"{model_type}-optimized.onnx")
            )
            if not original_models:
                # Look in the models directory with model-type subdirectory
                models_parent = quantized_dir.parent / "models" / model_type
                if models_parent.exists():
                    original_models = list(models_parent.glob("*.onnx"))
                else:
                    # Fallback to old structure
                    original_models = list(quantized_dir.parent.glob("models/*.onnx"))

            if original_models:
                original_model = original_models[0]
                quantized_models = uploader.collect_quantized_models(
                    model_quantized_dir, model_type
                )

                # Collect test images
                if test_images.is_file():
                    test_image_list = [test_images]
                else:
                    test_image_list = []
                    for ext in [".jpg", ".jpeg", ".png"]:
                        test_image_list.extend(test_images.glob(f"*{ext}"))

                if test_image_list and quantized_models:
                    # Create model dict for comparison
                    models = {"Original": original_model}
                    for q_model in quantized_models:
                        model_name = q_model.stem.replace(f"{model_type}-", "").title()
                        models[model_name] = q_model

                    # Generate comparison images with model-type prefix
                    comparison_output = (
                        quantized_dir.parent / "comparisons" / model_type
                    )
                    comparison_images = comparisons.generate_model_comparison_images(
                        models,
                        test_image_list[:4],
                        comparison_output,  # Limit to 4 images
                        0.5,  # confidence_threshold
                        beaker_cargo_path,
                    )

                    logger.info(f"Generated {len(comparison_images)} comparison images")
        except Exception as e:
            logger.warning(f"Failed to generate comparison images: {e}")

    uploader.upload_quantizations(
        model_quantized_dir,
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
    default=200.0,
    help="Maximum allowed difference in validation",
)
@click.option("--dry-run", is_flag=True, help="Perform dry run without uploading")
@click.option(
    "--beaker-cargo-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to Beaker Cargo.toml file (default: ../beaker/Cargo.toml)",
)
def full_pipeline(
    model_type: str,
    output_dir: Path,
    version: str,
    tolerance: float,
    dry_run: bool,
    beaker_cargo_path: Path | None,
) -> None:
    """Run the complete quantization pipeline: download -> quantize -> validate -> upload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    quantized_dir = output_dir / "quantized"
    comparisons_dir = output_dir / "comparisons"

    # Download models
    logger.info("Step 1: Downloading models...")

    if model_type in ["head", "all"]:
        head_dir = models_dir / "head"
        head_dir.mkdir(parents=True, exist_ok=True)
        downloader.download_head_model(head_dir)

    if model_type in ["cutout", "all"]:
        cutout_dir = models_dir / "cutout"
        cutout_dir.mkdir(parents=True, exist_ok=True)
        downloader.download_cutout_model(cutout_dir)

    # Find downloaded models
    model_files = []
    if model_type in ["head", "all"]:
        head_dir = models_dir / "head"
        if head_dir.exists():
            head_files = list(head_dir.glob("*.onnx"))
            model_files.extend(head_files)
        else:
            # Fallback to old structure
            head_files = list(models_dir.glob("*head*.onnx")) + list(
                models_dir.glob("best.onnx")
            )
            model_files.extend(head_files)

    if model_type in ["cutout", "all"]:
        cutout_dir = models_dir / "cutout"
        if cutout_dir.exists():
            cutout_files = list(cutout_dir.glob("*.onnx"))
            model_files.extend(cutout_files)
        else:
            # Fallback to old structure
            cutout_files = list(models_dir.glob("*cutout*.onnx"))
            model_files.extend(cutout_files)

    if not model_files:
        raise click.ClickException("No model files found after download")

    # Quantize each model
    logger.info("Step 2: Quantizing models...")
    all_quantized = []
    for model_file in model_files:
        # Determine model type from path
        detected_model_type = "unknown"
        if (
            "head" in str(model_file).lower()
            or model_file.parent.name == "head"
            or "best" in model_file.stem.lower()
        ):
            detected_model_type = "head"
        elif "cutout" in str(model_file).lower() or model_file.parent.name == "cutout":
            detected_model_type = "cutout"

        # Create model-type-specific quantized directory
        model_quantized_dir = quantized_dir / detected_model_type
        model_quantized_dir.mkdir(parents=True, exist_ok=True)

        for level in ["dynamic", "static"]:
            try:
                quantized_path = quantizer.quantize_model(
                    model_file, model_quantized_dir, level
                )
                all_quantized.append((model_file, quantized_path, detected_model_type))
                logger.info(f"✓ Created {quantized_path.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to create {level} quantization for {model_file.name}: {e}"
                )

    if not all_quantized:
        raise click.ClickException("No quantized models were created")

    # Validate quantized models with enhanced metrics
    logger.info("Step 3: Validating quantized models...")

    # Find test images using the same logic as validate command
    current_path = Path(__file__).parent
    test_images_dir = None

    for _ in range(6):
        if (current_path / "example.jpg").exists() or current_path.name == "beaker":
            test_images_dir = current_path
            break
        current_path = current_path.parent

    if not test_images_dir:
        test_images_dir = Path.cwd()

    validation_results = {}
    timing_results = {}

    for original, quantized_path, detected_model_type in all_quantized:
        try:
            is_valid, max_diff, detailed_metrics = (
                validator.validate_models_with_timing(
                    original, quantized_path, test_images_dir, tolerance
                )
            )
            if not is_valid:
                logger.error(f"Validation failed for {quantized_path.name}")
                raise click.ClickException(
                    f"Validation failed for {quantized_path.name}"
                )

            model_name = quantized_path.stem
            validation_results[model_name] = detailed_metrics
            timing_results[model_name] = detailed_metrics["quantized_timing"]

            logger.info(f"✓ {quantized_path.name} validated (max diff: {max_diff:.6f})")
        except Exception as e:
            logger.warning(f"Validation failed for {quantized_path.name}: {e}")

    # Generate comparison images and performance tables
    logger.info("Step 4: Generating comparisons and metrics...")

    # Find test images using same logic
    test_images = []
    for pattern in ["example*.jpg", "example*.png"]:
        test_images.extend(test_images_dir.glob(pattern))
    test_images = test_images[:4]  # Limit to 4 images

    # Process each model type
    for current_type in ["head", "cutout"]:
        if model_type not in [current_type, "all"]:
            continue

        # Find models for this type
        type_models = {}
        original_model = None

        for original, quantized_path, detected_model_type in all_quantized:
            if detected_model_type == current_type:
                if not original_model:
                    # Find or create optimized original
                    original_model = original
                    optimized_original = (
                        quantized_dir / current_type / f"{current_type}-optimized.onnx"
                    )
                    if not optimized_original.exists():
                        quantizer.optimize_model(original, optimized_original)
                    type_models["Original"] = optimized_original

                # Add quantized model
                model_name = quantized_path.stem.replace(f"{current_type}-", "").title()
                type_models[model_name] = quantized_path

        if (
            len(type_models) > 1 and test_images
        ):  # Need original + at least one quantized
            logger.info(f"Generating comparison images for {current_type} models...")
            try:
                # Create model-type-specific comparisons directory
                type_comparisons_dir = comparisons_dir / current_type
                comparison_images = comparisons.generate_model_comparison_images(
                    type_models,
                    test_images,
                    type_comparisons_dir,
                    0.5,
                    beaker_cargo_path,
                )
                logger.info(
                    f"Generated {len(comparison_images)} comparison images for {current_type}"
                )

                # Create performance table
                type_validation = {
                    k: v for k, v in validation_results.items() if current_type in k
                }
                type_timing = {
                    k: v for k, v in timing_results.items() if current_type in k
                }
                performance_table = comparisons.create_performance_table(
                    type_models, test_images, type_timing, type_validation
                )

                # Upload quantized models for this type
                logger.info(f"Step 5: Uploading {current_type} quantizations...")
                base_model_info = f"Base model: {original_model.name if original_model else 'Unknown'}"

                # Use model-type-specific quantized directory
                type_quantized_dir = quantized_dir / current_type

                uploader.upload_quantizations(
                    type_quantized_dir,
                    current_type,
                    version,
                    dry_run,
                    performance_table,
                    comparison_images,
                    base_model_info,
                )

            except Exception as e:
                logger.warning(
                    f"Failed to generate comparisons for {current_type}: {e}"
                )
                # Upload without comparisons
                type_quantized_dir = quantized_dir / current_type
                uploader.upload_quantizations(
                    type_quantized_dir, current_type, version, dry_run
                )

    logger.info("Full pipeline complete!")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
