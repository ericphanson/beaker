"""
Upload quantized models to GitHub releases.
"""

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            logger.info("GitHub CLI is authenticated")
            return True
        else:
            logger.error("GitHub CLI is not authenticated")
            return False
    except FileNotFoundError:
        logger.error("GitHub CLI (gh) not found. Please install it first.")
        return False


def install_gh_cli() -> bool:
    """Install GitHub CLI using package manager."""
    try:
        logger.info("Attempting to install GitHub CLI...")

        # Try different installation methods
        install_commands = [
            [
                "curl",
                "-fsSL",
                "https://cli.github.com/packages/githubcli-archive-keyring.gpg",
                "|",
                "sudo",
                "dd",
                "of=/usr/share/keyrings/githubcli-archive-keyring.gpg",
            ],
            [
                "echo",
                '"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main"',
                "|",
                "sudo",
                "tee",
                "/etc/apt/sources.list.d/github-cli.list",
            ],
            ["sudo", "apt", "update"],
            ["sudo", "apt", "install", "gh", "-y"],
        ]

        # For simplicity, try using snap if available
        try:
            result = subprocess.run(
                ["snap", "install", "gh"], capture_output=True, text=True, check=True
            )
            logger.info("GitHub CLI installed via snap")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try using apt-get directly
        try:
            result = subprocess.run(
                ["apt-get", "install", "gh", "-y"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("GitHub CLI installed via apt-get")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        logger.error("Could not install GitHub CLI automatically")
        return False

    except Exception as e:
        logger.error(f"Failed to install GitHub CLI: {e}")
        return False


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def collect_quantized_models(quantized_dir: Path, model_type: str) -> list[Path]:
    """Collect quantized model files for a specific model type."""
    # Look for models with the specific type in the name
    pattern_variants = [f"{model_type}-*.onnx", f"*{model_type}*.onnx"]

    model_files = []
    for pattern in pattern_variants:
        model_files.extend(quantized_dir.glob(pattern))

    # Filter out files that don't contain quantization indicators
    quantization_indicators = ["dynamic", "static", "int8", "fp16"]
    quantized_files = []

    for model_file in model_files:
        if any(indicator in model_file.name for indicator in quantization_indicators):
            quantized_files.append(model_file)

    # Remove duplicates and sort
    quantized_files = list(set(quantized_files))
    quantized_files.sort(key=lambda x: x.name)

    logger.info(f"Found {len(quantized_files)} quantized {model_type} models")
    return quantized_files


def generate_release_notes(
    model_files: list[Path],
    model_type: str,
    performance_table: str = "",
    base_model_info: str = "",
) -> str:
    """Generate release notes with model information, checksums, and performance metrics."""
    notes = [f"# {model_type.title()} Model Quantizations"]
    notes.append("")
    notes.append(
        f"This release contains quantized versions of the Beaker {model_type} detection models "
        "with optimizations and multiple quantization techniques."
    )
    notes.append("")

    if base_model_info:
        notes.append("## Base Model Information")
        notes.append("")
        notes.append(base_model_info)
        notes.append("")

    if performance_table:
        notes.append("## Performance Metrics")
        notes.append("")
        notes.append(
            "Performance measured on 4 example images with 5 inference runs each:"
        )
        notes.append("")
        notes.append(performance_table)
        notes.append("")
        notes.append(
            "**Note**: These metrics are based on a limited set of example images and may not be representative of general performance on diverse datasets."
        )
        notes.append("")

    notes.append("## Models Included")
    notes.append("")

    for model_file in model_files:
        checksum = calculate_file_checksum(model_file)
        size_mb = model_file.stat().st_size / (1024 * 1024)

        # Extract quantization type from filename
        quantization_type = "Unknown"
        if "fp16" in model_file.name:
            quantization_type = "FP16"
        elif "dynamic-int8" in model_file.name:
            quantization_type = "Dynamic INT8"
        elif "static-int8" in model_file.name:
            quantization_type = "Static INT8"
        elif "dynamic" in model_file.name:
            quantization_type = "Dynamic"
        elif "static" in model_file.name:
            quantization_type = "Static"
        elif "int8" in model_file.name:
            quantization_type = "INT8"

        notes.append(f"### {model_file.name}")
        notes.append(f"- **Type**: {quantization_type} Quantization")
        notes.append(f"- **Size**: {size_mb:.1f} MB")
        notes.append(f"- **SHA256**: `{checksum}`")
        notes.append("")

    notes.append("## Usage")
    notes.append("")
    notes.append("These quantized models are compatible with the Beaker CLI tool.")
    notes.append("You can use them by setting the appropriate environment variables:")
    notes.append("")
    notes.append("```bash")
    if model_type == "head":
        notes.append("# For head detection")
        notes.append("export BEAKER_HEAD_MODEL_URL=<download_url>")
        notes.append("beaker head image.jpg --crop")
    elif model_type == "cutout":
        notes.append("# For cutout processing")
        notes.append("export BEAKER_CUTOUT_MODEL_URL=<download_url>")
        notes.append("beaker cutout image.jpg")
    notes.append("```")
    notes.append("")

    notes.append("## Optimizations Applied")
    notes.append("")
    notes.append("All models include the following optimizations:")
    notes.append(
        "- **ONNX Simplification**: Models are optimized using onnx-simplifier"
    )
    notes.append(
        "- **Graph Optimization**: Redundant operations removed and computation graphs simplified"
    )
    notes.append(
        "- **Quantization**: Multiple quantization levels available (FP16, Dynamic INT8, Static INT8)"
    )
    notes.append("")

    notes.append("## Model Comparison")
    notes.append("")
    notes.append("Quantized models typically provide:")
    notes.append("- **Reduced model size**: 50-75% smaller file size")
    notes.append("- **Faster CPU inference**: Improved performance on CPU-only systems")
    notes.append("- **Maintained accuracy**: Minimal accuracy loss (typically <1%)")
    notes.append("- **Better compatibility**: Optimized for deployment environments")

    return "\n".join(notes)


def create_github_release(
    tag_name: str,
    title: str,
    notes: str,
    model_files: list[Path],
    comparison_images: list[Path] = None,
    dry_run: bool = False,
) -> bool:
    """Create a GitHub release with the specified models and comparison images."""
    try:
        if dry_run:
            logger.info(
                f"DRY RUN: Would create release '{tag_name}' with title '{title}'"
            )
            logger.info(f"DRY RUN: Would upload {len(model_files)} model files:")
            for model_file in model_files:
                logger.info(f"  - {model_file.name}")
            if comparison_images:
                logger.info(
                    f"DRY RUN: Would upload {len(comparison_images)} comparison images:"
                )
                for img_file in comparison_images:
                    logger.info(f"  - {img_file.name}")
            logger.info("DRY RUN: Release notes:")
            for line in notes.split("\n")[:10]:  # Show first 10 lines
                logger.info(f"    {line}")
            logger.info("    ... (truncated)")
            return True

        # Create the release
        logger.info(f"Creating GitHub release: {tag_name}")

        cmd = [
            "gh",
            "release",
            "create",
            tag_name,
            "--title",
            title,
            "--notes",
            notes,
            "--draft",  # Create as draft first
        ]

        # Add model files
        for model_file in model_files:
            cmd.append(str(model_file))

        # Add comparison images if provided
        if comparison_images:
            for img_file in comparison_images:
                cmd.append(str(img_file))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"Release created successfully: {tag_name}")
        logger.info("Release URL:", result.stdout.strip())

        # Publish the release (remove draft status)
        subprocess.run(
            ["gh", "release", "edit", tag_name, "--draft=false"],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"Release published: {tag_name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create GitHub release: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False


def check_release_exists(tag_name: str) -> bool:
    """Check if a GitHub release with the given tag already exists."""
    try:
        result = subprocess.run(
            ["gh", "release", "view", tag_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def upload_quantizations(
    quantized_dir: Path,
    model_type: str,
    version: str,
    dry_run: bool = False,
    performance_table: str = "",
    comparison_images: list[Path] = None,
    base_model_info: str = "",
) -> bool:
    """Upload quantized models to GitHub releases."""
    try:
        # Check if GitHub CLI is available
        if not check_gh_cli():
            if not install_gh_cli():
                logger.error("Cannot proceed without GitHub CLI")
                return False

            # Check again after installation
            if not check_gh_cli():
                logger.error("GitHub CLI installation failed or not authenticated")
                return False

        # Collect quantized models
        model_files = collect_quantized_models(quantized_dir, model_type)
        if not model_files:
            logger.error(f"No quantized {model_type} models found in {quantized_dir}")
            return False

        # Generate release information
        tag_name = f"{model_type}-quantizations-{version}"
        title = f"{model_type.title()} Model Quantizations {version}"
        notes = generate_release_notes(
            model_files, model_type, performance_table, base_model_info
        )

        # Check if release already exists
        if check_release_exists(tag_name):
            logger.warning(f"Release {tag_name} already exists")
            if not dry_run:
                response = input(
                    f"Do you want to overwrite release {tag_name}? (y/N): "
                )
                if response.lower() != "y":
                    logger.info("Upload cancelled")
                    return False

                # Delete existing release
                logger.info(f"Deleting existing release: {tag_name}")
                subprocess.run(
                    ["gh", "release", "delete", tag_name, "--yes"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

        # Create the release
        success = create_github_release(
            tag_name, title, notes, model_files, comparison_images, dry_run
        )

        if success:
            logger.info(
                f"Successfully {'would upload' if dry_run else 'uploaded'} {len(model_files)} quantized models"
            )
            if comparison_images:
                logger.info(
                    f"{'Would include' if dry_run else 'Included'} {len(comparison_images)} comparison images"
                )

        return success

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


def list_available_quantizations() -> dict[str, list[str]]:
    """List available quantized models from GitHub releases."""
    try:
        result = subprocess.run(
            ["gh", "release", "list"], capture_output=True, text=True, check=True
        )

        releases = []
        for line in result.stdout.strip().split("\n"):
            if "quantizations" in line:
                parts = line.split("\t")
                if parts:
                    releases.append(parts[0])  # Tag name

        # Organize by model type
        quantizations = {"head": [], "cutout": []}
        for release in releases:
            if "head" in release:
                quantizations["head"].append(release)
            elif "cutout" in release:
                quantizations["cutout"].append(release)

        return quantizations

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not list releases (GitHub CLI not available)")
        return {"head": [], "cutout": []}
