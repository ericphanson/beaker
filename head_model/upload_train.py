#!/usr/bin/env python3
"""
Release script for bird-head-detector models.

This script:
1. Checks if the repository is clean (no uncommitted changes)
2. Shows existing tags/versions
3. Finds available model files or uses specified model
4. Prompts for a new version number (or uses provided version)
5. Creates a git tag with format 'bird-head-detector-v{version}'
6. Uploads the selected model as a GitHub release asset
7. Pushes everything to the remote repository

Requirements:
- gh CLI tool installed and authenticated
- Clean git repository (no uncommitted changes)
- Trained model exists or is specified

Usage:
    # Interactive mode - choose from available models
    uv run python release.py

    # Specify model and version (creates tag: bird-head-detector-v1.2.0)
    uv run python release.py --model runs/detect/best_model/weights/best.pt --version 1.2.0

    # Specify just the model (will prompt for version)
    uv run python release.py --model my_custom_model.pt
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Import YOLO for ONNX export
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def run_command(cmd, capture=True, check=True, timeout=60):
    """Run a shell command and return the result."""
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
            )
            return result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check, timeout=timeout)
            return None, None
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds: {cmd}")
        return "", "Command timed out"
    except subprocess.CalledProcessError as e:
        if capture:
            return (
                e.stdout.strip() if e.stdout else "",
                e.stderr.strip() if e.stderr else "",
            )
        raise


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")

    # Check if gh CLI is installed
    stdout, stderr = run_command("gh --version", check=False)
    if not stdout:
        print("❌ GitHub CLI (gh) is not installed.")
        print("   Install it from: https://cli.github.com/")
        return False
    print(f"✅ GitHub CLI found: {stdout.split()[2]}")

    # Check if gh is authenticated
    stdout, stderr = run_command("gh auth status", check=False)
    if (
        "Logged in to github.com" not in stderr
        and "Logged in to github.com" not in stdout
    ):
        print("❌ GitHub CLI is not authenticated.")
        print("   Run: gh auth login")
        return False
    print("✅ GitHub CLI authenticated")

    # Check if we're in a git repository
    stdout, stderr = run_command("git rev-parse --git-dir", check=False)
    if not stdout:
        print("❌ Not in a git repository.")
        return False
    print("✅ Git repository detected")

    return True


def check_repo_clean():
    """Check if the repository has no uncommitted changes."""
    print("🧹 Checking repository status...")

    # Check for uncommitted changes
    stdout, stderr = run_command("git status --porcelain")
    if stdout:
        print("❌ Repository has uncommitted changes:")
        print(stdout)
        print("\n   Please commit or stash your changes before releasing.")
        return False

    print("✅ Repository is clean")
    return True


def get_existing_tags():
    """Get list of existing git tags."""
    stdout, stderr = run_command("git tag --sort=-version:refname")
    if stdout:
        return stdout.split("\n")
    return []


def validate_version(version):
    """Validate version format (semantic versioning)."""
    pattern = r"^v?\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$"
    return re.match(pattern, version) is not None


def generate_onnx_model(model_path, output_dir=None):
    """Generate ONNX model from PyTorch model."""
    if not YOLO_AVAILABLE:
        raise RuntimeError(
            "❌ Ultralytics package not available! ONNX export is required for releases.\n"
            "   Install with: uv sync --extra onnx"
        )

    try:
        print(f"🔄 Generating ONNX model from {model_path.name}...")

        # Load the model
        model = YOLO(str(model_path))

        # Determine output directory
        if output_dir is None:
            output_dir = model_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Generate ONNX file path
        onnx_path = output_dir / f"{model_path.stem}.onnx"

        # Export to ONNX with optimized settings
        print("   🚀 Exporting to ONNX format...")
        export_path = model.export(
            format="onnx",
            imgsz=640,  # Standard input size
            dynamic=False,  # Static input shape for better compatibility
            simplify=True,  # Simplify the ONNX model
            opset=11,  # ONNX opset 11 for good compatibility
            half=False,  # Use FP32 for CPU compatibility
        )

        # Move to desired location if needed
        export_path = Path(export_path)
        if export_path != onnx_path:
            if onnx_path.exists():
                onnx_path.unlink()
            export_path.rename(onnx_path)

        # Verify ONNX file was created successfully
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export completed but file not found: {onnx_path}")

        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✅ ONNX model generated: {onnx_path.name} ({onnx_size:.1f} MB)")
        return onnx_path

    except Exception as e:
        raise RuntimeError(f"❌ Failed to generate ONNX model: {e}") from e


def get_model_path():
    """Find available model files and let user choose."""
    # Search for all .pt files in runs/detect subdirectories
    available_models = []

    # Check if runs/detect exists
    runs_detect = Path("runs/multi-detect")
    if runs_detect.exists():
        # Find all subdirectories in runs/detect
        for run_dir in runs_detect.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / "weights"
                if weights_dir.exists():
                    # Look for best.pt and last.pt in each weights directory
                    for model_file in ["best.pt", "last.pt"]:
                        model_path = weights_dir / model_file
                        if model_path.exists():
                            available_models.append(model_path)

    # Also search in models directory and current directory
    for pattern in ["models/*.pt", "*.pt"]:
        for path in Path(".").glob(pattern):
            if path.is_file():
                available_models.append(path)

    # Remove duplicates and sort
    available_models = sorted(set(available_models))

    if not available_models:
        return None

    if len(available_models) == 1:
        return available_models[0]

    # Multiple models found - let user choose
    print(f"\n📦 Found {len(available_models)} model files:")
    for i, model_path in enumerate(available_models, 1):
        size_mb = model_path.stat().st_size / (1024 * 1024)
        mod_time = model_path.stat().st_mtime
        from datetime import datetime

        mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        print(f"   {i}. {model_path} ({size_mb:.1f} MB, modified: {mod_date})")

    while True:
        try:
            choice = input(
                f"\n🎯 Select model to upload (1-{len(available_models)}): "
            ).strip()
            if not choice:
                continue

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                return available_models[choice_idx]
            else:
                print(f"   Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("   Please enter a valid number")
        except KeyboardInterrupt:
            print("\n❌ Selection cancelled")
            return None


def collect_run_assets(model_path):
    """Collect all assets from the training run directory, including only the selected model."""
    # Get the run directory from the model path
    # Model path should be like: runs/detect/run_name/weights/best.pt
    if "runs/detect" in str(model_path):
        run_dir = model_path.parent.parent  # Go up from weights/ to run directory
    else:
        # If not from a training run, just return the model file itself
        assets = [model_path]
        # Generate ONNX model (will raise exception if it fails)
        onnx_path = generate_onnx_model(model_path)
        assets.append(onnx_path)
        return assets

    if not run_dir.exists():
        assets = [model_path]
        # Generate ONNX model (will raise exception if it fails)
        onnx_path = generate_onnx_model(model_path)
        assets.append(onnx_path)
        return assets

    # Define file patterns to include in the release (excluding other .pt files)
    include_patterns = [
        "*.png",  # All plots and visualizations
        "*.jpg",  # Training/validation batch images
        "*.yaml",  # Training arguments
        "*.csv",  # Results data
    ]

    assets = []
    for pattern in include_patterns:
        for file_path in run_dir.glob(pattern):
            if file_path.is_file():
                assets.append(file_path)

    # Add only the selected model file
    assets.append(model_path)

    # Generate ONNX model (will raise exception if it fails)
    onnx_path = generate_onnx_model(model_path, run_dir)
    assets.append(onnx_path)

    return sorted(assets)


def get_training_info(model_path):
    """Extract training information from args.yaml and dataset configuration."""
    # Get the run directory from the model path
    if "runs/detect" in str(model_path):
        run_dir = model_path.parent.parent  # Go up from weights/ to run directory
        args_file = run_dir / "args.yaml"
    else:
        return {"is_debug": False, "train_images": "~5,990", "val_images": "~5,794"}

    if not args_file.exists():
        return {"is_debug": False, "train_images": "~5,990", "val_images": "~5,794"}

    try:
        with open(args_file) as f:
            args = yaml.safe_load(f)

        # Check if this is a debug run by looking at the data path
        data_path = args.get("data", "")
        is_debug = "debug" in data_path.lower()

        # Try to read the actual dataset configuration
        if data_path and Path(data_path).exists():
            with open(data_path) as f:
                dataset_config = yaml.safe_load(f)

            # Count actual images if possible
            train_images = "unknown"
            val_images = "unknown"

            # Look for train and val paths in dataset config
            train_path = dataset_config.get("train", "")
            val_path = dataset_config.get("val", "")

            if train_path and Path(train_path).exists():
                train_labels = list(Path(train_path).glob("*.txt"))
                train_images = str(len(train_labels))

            if val_path and Path(val_path).exists():
                val_labels = list(Path(val_path).glob("*.txt"))
                val_images = str(len(val_labels))

            # If we couldn't count, use estimates based on debug status
            if train_images == "unknown":
                train_images = "~599" if is_debug else "~5,990"
            if val_images == "unknown":
                val_images = "~579" if is_debug else "~5,794"

        else:
            # Fallback estimates
            train_images = "~599" if is_debug else "~5,990"
            val_images = "~579" if is_debug else "~5,794"

        return {
            "is_debug": is_debug,
            "train_images": train_images,
            "val_images": val_images,
            "epochs": args.get("epochs", "unknown"),
            "data_path": data_path,
        }

    except Exception as e:
        print(f"⚠️ Could not parse training config: {e}")
        return {"is_debug": False, "train_images": "~5,990", "val_images": "~5,794"}


def create_release(version, model_path):
    """Create a GitHub release with the model and training assets."""
    print(f"🚀 Creating release {version}...")

    # Ensure version starts with 'bird-head-detector-v'
    if not version.startswith("bird-head-detector-"):
        if version.startswith("v"):
            version = f"bird-head-detector-{version}"
        else:
            version = f"bird-head-detector-v{version}"

    # Create git tag
    print(f"📝 Creating git tag: {version}")
    stdout, stderr = run_command(f'git tag -a {version} -m "Release {version}"')
    if stderr and "already exists" in stderr:
        print(f"❌ Tag {version} already exists!")
        return False

    # Push tag to remote
    print("📤 Pushing tag to remote...")
    stdout, stderr = run_command(f"git push origin {version}")

    # Collect all training run assets
    assets = collect_run_assets(model_path)

    # Get training information from args.yaml
    training_info = get_training_info(model_path)

    # Create release notes with asset list
    model_files = []
    onnx_files = []
    plot_files = []
    data_files = []

    for asset in assets:
        if asset.suffix == ".pt":
            model_files.append(asset.name)
        elif asset.suffix == ".onnx":
            onnx_files.append(asset.name)
        elif asset.suffix in [".png", ".jpg"]:
            plot_files.append(asset.name)
        elif asset.suffix in [".csv", ".yaml"]:
            data_files.append(asset.name)

    # Build file list for release notes
    files_section = "## Files\n"
    if model_files:
        files_section += "### Model Weights\n"
        files_section += "- `bird-head-detector.pt`: Trained model weights (PyTorch)\n"

    if onnx_files:
        files_section += (
            "- `bird-head-detector.onnx`: Trained model weights (ONNX format)\n"
        )

    if plot_files:
        files_section += "\n### Training Visualizations\n"
        for f in plot_files:
            files_section += f"- `{f}`: Training plots and visualizations\n"

    if data_files:
        files_section += "\n### Training Data\n"
        for f in data_files:
            files_section += f"- `{f}`: Training configuration and results\n"

    # Build model details section with dynamic training info
    debug_note = " (Debug Training)" if training_info["is_debug"] else ""
    epochs_info = (
        f"- **Epochs**: {training_info['epochs']}\n"
        if training_info.get("epochs") != "unknown"
        else ""
    )

    # Create release with all assets
    print("🎁 Creating GitHub release...")
    release_title = f"Bird Head Detector Model {version}{debug_note}"
    release_notes = f"""This release includes a trained YOLOv8n model for bird head detection with complete training artifacts.

## Model Details
- **Architecture**: YOLOv8n
- **Dataset**: CUB-200-2011 (bird head parts)
{epochs_info}- **Training Images**: {training_info["train_images"]}
- **Validation Images**: {training_info["val_images"]}
- **Classes**: 1 (bird_head)

{files_section}
"""

    # Write release notes to temporary file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(release_notes)
        notes_file = f.name

    try:
        # Create the release (without assets first)
        cmd = f'gh release create {version} --title "{release_title}" --notes-file "{notes_file}"'
        print(f"🔧 Running: {cmd}")
        stdout, stderr = run_command(cmd)

        if stderr and "already exists" in stderr:
            print(f"❌ Release {version} already exists!")
            # Clean up the tag we just created
            run_command(f"git tag -d {version}", check=False)
            run_command(f"git push origin --delete {version}", check=False)
            return False

        print("✅ Release created successfully!")
    finally:
        # Clean up temporary file
        try:
            os.unlink(notes_file)
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete temporary file {notes_file}: {e}")

    # Upload assets to the release
    if assets:
        print(f"📦 Uploading {len(assets)} assets to release...")

        # Create temporary directory for renamed files
        with tempfile.TemporaryDirectory() as temp_upload_dir:
            temp_path = Path(temp_upload_dir)

            # Prepare asset paths, creating renamed copies for model files
            upload_files = []
            for asset in assets:
                if asset.suffix == ".pt":
                    # Create renamed copy of PT file
                    renamed_pt = temp_path / "bird-head-detector.pt"
                    renamed_pt.write_bytes(asset.read_bytes())
                    upload_files.append(str(renamed_pt))
                    print(f"   📝 Renaming {asset.name} → bird-head-detector.pt")
                elif asset.suffix == ".onnx":
                    # Create renamed copy of ONNX file
                    renamed_onnx = temp_path / "bird-head-detector.onnx"
                    renamed_onnx.write_bytes(asset.read_bytes())
                    upload_files.append(str(renamed_onnx))
                    print(f"   📝 Renaming {asset.name} → bird-head-detector.onnx")
                else:
                    upload_files.append(str(asset))

            # Upload all files
            files_str = " ".join(f'"{f}"' for f in upload_files)
            upload_cmd = f"gh release upload {version} {files_str}"
            print("🔧 Running upload command...")
            stdout, stderr = run_command(
                upload_cmd, timeout=300
            )  # 5 minute timeout for uploads

            if stderr and "timed out" in stderr:
                print(
                    "❌ Upload timed out. Try uploading fewer assets or check network connection."
                )
                return False
            elif stderr:
                print(f"⚠️ Warning during asset upload: {stderr}")

        print(f"✅ Uploaded {len(assets)} assets:")
        for asset in assets:
            size_mb = asset.stat().st_size / (1024 * 1024)
            if asset.suffix == ".pt":
                print(f"   - bird-head-detector.pt ({size_mb:.1f} MB)")
            elif asset.suffix == ".onnx":
                print(f"   - bird-head-detector.onnx ({size_mb:.1f} MB)")
            else:
                print(f"   - {asset.name} ({size_mb:.1f} MB)")
    else:
        print("ℹ️ No additional assets to upload")

    print(f"✅ Release {version} created successfully!")
    print(f"🔗 View at: https://github.com/{get_repo_info()}/releases/tag/{version}")

    return True


def get_repo_info():
    """Get repository owner/name from remote origin."""
    stdout, stderr = run_command("git remote get-url origin")
    if "github.com" in stdout:
        # Extract owner/repo from URL
        if stdout.startswith("https://"):
            # https://github.com/owner/repo.git
            parts = (
                stdout.replace("https://github.com/", "").replace(".git", "").split("/")
            )
        else:
            # git@github.com:owner/repo.git
            parts = stdout.replace("git@github.com:", "").replace(".git", "").split("/")

        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

    return "unknown/unknown"


def main():
    """Main release process."""
    parser = argparse.ArgumentParser(
        description="Release bird head detector models to GitHub"
    )
    parser.add_argument(
        "--model", type=str, help="Path to specific model file to upload"
    )
    parser.add_argument(
        "--version", type=str, help="Version number for the release (e.g., 1.0.0)"
    )
    args = parser.parse_args()

    print("🚀 Bird Head Detector Release Script")
    print("=" * 40)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Check if repo is clean
    if not check_repo_clean():
        sys.exit(1)

    # Find or use specified model file
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Specified model not found: {model_path}")
            sys.exit(1)
        if not model_path.suffix == ".pt":
            print(f"❌ Model file must have .pt extension: {model_path}")
            sys.exit(1)
        print(f"✅ Using specified model: {model_path}")
    else:
        model_path = get_model_path()
        if not model_path:
            print("❌ No trained models found!")
            print("   Searched in:")
            print("   - runs/detect/*/weights/*.pt")
            print("   - models/*.pt")
            print("   - *.pt")
            print("\n   Train a model first using: uv run python train.py")
            print("   Or specify a model with: --model path/to/model.pt")
            sys.exit(1)

    print(f"✅ Found model: {model_path}")
    model_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Size: {model_size:.1f} MB")

    # Test ONNX generation early to fail fast
    print("\n🔍 Validating ONNX export capability...")
    try:
        # Test with a temporary directory to avoid cluttering
        with tempfile.TemporaryDirectory() as temp_dir:
            test_onnx_path = generate_onnx_model(model_path, temp_dir)
            print(f"✅ ONNX export test successful: {test_onnx_path.name}")
    except Exception as e:
        print(f"❌ ONNX export validation failed: {e}")
        print("\n💡 This could be due to:")
        print("   - Missing ONNX dependencies (run: uv sync --extra onnx)")
        print("   - Incompatible model format")
        print("   - System/environment issues")
        sys.exit(1)

    # Show what assets will be uploaded
    try:
        assets = collect_run_assets(model_path)
    except Exception as e:
        print(f"❌ Failed to collect release assets: {e}")
        sys.exit(1)
    if assets:
        print(f"\n📦 Assets to upload ({len(assets)} files):")
        total_size = 0
        for asset in assets:
            size_mb = asset.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   - {asset.name} ({size_mb:.1f} MB)")
        print(f"   Total size: {total_size:.1f} MB")
    else:
        print("\n📦 Only model file will be uploaded (no training run detected)")

    # Show existing tags
    existing_tags = get_existing_tags()
    if existing_tags:
        print("\n📋 Existing versions:")
        for tag in existing_tags[:10]:  # Show last 10 tags
            print(f"   {tag}")
        if len(existing_tags) > 10:
            print(f"   ... and {len(existing_tags) - 10} more")
    else:
        print("\n📋 No existing versions found")

    # Get version from user or command line
    if args.version:
        version = args.version
        if not validate_version(version):
            print(f"❌ Invalid version format: {version}")
            print("   Use semantic versioning (e.g., 1.0.0)")
            sys.exit(1)

        # Normalize version (add 'bird-head-detector-v' prefix if missing)
        if version.startswith("bird-head-detector-"):
            normalized_version = version
        elif version.startswith("v"):
            normalized_version = f"bird-head-detector-{version}"
        else:
            normalized_version = f"bird-head-detector-v{version}"

        if normalized_version in existing_tags:
            print(f"❌ Version {normalized_version} already exists!")
            sys.exit(1)
    else:
        print("\n🏷️  Enter new version number:")
        while True:
            version = input("   Version (e.g., 1.0.0 or v1.0.0): ").strip()

            if not version:
                print("   Version cannot be empty!")
                continue

            if not validate_version(version):
                print(
                    "   Invalid version format! Use semantic versioning (e.g., 1.0.0)"
                )
                continue

            # Normalize version (add 'bird-head-detector-v' prefix if missing)
            if version.startswith("bird-head-detector-"):
                normalized_version = version
            elif version.startswith("v"):
                normalized_version = f"bird-head-detector-{version}"
            else:
                normalized_version = f"bird-head-detector-v{version}"

            if normalized_version in existing_tags:
                print(f"   Version {normalized_version} already exists!")
                continue

            break

    # Confirm release
    print("\n📋 Release Summary:")
    print(f"   Version: {normalized_version}")
    print(f"   Model: {model_path}")
    print(f"   Size: {model_size:.1f} MB")
    print(f"   Repository: {get_repo_info()}")

    confirm = (
        input(f"\n❓ Create release {normalized_version}? (y/N): ").strip().lower()
    )
    if confirm not in ["y", "yes"]:
        print("❌ Release cancelled")
        sys.exit(0)

    # Create the release
    if create_release(version, model_path):
        print(f"\n🎉 Release {normalized_version} completed successfully!")
    else:
        print("\n❌ Release failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
