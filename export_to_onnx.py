#!/usr/bin/env python3
"""
Export bird-head-detector models to ONNX format.

This script can:
1. Download a model from a GitHub release using a git tag
2. Use a local .pt file directly
3. Convert the model to ONNX format using Ultralytics
4. Save the ONNX model in the models/ directory

Requirements:
- ultralytics package installed
- ONNX dependencies (install with: uv sync --extra onnx)
- gh CLI tool (for downloading from GitHub releases)
- Internet connection (for GitHub downloads)

Usage:
    # Export from GitHub release tag
    uv run python export_to_onnx.py --tag v1.0.0

    # Export from local file
    uv run python export_to_onnx.py --model path/to/model.pt

    # Export with custom output name
    uv run python export_to_onnx.py --model model.pt --name custom_model

    # Export with specific image size
    uv run python export_to_onnx.py --tag bird-head-detector-v1.0.0 --imgsz 640
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics package not found.")
    print("   Install with: uv add ultralytics")
    print("   Or install with ONNX dependencies: uv sync --extra onnx")
    sys.exit(1)


def run_command(cmd, capture=True, check=True, timeout=120):
    """Run a shell command and return the result."""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=check, timeout=timeout
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
            return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
        raise


def check_gh_cli():
    """Check if GitHub CLI is available and authenticated."""
    # Check if gh CLI is installed
    stdout, stderr = run_command("gh --version", check=False)
    if not stdout:
        print("❌ GitHub CLI (gh) is not installed.")
        print("   Install it from: https://cli.github.com/")
        return False

    # Check if gh is authenticated
    stdout, stderr = run_command("gh auth status", check=False)
    if "Logged in to github.com" not in stderr and "Logged in to github.com" not in stdout:
        print("❌ GitHub CLI is not authenticated.")
        print("   Run: gh auth login")
        return False

    return True


def get_repo_info():
    """Get repository owner/name from remote origin."""
    stdout, stderr = run_command("git remote get-url origin", check=False)
    if not stdout or "github.com" not in stdout:
        # Fallback to hardcoded repo info
        return "ericphanson/beaker"

    # Extract owner/repo from URL
    if stdout.startswith("https://"):
        # https://github.com/owner/repo.git
        parts = stdout.replace("https://github.com/", "").replace(".git", "").split("/")
    else:
        # git@github.com:owner/repo.git
        parts = stdout.replace("git@github.com:", "").replace(".git", "").split("/")

    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"

    return "ericphanson/beaker"


def download_model_from_release(tag, target_dir):
    """Download model file from GitHub release."""
    print(f"📥 Downloading model from release {tag}...")

    if not check_gh_cli():
        return None

    repo = get_repo_info()
    print(f"🔍 Looking for release {tag} in {repo}")

    # List release assets
    cmd = f"gh release view {tag} --repo {repo} --json assets"
    stdout, stderr = run_command(cmd, check=False)

    if not stdout:
        print(f"❌ Could not find release {tag}")
        if stderr:
            print(f"   Error: {stderr}")
        return None

    try:
        import json

        release_data = json.loads(stdout)
        assets = release_data.get("assets", [])
    except json.JSONDecodeError:
        print("❌ Could not parse release data")
        return None

    # Find the model file (.pt extension)
    model_assets = [asset for asset in assets if asset["name"].endswith(".pt")]

    if not model_assets:
        print(f"❌ No .pt model files found in release {tag}")
        available_assets = [asset["name"] for asset in assets]
        if available_assets:
            print(f"   Available assets: {', '.join(available_assets)}")
        return None

    if len(model_assets) > 1:
        print(f"📦 Found {len(model_assets)} model files:")
        for i, asset in enumerate(model_assets, 1):
            size_mb = asset["size"] / (1024 * 1024)
            print(f"   {i}. {asset['name']} ({size_mb:.1f} MB)")

        while True:
            try:
                choice = input(f"\n🎯 Select model to download (1-{len(model_assets)}): ").strip()
                if not choice:
                    continue

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_assets):
                    selected_asset = model_assets[choice_idx]
                    break
                else:
                    print(f"   Please enter a number between 1 and {len(model_assets)}")
            except ValueError:
                print("   Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled")
                return None
    else:
        selected_asset = model_assets[0]

    asset_name = selected_asset["name"]
    print(f"📦 Downloading {asset_name}...")

    # Download the asset
    target_path = target_dir / asset_name
    cmd = f'gh release download {tag} --repo {repo} --pattern "{asset_name}" --dir "{target_dir}"'
    stdout, stderr = run_command(cmd, timeout=300)  # 5 minute timeout

    if not target_path.exists():
        print("❌ Download failed")
        if stderr:
            print(f"   Error: {stderr}")
        return None

    print(f"✅ Downloaded {asset_name} ({target_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return target_path


def export_to_onnx(model_path, output_name, imgsz=640):
    """Export model to ONNX format using Ultralytics."""
    print(f"🔄 Loading model from {model_path}...")

    try:
        # Load the model
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully")

        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Export to ONNX
        output_path = models_dir / f"{output_name}.onnx"
        print(f"🚀 Exporting to ONNX format (image size: {imgsz})...")
        print(f"   Output: {output_path}")

        # Export with specified image size
        export_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,  # Static input shape for better compatibility
            simplify=True,  # Simplify the ONNX model
        )

        # Move the exported file to our desired location with the correct name
        export_path = Path(export_path)
        if export_path != output_path:
            if output_path.exists():
                output_path.unlink()  # Remove existing file
            export_path.rename(output_path)

        print("✅ ONNX export completed!")
        print(f"   📁 Saved as: {output_path}")
        print(f"   📏 Size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")

        # Print model info
        print("\n📊 Model Information:")
        print(f"   Input shape: [1, 3, {imgsz}, {imgsz}]")
        print("   Format: ONNX")
        print(f"   Classes: {len(model.names)} ({', '.join(model.names.values())})")

        return output_path

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None


def main():
    """Main export process."""
    parser = argparse.ArgumentParser(description="Export bird head detector models to ONNX format")
    parser.add_argument("--tag", type=str, help="Git tag to download model from GitHub release")
    parser.add_argument("--model", type=str, help="Path to local .pt model file")
    parser.add_argument("--name", type=str, help="Output name for ONNX model (without extension)")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size for ONNX model (default: 640)"
    )

    args = parser.parse_args()

    print("🔄 Bird Head Detector ONNX Export")
    print("=" * 40)

    # Validate arguments
    if not args.tag and not args.model:
        print("❌ Either --tag or --model must be specified")
        print("   Use --tag to download from GitHub release")
        print("   Use --model to use local .pt file")
        sys.exit(1)

    if args.tag and args.model:
        print("❌ Cannot specify both --tag and --model")
        print("   Use either --tag OR --model, not both")
        sys.exit(1)

    # Determine model path and output name
    model_path = None
    output_name = args.name

    if args.tag:
        # Download from GitHub release
        if not output_name:
            # Use full tag as output name
            output_name = args.tag

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = download_model_from_release(args.tag, temp_path)
            if not model_path:
                sys.exit(1)

            # Export to ONNX
            result = export_to_onnx(model_path, output_name, args.imgsz)
            if not result:
                sys.exit(1)

    else:
        # Use local model file
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)

        if not model_path.suffix == ".pt":
            print(f"❌ Model file must have .pt extension: {model_path}")
            sys.exit(1)

        if not output_name:
            # Use model filename (without extension) as output name
            output_name = model_path.stem

        print(f"✅ Using local model: {model_path}")

        # Export to ONNX
        result = export_to_onnx(model_path, output_name, args.imgsz)
        if not result:
            sys.exit(1)

    print("\n🎉 Export completed successfully!")
    print("📁 ONNX model saved in models/ directory")


if __name__ == "__main__":
    main()
