#!/usr/bin/env python3
"""
Run inference with trained YOLOv8n bird head detection model.
Automatically downloads model weights from GitHub releases if not found locally.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
from platformdirs import user_cache_dir
from ultralytics import YOLO

# Configure UTF-8 encoding to handle Unicode emoji characters on all platforms
os.environ["PYTHONIOENCODING"] = "utf-8"
# Reconfigure stdout/stderr to use UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def get_cache_dir():
    """Get the cache directory for bird-head-detector models."""
    cache_dir = Path(user_cache_dir("bird-head-detector", "ericphanson"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_repo_info():
    """Get repository owner/name from remote origin or use default."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            cwd=".",
        )
        stdout = result.stdout.strip()

        if "github.com" in stdout:
            # Extract owner/repo from URL
            if stdout.startswith("https://"):
                # https://github.com/owner/repo.git
                parts = stdout.replace("https://github.com/", "").replace(".git", "").split("/")
            else:
                # git@github.com:owner/repo.git
                parts = stdout.replace("git@github.com:", "").replace(".git", "").split("/")

            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    except:
        pass

    return "ericphanson/beaker"  # Fallback default


def download_latest_model(models_dir):
    """Download the latest model from GitHub releases."""
    repo = get_repo_info()
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    try:
        print(f"🔍 Checking for latest release from {repo}...")

        # Get latest release info
        with urllib.request.urlopen(api_url) as response:
            release_data = json.loads(response.read())

        # Find .pt model file in assets
        model_asset = None
        for asset in release_data.get("assets", []):
            if asset["name"] == "bird-head-detector.pt":
                model_asset = asset
                break
            elif asset["name"].endswith(".pt"):
                # Fallback to any .pt file if bird-head-detector.pt not found
                model_asset = asset

        if not model_asset:
            print("❌ No model file (.pt) found in latest release")
            return None

        # Create models directory
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / model_asset["name"]

        # Download if not already exists
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path

        print(
            f"📥 Downloading {model_asset['name']} ({model_asset['size'] / (1024 * 1024):.1f} MB)..."
        )
        print(f"   From: {release_data['tag_name']} - {release_data['name']}")

        # Download the model
        urllib.request.urlretrieve(model_asset["browser_download_url"], model_path)

        print(f"✅ Downloaded model: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Failed to download model from releases: {e}")
        return None


def find_or_download_model(model_path_arg):
    """Find model locally or download from GitHub releases."""
    model_path = Path(model_path_arg)

    # If explicit path provided and exists, use it
    if model_path_arg != "runs/detect/bird_head_yolov8n/weights/best.pt" and model_path.exists():
        return model_path

    # Check local training outputs (for development)
    local_paths = [
        Path("runs/detect/bird_head_yolov8n/weights/best.pt"),
        Path("runs/detect/bird_head_yolov8n_debug/weights/best.pt"),
    ]

    for path in local_paths:
        if path.exists():
            print(f"✅ Found local model: {path}")
            return path

    # Use cache directory for downloaded models
    cache_dir = get_cache_dir()
    models_dir = cache_dir / "models"

    # Check cached models directory
    if models_dir.exists():
        # First look for the standardized name
        standard_model = models_dir / "bird-head-detector.pt"
        if standard_model.exists():
            print(f"✅ Found cached model: {standard_model}")
            return standard_model

        # Then look for any other .pt file
        for model_file in models_dir.glob("*.pt"):
            print(f"✅ Found cached model: {model_file}")
            return model_file

    # Try to download from releases
    print("🌐 No local model found, checking GitHub releases...")
    downloaded_model = download_latest_model(models_dir)
    if downloaded_model:
        return downloaded_model

    return None


def create_square_crop(image_path, bbox, output_dir=None, padding=0.25):
    """Create a square crop around the detection bounding box."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Expand bounding box by specified padding
    width = x2 - x1
    height = y2 - y1
    expand_w = width * padding
    expand_h = height * padding

    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(w, x2 + expand_w)
    y2 = min(h, y2 + expand_h)

    # Calculate center and current dimensions after expansion
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_width = x2 - x1
    new_height = y2 - y1

    # Make square by using the larger dimension
    size = max(new_width, new_height)
    half_size = size / 2

    # Calculate square bounds
    crop_x1 = max(0, int(center_x - half_size))
    crop_y1 = max(0, int(center_y - half_size))
    crop_x2 = min(w, int(center_x + half_size))
    crop_y2 = min(h, int(center_y + half_size))

    # Adjust if we hit image boundaries to maintain square
    actual_width = crop_x2 - crop_x1
    actual_height = crop_y2 - crop_y1

    if actual_width < actual_height:
        # Adjust width
        diff = actual_height - actual_width
        crop_x1 = max(0, crop_x1 - diff // 2)
        crop_x2 = min(w, crop_x2 + diff // 2)
    elif actual_height < actual_width:
        # Adjust height
        diff = actual_width - actual_height
        crop_y1 = max(0, crop_y1 - diff // 2)
        crop_y2 = min(h, crop_y2 + diff // 2)

    # Crop the image
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Determine output path based on whether output_dir is specified
    input_path = Path(image_path)
    input_name = input_path.stem
    input_ext = input_path.suffix

    if output_dir:
        # If output_dir specified, use identical name to input in that directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
    else:
        # Default: place next to input with -crop suffix
        output_path = input_path.parent / f"{input_name}-crop{input_ext}"

    cv2.imwrite(str(output_path), cropped)
    return output_path


def detect_device(requested_device="auto"):
    """Detect the best available device for inference or use user's choice."""
    # If user specified a specific device, try to use it
    if requested_device != "auto":
        if requested_device == "cpu":
            return "cpu"
        elif requested_device == "mps":
            try:
                import torch

                if torch.backends.mps.is_available():
                    # Test if MPS actually works
                    test_tensor = torch.zeros(1, device="mps")
                    del test_tensor
                    return "mps"
                else:
                    print("⚠️  MPS not available on this system, falling back to CPU")
                    return "cpu"
            except (ImportError, RuntimeError) as e:
                print(f"⚠️  Cannot use MPS ({e}), falling back to CPU")
                return "cpu"
        elif requested_device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                else:
                    print("⚠️  CUDA not available on this system, falling back to CPU")
                    return "cpu"
            except ImportError as e:
                print(f"⚠️  Cannot use CUDA ({e}), falling back to CPU")
                return "cpu"

    # Auto-detection logic
    try:
        import torch

        if torch.backends.mps.is_available():
            # Test if MPS actually works by trying a small operation
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                return "mps"  # Apple Silicon Mac with working MPS
            except RuntimeError as e:
                if "MPS backend out of memory" in str(e):
                    print("⚠️  MPS out of memory, falling back to CPU")
                    return "cpu"
                else:
                    print(f"⚠️  MPS test failed ({e}), falling back to CPU")
                    return "cpu"
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"  # Fallback to CPU
    except ImportError:
        # If PyTorch isn't available, let YOLO handle device selection
        return None


def save_bounding_box_image(image_path, bbox, output_dir=None):
    """Save image with bounding box drawn on it."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Draw bounding box
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Determine output path based on whether output_dir is specified
    input_path = Path(image_path)
    input_name = input_path.stem
    input_ext = input_path.suffix

    if output_dir:
        # If output_dir specified, use identical name to input in that directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
    else:
        # Default: place next to input with -bounding-box suffix
        output_path = input_path.parent / f"{input_name}-bounding-box{input_ext}"

    cv2.imwrite(str(output_path), img)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Bird head detection inference")
    parser.add_argument("source", type=str, help="Source image or directory")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/bird_head_yolov8n/weights/best.pt",
        help="Path to model weights (will auto-download from releases if not found)",
    )
    parser.add_argument(
        "--save-bounding-box",
        action="store_true",
        help="Save detection results with bounding boxes",
    )
    parser.add_argument("--show", action="store_true", help="Show detection results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--skip-crop", action="store_true", help="Skip creating square crops around detected heads"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save outputs (default: next to input)"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Padding around bounding box as fraction (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference (default: auto-detect)",
    )

    args = parser.parse_args()

    # Find or download model
    model_path = find_or_download_model(args.model)
    if model_path is None:
        print("\n❌ Error: No model found!")
        print("Options:")
        print("1. Train a model first: uv run python train.py")
        print("2. Specify a model path: --model path/to/model.pt")
        print("3. Ensure GitHub releases contain .pt model files")
        return

    print(f"\n🤖 Loading model: {model_path}")

    # Load model
    model = YOLO(model_path)

    # Run inference
    print(f"🔍 Running inference on: {args.source}")
    device = detect_device(args.device)
    if device:
        if args.device == "auto":
            print(f"🚀 Auto-detected device: {device}")
        else:
            print(f"🚀 Using device: {device}")

    # Try inference with the detected device, fallback to CPU if MPS fails
    try:
        results = model(
            args.source,
            save=False,  # We handle saving manually
            show=args.show,
            conf=args.conf,
            device=device,  # Use detected/requested device
        )
    except RuntimeError as e:
        if device == "mps" and "MPS backend out of memory" in str(e):
            print("⚠️  MPS out of memory during inference, retrying with CPU...")
            device = "cpu"
            print(f"🚀 Retrying with device: {device}")
            results = model(
                args.source,
                save=False,  # We handle saving manually
                show=args.show,
                conf=args.conf,
                device=device,  # Fallback to CPU
            )
        else:
            # Re-raise if it's not an MPS memory error
            raise

    # Process results and create outputs
    total_detections = 0
    crops_created = 0
    bbox_images_created = 0
    output_dir = Path(args.output_dir) if args.output_dir else None

    for i, result in enumerate(results):
        if result.boxes is not None:
            detections = len(result.boxes)
            total_detections += detections

            # Process if at least one detection
            if detections >= 1:
                # Get the source image path
                source_path = Path(result.path) if hasattr(result, "path") else Path(args.source)

                # Get the highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()

                # Get bounding box for highest confidence detection (in xyxy format)
                bbox = result.boxes.xyxy[best_idx].cpu().numpy()
                best_conf = confidences[best_idx]

                # Create crop by default (unless skipped)
                if not args.skip_crop:
                    crop_path = create_square_crop(source_path, bbox, output_dir, args.padding)
                    if crop_path:
                        crops_created += 1
                        if detections == 1:
                            print(f"✂️  Created crop: {crop_path}")
                        else:
                            print(
                                f"✂️  Created crop: {crop_path} (used highest confidence: {best_conf:.3f}, {detections} total detections)"
                            )

                # Save bounding box image if requested
                if args.save_bounding_box:
                    bbox_path = save_bounding_box_image(source_path, bbox, output_dir)
                    if bbox_path:
                        bbox_images_created += 1
                        if detections == 1:
                            print(f"📦 Created bounding box image: {bbox_path}")
                        else:
                            print(
                                f"📦 Created bounding box image: {bbox_path} (used highest confidence: {best_conf:.3f}, {detections} total detections)"
                            )

            elif not args.skip_crop or args.save_bounding_box:
                # Only print no detections message if we would have created outputs
                print(f"ℹ️  No detections in {result.path}, no outputs created")

    # Print summary
    print(f"✅ Completed! Found {total_detections} bird head detections")

    if not args.skip_crop:
        if output_dir:
            print(f"✂️  Created {crops_created} square head crops in: {output_dir}")
        else:
            print(f"✂️  Created {crops_created} square head crops next to original images")

    if args.save_bounding_box:
        if output_dir:
            print(f"📦 Created {bbox_images_created} bounding box images in: {output_dir}")
        else:
            print(f"📦 Created {bbox_images_created} bounding box images next to original images")


if __name__ == "__main__":
    main()
