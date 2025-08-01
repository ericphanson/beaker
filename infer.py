#!/usr/bin/env python3
"""
Run inference with trained YOLOv8n bird head detection model.
Automatically downloads model weights from GitHub releases if not found locally.
"""

import argparse
import subprocess
import urllib.request
import json
from pathlib import Path
from ultralytics import YOLO


def get_repo_info():
    """Get repository owner/name from remote origin."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], 
            capture_output=True, text=True, check=True
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
    
    return "ericphanson/bird-head-detector"  # Fallback default


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
        for asset in release_data.get('assets', []):
            if asset['name'].endswith('.pt'):
                model_asset = asset
                break
        
        if not model_asset:
            print("❌ No model file (.pt) found in latest release")
            return None
        
        # Create models directory
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / model_asset['name']
        
        # Download if not already exists
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path
        
        print(f"📥 Downloading {model_asset['name']} ({model_asset['size'] / (1024*1024):.1f} MB)...")
        print(f"   From: {release_data['tag_name']} - {release_data['name']}")
        
        # Download the model
        urllib.request.urlretrieve(model_asset['browser_download_url'], model_path)
        
        print(f"✅ Downloaded model: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Failed to download model from releases: {e}")
        return None


def find_or_download_model(model_path_arg):
    """Find model locally or download from GitHub releases."""
    model_path = Path(model_path_arg)
    
    # If explicit path provided and exists, use it
    if model_path_arg != 'runs/detect/bird_head_yolov8n/weights/best.pt' and model_path.exists():
        return model_path
    
    # Check local training outputs
    local_paths = [
        Path("runs/detect/bird_head_yolov8n/weights/best.pt"),
        Path("runs/detect/bird_head_yolov8n_debug/weights/best.pt"),
    ]
    
    for path in local_paths:
        if path.exists():
            print(f"✅ Found local model: {path}")
            return path
    
    # Check downloaded models directory
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            print(f"✅ Found downloaded model: {model_file}")
            return model_file
    
    # Try to download from releases
    print("🌐 No local model found, checking GitHub releases...")
    downloaded_model = download_latest_model(models_dir)
    if downloaded_model:
        return downloaded_model
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Bird head detection inference")
    parser.add_argument(
        "--model", 
        type=str, 
        default="runs/detect/bird_head_yolov8n/weights/best.pt",
        help="Path to model weights (will auto-download from releases if not found)"
    )
    parser.add_argument("--source", type=str, required=True, help="Source image/video/directory")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    parser.add_argument("--show", action="store_true", help="Show detection results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
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
    results = model(
        args.source,
        save=args.save,
        show=args.show,
        conf=args.conf,
        device="mps"  # Use MPS on Mac
    )
    
    # Print summary
    total_detections = sum(len(result.boxes) if result.boxes is not None else 0 for result in results)
    print(f"✅ Completed! Found {total_detections} bird head detections")
    
    if args.save:
        print(f"💾 Results saved to: runs/detect/predict/")


if __name__ == "__main__":
    main()
