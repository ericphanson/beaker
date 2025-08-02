#!/usr/bin/env python3
"""
Export bird-head-detector models to ONNX format.

This script converts local PyTorch (.pt) model files to ONNX format for deployment.
ONNX models are automatically generated during releases - this script is for
custom exports or testing purposes.

Model Size Optimization:
- FP16 precision (half=True) reduces size by ~50% but only works on GPU exports
- Smaller input sizes (--imgsz) slightly reduce model size
- ONNX simplification is enabled by default
- Use --no-optimize to disable optimizations for maximum compatibility

Requirements:
- ultralytics package installed
- ONNX dependencies (install with: uv sync --extra onnx)

Usage:
    # Export from local file (optimized by default)
    uv run python export_to_onnx.py --model path/to/model.pt

    # Export with custom output name
    uv run python export_to_onnx.py --model model.pt --name custom_model

    # Export without optimization (larger file, FP32 precision)
    uv run python export_to_onnx.py --model model.pt --no-optimize

    # Export with specific image size and opset version
    uv run python export_to_onnx.py --model model.pt --imgsz 320 --opset 12
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics package not found.")
    print("   Install with: uv add ultralytics")
    print("   Or install with ONNX dependencies: uv sync --extra onnx")
    sys.exit(1)


def export_to_onnx(model_path, output_name, imgsz=640, optimize=True, opset=11):
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
        if optimize:
            print("   🔧 Optimization enabled (smaller file size)")
        print(f"   Output: {output_path}")

        # Export with specified image size and optimization settings
        # Note: FP16 (half precision) is only supported on GPU exports
        use_half = optimize and model.device.type == "cuda"

        export_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,  # Static input shape for better compatibility
            simplify=True,  # Simplify the ONNX model
            opset=opset,  # ONNX opset version
            half=use_half,  # Use FP16 precision only on GPU
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
        print(f"   Precision: {'FP16 (optimized)' if use_half else 'FP32 (full precision)'}")
        print(f"   ONNX Opset: {opset}")
        print(f"   Classes: {len(model.names)} ({', '.join(model.names.values())})")

        return output_path

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None


def main():
    """Main export process."""
    parser = argparse.ArgumentParser(description="Export bird head detector models to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to local .pt model file")
    parser.add_argument("--name", type=str, help="Output name for ONNX model (without extension)")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size for ONNX model (default: 640)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimizations (larger file, FP32 precision)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11, use 12+ for newer features)",
    )

    args = parser.parse_args()

    print("🔄 Bird Head Detector ONNX Export")
    print("=" * 40)

    # Use local model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    if not model_path.suffix == ".pt":
        print(f"❌ Model file must have .pt extension: {model_path}")
        sys.exit(1)

    output_name = args.name
    if not output_name:
        # Use model filename (without extension) as output name
        output_name = model_path.stem

    print(f"✅ Using local model: {model_path}")

    # Export to ONNX
    result = export_to_onnx(model_path, output_name, args.imgsz, not args.no_optimize, args.opset)
    if not result:
        sys.exit(1)

    print("\n🎉 Export completed successfully!")
    print("📁 ONNX model saved in models/ directory")


if __name__ == "__main__":
    main()
