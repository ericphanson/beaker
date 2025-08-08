#!/usr/bin/env python3
"""
Test script to demonstrate the new Beaker CLI-based model comparisons.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantizations.comparisons import generate_model_comparison_images


def main():
    """Test the new comparison functionality."""

    # Example paths (adjust these to your actual setup)
    models = {
        "Original": Path("output/quantized/head/head-optimized.onnx"),
        "Dynamic-Int8": Path("output/quantized/head/head-dynamic-int8.onnx"),
    }

    test_images = [
        Path("examples/example.jpg"),
        Path("examples/example.png"),  # Add more test images as needed
    ]

    output_dir = Path("output/comparisons/head")
    beaker_cargo_path = Path("../beaker/Cargo.toml")

    print("Testing Beaker CLI-based model comparisons...")
    print(f"Models: {list(models.keys())}")
    print(f"Test images: {[img.name for img in test_images if img.exists()]}")
    print(f"Output directory: {output_dir}")
    print(f"Beaker Cargo.toml: {beaker_cargo_path}")
    print()

    # Filter to only existing files
    existing_models = {name: path for name, path in models.items() if path.exists()}
    existing_images = [img for img in test_images if img.exists()]

    if not existing_models:
        print("❌ No model files found. Please run quantization first.")
        return

    if not existing_images:
        print("❌ No test images found. Please add some test images to examples/")
        return

    if not beaker_cargo_path.exists():
        print(
            "❌ Beaker Cargo.toml not found. Please adjust the path or clone the beaker repo."
        )
        return

    print(f"Found {len(existing_models)} models and {len(existing_images)} test images")
    print("Running comparisons...")

    try:
        comparison_files = generate_model_comparison_images(
            existing_models,
            existing_images,
            output_dir,
            beaker_cargo_path=beaker_cargo_path,
        )

        print(f"✅ Generated {len(comparison_files)} comparison files:")
        for file_path in comparison_files:
            print(f"  - {file_path}")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
