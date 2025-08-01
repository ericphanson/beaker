#!/usr/bin/env python3
"""
Run inference with trained YOLOv8n bird head detection model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Run bird head detection inference')
    parser.add_argument('--model', type=str,
                       default='runs/detect/bird_head_yolov8n/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, video, or directory')
    parser.add_argument('--save', action='store_true',
                       help='Save results to runs/detect/predict/')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Train a model first using: uv run python train.py")
        return

    # Load trained model
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)

    # Run inference
    print(f"🔍 Running inference on: {args.source}")
    results = model(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show
    )

    # Print results summary
    if results:
        print(f"✅ Inference completed!")
        if args.save:
            print(f"💾 Results saved to: runs/detect/predict/")

        # Print detection count for each image
        for i, result in enumerate(results):
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"🐦 Image {i+1}: Found {detections} bird head(s)")


if __name__ == "__main__":
    main()
