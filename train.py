#!/usr/bin/env python3
"""
Train YOLOv8n model for bird head detection on M1 MacBook Pro with Comet.ml tracking.

Required Environment Variables for Comet.ml:
    COMET_API_KEY: Your Comet.ml API key (get from https://www.comet.ml/api/my/settings/)
    COMET_PROJECT_NAME: (Optional) Project name in Comet.ml (default: 'bird-head-detector')
    COMET_WORKSPACE: (Optional) Your Comet.ml workspace/username

Example:
    export COMET_API_KEY="your-api-key-here"
    export COMET_PROJECT_NAME="bird-head-detector"
    export COMET_WORKSPACE="your-username"
    uv run python train.py
"""

from ultralytics import YOLO
import torch
import os
import comet_ml


def setup_comet():
    """Setup Comet.ml experiment tracking."""
    api_key = os.getenv('COMET_API_KEY')
    if not api_key:
        print("⚠️  COMET_API_KEY not found. Comet.ml tracking will be disabled.")
        print("   Set your API key: export COMET_API_KEY='your-api-key'")
        print("   Get your API key from: https://www.comet.ml/api/my/settings/")
        return None

    project_name = os.getenv('COMET_PROJECT_NAME', 'bird-head-detector')
    workspace = os.getenv('COMET_WORKSPACE')

    try:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )

        # Log hyperparameters
        experiment.log_parameters({
            'model': 'yolov8n',
            'task': 'bird_head_detection',
            'dataset': 'CUB-200-2011',
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
            'architecture': 'YOLOv8n',
        })

        print(f"✅ Comet.ml experiment started: {experiment.url}")
        return experiment

    except Exception as e:
        print(f"❌ Failed to initialize Comet.ml: {e}")
        print("   Training will continue without experiment tracking.")
        return None


def main():
    # Setup Comet.ml tracking
    experiment = setup_comet()

    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        device = 'mps'
    else:
        print("❌ MPS not available, falling back to CPU")
        device = 'cpu'

    # Load a pretrained YOLOv8n model
    print("📦 Loading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')

    # Configure Comet.ml integration for YOLO
    if experiment:
        # YOLO will automatically detect and use the active Comet experiment
        os.environ['COMET_MODE'] = 'online'

    # Train the model on bird head dataset
    print("🚀 Starting training...")
    results = model.train(
        data='data/yolo/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust based on M1 memory
        device=device,  # Use Metal Performance Shaders or CPU
        project='runs/detect',
        name='bird_head_yolov8n',
        workers=0,  # Prevent multiprocessing issues on M1
        verbose=True
    )

    # Log final results to Comet.ml
    if experiment:
        try:
            # Log final metrics
            final_metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    experiment.log_metric(f"final_{key}", value)

            # Log model artifacts
            best_model_path = 'runs/detect/bird_head_yolov8n/weights/best.pt'
            if os.path.exists(best_model_path):
                experiment.log_model('best_model', best_model_path)
                print("📤 Model uploaded to Comet.ml")

            experiment.end()
            print("✅ Comet.ml experiment completed")

        except Exception as e:
            print(f"⚠️  Error logging to Comet.ml: {e}")

    print("✅ Training completed!")
    print(f"📊 Best model saved to: runs/detect/bird_head_yolov8n/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
