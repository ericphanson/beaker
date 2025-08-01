#!/usr/bin/env python3
"""
Train YOLOv8n model for bird head detection on M1 MacBook Pro with Comet.ml tracking.

Debug Mode:
    Set TRAINING_CONFIG['debug_run'] = True for quick testing with reduced data and epochs.
    This creates a subset dataset (10% by default) and trains for only 5 epochs.

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

# Training Configuration
TRAINING_CONFIG = {
    'model': 'yolov8n',
    'model_file': 'yolov8n.pt',
    'data': 'data/yolo/dataset.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # Adjust based on M1 memory
    'project': 'runs/detect',
    'name': 'bird_head_yolov8n',
    'workers': 0,  # Prevent multiprocessing issues on M1
    'verbose': True,
    'task': 'bird_head_detection',
    'dataset': 'CUB-200-2011',
    'architecture': 'YOLOv8n',

    # Debug Configuration
    'debug_run': False,  # Set to True for quick testing
    'debug_epochs': 5,   # Reduced epochs for debug
    'debug_fraction': 0.1,  # Use 10% of data for debug (0.1 = 10%)
}


def setup_comet(device):
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

        # Log hyperparameters from global config
        log_params = TRAINING_CONFIG.copy()
        log_params['device'] = device
        experiment.log_parameters(log_params)

        print(f"✅ Comet.ml experiment started: {experiment.url}")
        return experiment

    except Exception as e:
        print(f"❌ Failed to initialize Comet.ml: {e}")
        print("   Training will continue without experiment tracking.")
        return None


def create_debug_dataset():
    """Create a subset dataset configuration for debug runs."""
    import yaml
    from pathlib import Path
    import shutil
    import random

    debug_dir = Path("data/yolo_debug")
    debug_dir.mkdir(exist_ok=True)

    # Create debug directories
    for split in ['train', 'val']:
        (debug_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (debug_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy subset of files
    original_dir = Path("data/yolo")
    fraction = TRAINING_CONFIG['debug_fraction']

    for split in ['train', 'val']:
        original_images = list((original_dir / split / 'images').glob('*.jpg'))
        sample_size = max(1, int(len(original_images) * fraction))
        sampled_images = random.sample(original_images, sample_size)

        print(f"📊 Debug: Using {len(sampled_images)}/{len(original_images)} {split} images ({fraction*100:.1f}%)")

        for img_path in sampled_images:
            # Copy image
            dst_img = debug_dir / split / 'images' / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy corresponding label
            label_name = img_path.stem + '.txt'
            src_label = original_dir / split / 'labels' / label_name
            dst_label = debug_dir / split / 'labels' / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

    # Create debug dataset.yaml
    debug_yaml = {
        'path': str(debug_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['bird_head']
    }

    yaml_path = debug_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(debug_yaml, f, default_flow_style=False)

    return str(yaml_path)


def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        device = 'mps'
    else:
        print("❌ MPS not available, falling back to CPU")
        device = 'cpu'

    # Check if running in debug mode
    is_debug = TRAINING_CONFIG['debug_run']
    if is_debug:
        print("🐛 DEBUG MODE: Quick training run enabled")
        print(f"   - Epochs: {TRAINING_CONFIG['debug_epochs']} (vs {TRAINING_CONFIG['epochs']} normal)")
        print(f"   - Data subset: {TRAINING_CONFIG['debug_fraction']*100:.1f}% of full dataset")

        # Create debug dataset
        data_config = create_debug_dataset()
        epochs = TRAINING_CONFIG['debug_epochs']
        run_name = f"{TRAINING_CONFIG['name']}_debug"
    else:
        print("🚀 FULL TRAINING MODE")
        data_config = TRAINING_CONFIG['data']
        epochs = TRAINING_CONFIG['epochs']
        run_name = TRAINING_CONFIG['name']

    # Setup Comet.ml tracking with device info
    experiment = setup_comet(device)

    # Log debug mode info to Comet.ml
    if experiment:
        experiment.log_parameter('debug_mode', is_debug)
        if is_debug:
            experiment.log_parameter('debug_epochs', epochs)
            experiment.log_parameter('debug_fraction', TRAINING_CONFIG['debug_fraction'])

    # Load a pretrained YOLOv8n model
    print(f"📦 Loading {TRAINING_CONFIG['model']} pretrained model...")
    model = YOLO(TRAINING_CONFIG['model_file'])

    # Configure Comet.ml integration for YOLO
    if experiment:
        # YOLO will automatically detect and use the active Comet experiment
        os.environ['COMET_MODE'] = 'online'

    # Train the model on bird head dataset
    mode_text = "DEBUG" if is_debug else "FULL"
    print(f"🚀 Starting {mode_text} training...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=TRAINING_CONFIG['imgsz'],
        batch=TRAINING_CONFIG['batch'],
        device=device,
        project=TRAINING_CONFIG['project'],
        name=run_name,
        workers=TRAINING_CONFIG['workers'],
        verbose=TRAINING_CONFIG['verbose']
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
            best_model_path = f"{TRAINING_CONFIG['project']}/{run_name}/weights/best.pt"
            if os.path.exists(best_model_path):
                experiment.log_model('best_model', best_model_path)
                print("📤 Model uploaded to Comet.ml")

            experiment.end()
            print("✅ Comet.ml experiment completed")

        except Exception as e:
            print(f"⚠️  Error logging to Comet.ml: {e}")

    mode_text = "DEBUG" if is_debug else "FULL"
    print(f"✅ {mode_text} training completed!")
    print(f"📊 Best model saved to: {TRAINING_CONFIG['project']}/{run_name}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
