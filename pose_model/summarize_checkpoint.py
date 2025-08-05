#!/usr/bin/env python3
"""
Script to load a VHR-BirdPose model checkpoint and display model summary.
This script loads a .pth checkpoint and uses the get_model_summary function from utils.
Uses YAML configuration file for model setup.
"""

import os
import sys
import torch
import logging
import yaml

verbose = False

# Add the upstream directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "upstream"))

from upstream.pose_vhr import get_pose_net
from upstream.utils import get_model_summary


def load_config_from_yaml(yaml_path):
    """
    Load configuration from YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def load_model_checkpoint(checkpoint_path, cfg):
    """
    Load the VHR-BirdPose model and its checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file
        cfg (dict): Model configuration dictionary

    Returns:
        torch.nn.Module: Loaded model with checkpoint weights
    """
    print(
        f"Creating VHR-BirdPose model with '{cfg['MODEL']['EXTRA']['FUSE_STREGY']}' fusion strategy..."
    )

    # Create the model
    model = get_pose_net(cfg, is_train=False)

    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Found 'state_dict' key in checkpoint")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Found 'model' key in checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")

        # Print some checkpoint info
        print(f"Checkpoint contains {len(state_dict)} parameters")
        sample_keys = list(state_dict.keys())[:5]
        print(f"Sample parameter keys: {sample_keys}")

        # Load the state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            print("✓ Checkpoint loaded successfully!")
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}):")
                for key in missing_keys:
                    print(f"  - {key}")
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys:
                    print(f"  + {key}")
        except Exception as e:
            print(f"⚠ Warning: Could not load some parameters: {e}")
            # Try to load with prefix removal if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes if present
                new_key = k.replace("module.", "").replace("backbone.", "")
                new_state_dict[new_key] = v

            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    new_state_dict, strict=False
                )
                print("✓ Checkpoint loaded successfully with key adjustments!")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            except Exception as e2:
                print(f"✗ Error loading checkpoint: {e2}")
                return None
    else:
        print(f"✗ Checkpoint file not found: {checkpoint_path}")
        return None

    return model


def main():
    """Main function to load model and display summary."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Paths
    config_path = os.path.join(
        os.path.dirname(__file__), "w32_256x256_adam_lr1e-3_ak_vhr_s.yaml"
    )
    checkpoint_path = os.path.join(os.path.dirname(__file__), "vhr_birdpose_s_add.pth")

    print("=" * 80)
    print("VHR-BirdPose Model Checkpoint Loader")
    print("=" * 80)

    # Load configuration from YAML
    try:
        cfg = load_config_from_yaml(config_path)
        print(f"✓ Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return

    # Print model configuration info
    print("Model configuration:")
    print(f"  - Model name: {cfg['MODEL']['NAME']}")
    print(f"  - Number of joints: {cfg['MODEL']['NUM_JOINTS']}")
    print(f"  - Input size: {cfg['MODEL']['IMAGE_SIZE']}")
    print(f"  - Heatmap size: {cfg['MODEL']['HEATMAP_SIZE']}")
    print(f"  - Fusion strategy: {cfg['MODEL']['EXTRA']['FUSE_STREGY']}")
    print(f"  - ViT depth: {cfg['MODEL']['EXTRA']['VIT']['DEPTH']}")
    print(f"  - ViT heads: {cfg['MODEL']['EXTRA']['VIT']['NUM_HEADS']}")

    # Load the model and checkpoint
    model = load_model_checkpoint(checkpoint_path, cfg)

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Set model to evaluation mode
    model.eval()

    # Create a dummy input tensor for model summary
    # Use the input size from configuration
    batch_size = 1
    channels = 3  # RGB
    height, width = cfg["MODEL"]["IMAGE_SIZE"]

    dummy_input = torch.randn(batch_size, channels, height, width)

    print("\n" + "=" * 80)
    print("Generating Model Summary")
    print("=" * 80)
    print(f"Input shape: {dummy_input.shape}")

    # Generate and display model summary
    try:
        summary = get_model_summary(model, dummy_input, verbose=True)
        if verbose:
            print("Generating detailed model summary...")
            print(summary)
    except Exception as e:
        print(f"Error generating detailed model summary: {e}")
        print("Attempting basic model information...")

        # Fallback: basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        # Try a forward pass to check model functionality
        try:
            print("\nTesting model forward pass...")
            with torch.no_grad():
                output = model(dummy_input)
                print("✓ Model forward pass successful!")
                print(f"  - Input shape: {dummy_input.shape}")
                print(f"  - Output shape: {output.shape}")
                print(f"  - Expected heatmap size: {cfg['MODEL']['HEATMAP_SIZE']}")

                # Check if output shape matches expected heatmap size
                expected_shape = [batch_size, cfg["MODEL"]["NUM_JOINTS"]] + cfg[
                    "MODEL"
                ]["HEATMAP_SIZE"]
                if list(output.shape) == expected_shape:
                    print("✓ Output shape matches configuration!")
                else:
                    print(f"⚠ Output shape mismatch. Expected: {expected_shape}")

        except Exception as e2:
            print(f"✗ Model forward pass failed: {e2}")

    print("\n" + "=" * 80)
    print("Model loading and summary generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
