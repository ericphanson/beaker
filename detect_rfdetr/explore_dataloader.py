#!/usr/bin/env python3
"""
Interactive script to explore the data loader format
"""

import torch
import sys

sys.path.append("rfdetr")

from rfdetr.main import populate_args
from rfdetr.datasets import build_dataset
import rfdetr.util.misc as utils
from torch.utils.data import DataLoader


def explore_dataloader():
    """Interactively explore the data loader to understand the format"""

    # Create minimal args for dataset building
    args = populate_args(
        dataset_file="roboflow",  # Changed to roboflow since structure is different
        coco_path="../data/cub_coco_parts",  # Updated to correct directory
        dataset_dir="../data/cub_coco_parts",  # Updated dataset directory
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        resolution=640,
    )

    # Add missing attributes that the dataset building expects
    args.patch_size = 16
    args.num_windows = 4
    args.square_resize = False
    args.square_resize_div_64 = False

    print("Building dataset...")
    try:
        dataset_train = build_dataset(
            image_set="train", args=args, resolution=args.resolution
        )
        print(f"Dataset built successfully. Length: {len(dataset_train)}")
    except Exception as e:
        print(f"Error building dataset: {e}")
        return

    # Create data loader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    print("Data loader created successfully")

    # Get first batch
    print("\nGetting first batch...")
    try:
        batch = next(iter(data_loader_train))
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch)}")

        # Explore batch structure
        for i, item in enumerate(batch):
            print(f"\nBatch item {i}:")
            print(f"  Type: {type(item)}")

            if hasattr(item, "tensors"):
                print(f"  Has tensors attribute: {item.tensors.shape}")
                print(f"  Tensors dtype: {item.tensors.dtype}")
                print(f"  Has mask: {item.mask is not None}")
                if item.mask is not None:
                    print(f"  Mask shape: {item.mask.shape}")
            elif isinstance(item, torch.Tensor):
                print(f"  Tensor shape: {item.shape}")
                print(f"  Tensor dtype: {item.dtype}")
            elif isinstance(item, list):
                print(f"  List length: {len(item)}")
                if len(item) > 0:
                    print(f"  First item type: {type(item[0])}")
                    if isinstance(item[0], dict):
                        print(f"  First dict keys: {list(item[0].keys())}")
                        for key, value in item[0].items():
                            if isinstance(value, torch.Tensor):
                                print(f"    {key}: {value.shape}, {value.dtype}")
                            else:
                                print(f"    {key}: {type(value)}, {value}")
            else:
                print(f"  Content: {item}")

    except Exception as e:
        print(f"Error getting batch: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    explore_dataloader()
