#!/usr/bin/env python3
"""
Checkpoint Analysis Script for Bird Head Detection Model

This script analyzes the performance evolution of YOLOv8 model checkpoints
by evaluating each checkpoint on the validation set and plotting metrics
over training epochs.

Features:
- Per-class mAP@0.5 and mAP@0.5:0.95 tracking
- Configurable epoch subsampling (e.g., every 5 or 10 epochs)
- Validation set fraction control for faster evaluation
- Beautiful matplotlib plots with class-specific trends
- Progress tracking with estimated time remaining
- Caching of evaluation results to avoid re-computation

Usage:
    # Analyze all checkpoints
    python analyze_checkpoints.py

    # Analyze every 5th epoch with 50% of validation data
    python analyze_checkpoints.py --epoch-step 5 --val-fraction 0.5

    # Analyze specific epoch range
    python analyze_checkpoints.py --start-epoch 10 --end-epoch 50

    # Use cached results if available
    python analyze_checkpoints.py --use-cache
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

# Set up plot styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Class names for the 4-class bird detection model
CLASS_NAMES = ["bird", "head", "eye", "beak"]
CLASS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
]  # Distinct colors for each class


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze YOLOv8 checkpoint performance over training epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all checkpoints
    python analyze_checkpoints.py

    # Analyze every 5th epoch with 50% validation data
    python analyze_checkpoints.py --epoch-step 5 --val-fraction 0.5

    # Analyze epochs 10-50 only
    python analyze_checkpoints.py --start-epoch 10 --end-epoch 50

    # Use faster evaluation settings
    python analyze_checkpoints.py --epoch-step 10 --val-fraction 0.2 --conf-thres 0.25
        """,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs/multi-detect/bird_multi_yolov8n8/weights",
        help="Directory containing checkpoint files (default: runs/multi-detect/bird_multi_yolov8n8/weights)",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="../data/yolo-4-class/dataset.yaml",
        help="Path to dataset YAML configuration (default: ../data/yolo-4-class/dataset.yaml)",
    )

    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Analyze every N epochs (default: 1, i.e., all epochs)",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Starting epoch to analyze (default: 0)",
    )

    parser.add_argument(
        "--end-epoch",
        type=int,
        default=None,
        help="Last epoch to analyze (default: None, i.e., all available)",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation set to use (0.1-1.0, default: 1.0)",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold for evaluation (default: 0.001)",
    )

    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (default: 0.6)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint_analysis",
        help="Directory to save plots and results (default: checkpoint_analysis)",
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached evaluation results if available",
    )

    parser.add_argument(
        "--save-cache",
        action="store_true",
        default=True,
        help="Save evaluation results to cache (default: True)",
    )

    return parser.parse_args()


def find_checkpoint_files(
    checkpoint_dir: Path,
    start_epoch: int = 0,
    end_epoch: Optional[int] = None,
    epoch_step: int = 1,
) -> List[Tuple[int, Path]]:
    """Find available checkpoint files in the specified range."""
    checkpoint_files = []

    # Find all epoch checkpoint files
    for checkpoint_path in sorted(checkpoint_dir.glob("epoch*.pt")):
        try:
            # Extract epoch number from filename
            epoch_num = int(checkpoint_path.stem.replace("epoch", ""))

            # Check if epoch is in the desired range and step
            if epoch_num < start_epoch:
                continue
            if end_epoch is not None and epoch_num > end_epoch:
                continue
            if (epoch_num - start_epoch) % epoch_step != 0:
                continue

            checkpoint_files.append((epoch_num, checkpoint_path))

        except ValueError:
            # Skip files that don't follow the epoch naming convention
            continue

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])

    print(f"Found {len(checkpoint_files)} checkpoint files to analyze")
    if checkpoint_files:
        epochs = [epoch for epoch, _ in checkpoint_files]
        print(f"Epoch range: {min(epochs)} - {max(epochs)} (step: {epoch_step})")

    return checkpoint_files


def create_validation_subset(
    data_config: str, val_fraction: float, output_dir: Path
) -> str:
    """Create a subset of the validation set for faster evaluation."""
    if val_fraction >= 1.0:
        return data_config

    import random
    import yaml

    # Set random seed for reproducible subsampling
    random.seed(42)

    print(f"Creating validation subset with {val_fraction:.1%} of data...")

    # Load original dataset config
    with open(data_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Create subset directory
    subset_dir = output_dir / f"val_subset_{val_fraction:.1f}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Get original validation directory
    original_data_path = Path(data_config).parent
    val_images_dir = (
        original_data_path / dataset_config["val"] / "images"
        if "images" not in dataset_config["val"]
        else original_data_path / dataset_config["val"]
    )
    val_labels_dir = val_images_dir.parent / "labels"

    # Create subset directories
    subset_images_dir = subset_dir / "images"
    subset_labels_dir = subset_dir / "labels"
    subset_images_dir.mkdir(exist_ok=True)
    subset_labels_dir.mkdir(exist_ok=True)

    # Get all validation images
    val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))

    # Sample subset
    subset_size = max(1, int(len(val_images) * val_fraction))
    sampled_images = random.sample(val_images, subset_size)

    print(f"Using {len(sampled_images)}/{len(val_images)} validation images")

    # Create symlinks for sampled images and labels
    for img_path in sampled_images:
        # Create symlink for image
        dst_img = subset_images_dir / img_path.name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.absolute())

        # Create symlink for label if it exists
        label_name = img_path.stem + ".txt"
        src_label = val_labels_dir / label_name
        dst_label = subset_labels_dir / label_name
        if src_label.exists() and not dst_label.exists():
            dst_label.symlink_to(src_label.absolute())

    # Create subset dataset config
    subset_config = dataset_config.copy()
    subset_config["path"] = str(subset_dir.parent.absolute())
    subset_config["val"] = f"{subset_dir.name}/images"

    subset_yaml_path = subset_dir / "dataset.yaml"
    with open(subset_yaml_path, "w") as f:
        yaml.dump(subset_config, f, default_flow_style=False)

    return str(subset_yaml_path)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_config: str,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
) -> Dict:
    """Evaluate a single checkpoint and return metrics."""
    print(f"  Evaluating {checkpoint_path.name}...")

    # Load model
    model = YOLO(str(checkpoint_path))

    # Run validation
    results = model.val(
        data=data_config,
        conf=conf_thres,
        iou=iou_thres,
        plots=False,
        verbose=False,
        save_json=False,
    )

    # Extract metrics using the correct property access
    metrics = {
        "epoch": int(checkpoint_path.stem.replace("epoch", "")),
        "checkpoint_path": str(checkpoint_path),
        # Overall metrics
        "map50": float(results.box.map50),  # mAP@0.5
        "map": float(results.box.map),  # mAP@0.5:0.95
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": float(
            2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr)
            if (results.box.mp + results.box.mr) > 0
            else 0
        ),
        # Per-class mAP@0.5
        "map50_per_class": results.box.ap50.tolist()
        if len(results.box.ap50) > 0
        else [0] * len(CLASS_NAMES),
        # Per-class mAP@0.5:0.95
        "map_per_class": results.box.maps.tolist()
        if len(results.box.maps) > 0
        else [0] * len(CLASS_NAMES),
        # Per-class precision and recall
        "precision_per_class": results.box.p.tolist()
        if hasattr(results.box, "p") and results.box.p is not None
        else [0] * len(CLASS_NAMES),
        "recall_per_class": results.box.r.tolist()
        if hasattr(results.box, "r") and results.box.r is not None
        else [0] * len(CLASS_NAMES),
    }

    return metrics


def load_cache(cache_path: Path) -> Dict:
    """Load cached evaluation results."""
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    return {}


def save_cache(cache_path: Path, results: Dict):
    """Save evaluation results to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"✅ Results cached to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def evaluate_all_checkpoints(
    checkpoint_files: List[Tuple[int, Path]], data_config: str, args, cache_path: Path
) -> List[Dict]:
    """Evaluate all checkpoint files and return results."""

    # Load cache if requested
    cached_results = load_cache(cache_path) if args.use_cache else {}

    all_results = []
    total_checkpoints = len(checkpoint_files)

    print(f"\n🚀 Starting evaluation of {total_checkpoints} checkpoints...")
    start_time = time.time()

    for i, (epoch_num, checkpoint_path) in enumerate(checkpoint_files, 1):
        # Check cache first
        cache_key = f"{checkpoint_path.name}_{args.conf_thres}_{args.iou_thres}_{args.val_fraction}"
        if cache_key in cached_results:
            print(
                f"  [{i:2d}/{total_checkpoints}] Using cached result for {checkpoint_path.name}"
            )
            all_results.append(cached_results[cache_key])
            continue

        # Progress update with ETA
        elapsed = time.time() - start_time
        if i > 1:
            avg_time_per_checkpoint = elapsed / (i - 1)
            remaining_time = avg_time_per_checkpoint * (total_checkpoints - i + 1)
            eta_str = f"ETA: {remaining_time/60:.1f}m"
        else:
            eta_str = "ETA: calculating..."

        print(f"  [{i:2d}/{total_checkpoints}] {checkpoint_path.name} ({eta_str})")

        # Evaluate checkpoint
        result = evaluate_checkpoint(
            checkpoint_path, data_config, args.conf_thres, args.iou_thres
        )

        all_results.append(result)
        # Cache the result
        cached_results[cache_key] = result

        # Save cache periodically
        if args.save_cache and i % 5 == 0:
            save_cache(cache_path, cached_results)

    # Final cache save
    if args.save_cache:
        save_cache(cache_path, cached_results)

    total_time = time.time() - start_time
    print(f"\n✅ Evaluation completed in {total_time/60:.1f} minutes")
    print(
        f"📊 Successfully evaluated {len(all_results)}/{total_checkpoints} checkpoints"
    )

    return all_results


def create_plots(results: List[Dict], output_dir: Path):
    """Create comprehensive plots showing metric evolution."""

    if not results:
        print("❌ No results to plot!")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    df = df.sort_values("epoch")

    print(f"📈 Creating plots for {len(df)} epochs...")

    # Create output directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib parameters
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Overall metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Overall Model Performance Evolution", fontsize=16, fontweight="bold")

    # mAP@0.5
    axes[0, 0].plot(
        df["epoch"], df["map50"], "b-", linewidth=2, marker="o", markersize=4
    )
    axes[0, 0].set_title("mAP@0.5", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("mAP@0.5")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # mAP@0.5:0.95
    axes[0, 1].plot(df["epoch"], df["map"], "g-", linewidth=2, marker="o", markersize=4)
    axes[0, 1].set_title("mAP@0.5:0.95", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mAP@0.5:0.95")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Precision and Recall
    axes[1, 0].plot(
        df["epoch"],
        df["precision"],
        "r-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Precision",
    )
    axes[1, 0].plot(
        df["epoch"],
        df["recall"],
        "orange",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Recall",
    )
    axes[1, 0].set_title("Precision and Recall", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # F1 Score
    axes[1, 1].plot(
        df["epoch"], df["f1"], "purple", linewidth=2, marker="o", markersize=4
    )
    axes[1, 1].set_title("F1 Score", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    overall_plot_path = plots_dir / "overall_metrics.png"
    plt.savefig(overall_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Overall metrics plot saved: {overall_plot_path}")

    # 2. Per-class mAP@0.5 plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        class_map50 = [row[i] for row in df["map50_per_class"]]
        ax.plot(
            df["epoch"],
            class_map50,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=class_name.capitalize(),
            color=CLASS_COLORS[i],
        )

    ax.set_title("Per-Class mAP@0.5 Evolution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Add annotations for final values
    for i, class_name in enumerate(CLASS_NAMES):
        final_map = df["map50_per_class"].iloc[-1][i]
        ax.annotate(
            f"{final_map:.3f}",
            xy=(df["epoch"].iloc[-1], final_map),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CLASS_COLORS[i], alpha=0.2),
        )

    plt.tight_layout()
    class_map50_plot_path = plots_dir / "per_class_map50.png"
    plt.savefig(class_map50_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Per-class mAP@0.5 plot saved: {class_map50_plot_path}")

    # 3. Per-class mAP@0.5:0.95 plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        class_map = [row[i] for row in df["map_per_class"]]
        ax.plot(
            df["epoch"],
            class_map,
            linewidth=2.5,
            marker="s",
            markersize=5,
            label=class_name.capitalize(),
            color=CLASS_COLORS[i],
        )

    ax.set_title("Per-Class mAP@0.5:0.95 Evolution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP@0.5:0.95", fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Add annotations for final values
    for i, class_name in enumerate(CLASS_NAMES):
        final_map = df["map_per_class"].iloc[-1][i]
        ax.annotate(
            f"{final_map:.3f}",
            xy=(df["epoch"].iloc[-1], final_map),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CLASS_COLORS[i], alpha=0.2),
        )

    plt.tight_layout()
    class_map_plot_path = plots_dir / "per_class_map.png"
    plt.savefig(class_map_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Per-class mAP@0.5:0.95 plot saved: {class_map_plot_path}")

    # 4. Combined comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Comprehensive Class Performance Analysis", fontsize=16, fontweight="bold"
    )

    # mAP@0.5 comparison
    for i, class_name in enumerate(CLASS_NAMES):
        class_map50 = [row[i] for row in df["map50_per_class"]]
        axes[0, 0].plot(
            df["epoch"],
            class_map50,
            linewidth=2,
            label=class_name.capitalize(),
            color=CLASS_COLORS[i],
        )
    axes[0, 0].set_title("mAP@0.5 by Class")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("mAP@0.5")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # mAP@0.5:0.95 comparison
    for i, class_name in enumerate(CLASS_NAMES):
        class_map = [row[i] for row in df["map_per_class"]]
        axes[0, 1].plot(
            df["epoch"],
            class_map,
            linewidth=2,
            label=class_name.capitalize(),
            color=CLASS_COLORS[i],
        )
    axes[0, 1].set_title("mAP@0.5:0.95 by Class")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mAP@0.5:0.95")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Class performance ranking at final epoch
    final_map50_per_class = df["map50_per_class"].iloc[-1]
    final_map_per_class = df["map_per_class"].iloc[-1]

    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.35

    bars1 = axes[1, 0].bar(
        x_pos - width / 2,
        final_map50_per_class,
        width,
        label="mAP@0.5",
        color=CLASS_COLORS,
        alpha=0.8,
    )
    bars2 = axes[1, 0].bar(
        x_pos + width / 2,
        final_map_per_class,
        width,
        label="mAP@0.5:0.95",
        color=CLASS_COLORS,
        alpha=0.5,
    )

    axes[1, 0].set_title("Final Performance by Class")
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("mAP Score")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name.capitalize() for name in CLASS_NAMES])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    axes[1, 0].set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[1, 0].annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Performance improvement over training
    initial_map50 = df["map50_per_class"].iloc[0]
    final_map50 = df["map50_per_class"].iloc[-1]
    improvement = [
        final - initial for final, initial in zip(final_map50, initial_map50)
    ]

    bars = axes[1, 1].bar(
        range(len(CLASS_NAMES)), improvement, color=CLASS_COLORS, alpha=0.8
    )
    axes[1, 1].set_title("mAP@0.5 Improvement (Final - Initial)")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("mAP@0.5 Improvement")
    axes[1, 1].set_xticks(range(len(CLASS_NAMES)))
    axes[1, 1].set_xticklabels([name.capitalize() for name in CLASS_NAMES])
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].annotate(
            f"{height:+.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    combined_plot_path = plots_dir / "comprehensive_analysis.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Comprehensive analysis plot saved: {combined_plot_path}")


def save_results_summary(results: List[Dict], output_dir: Path):
    """Save detailed results summary to files."""

    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("epoch")

    # Save CSV
    csv_path = output_dir / "checkpoint_metrics.csv"

    # Flatten per-class metrics for CSV
    csv_data = []
    for _, row in df.iterrows():
        base_data = {
            "epoch": row["epoch"],
            "map50": row["map50"],
            "map": row["map"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        }

        # Add per-class metrics
        for i, class_name in enumerate(CLASS_NAMES):
            base_data[f"map50_{class_name}"] = row["map50_per_class"][i]
            base_data[f"map_{class_name}"] = row["map_per_class"][i]
            if len(row["precision_per_class"]) > i:
                base_data[f"precision_{class_name}"] = row["precision_per_class"][i]
            if len(row["recall_per_class"]) > i:
                base_data[f"recall_{class_name}"] = row["recall_per_class"][i]

        csv_data.append(base_data)

    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"  ✅ Metrics CSV saved: {csv_path}")

    # Save JSON with full details
    json_path = output_dir / "checkpoint_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Detailed JSON saved: {json_path}")

    # Create summary statistics
    summary_stats = {
        "total_epochs_analyzed": len(results),
        "epoch_range": {"start": df["epoch"].min(), "end": df["epoch"].max()},
        "best_overall_performance": {
            "epoch": df.loc[df["map50"].idxmax(), "epoch"],
            "map50": df["map50"].max(),
            "map": df.loc[df["map50"].idxmax(), "map"],
        },
        "final_performance": {
            "epoch": df["epoch"].iloc[-1],
            "map50": df["map50"].iloc[-1],
            "map": df["map"].iloc[-1],
            "per_class_map50": dict(zip(CLASS_NAMES, df["map50_per_class"].iloc[-1])),
            "per_class_map": dict(zip(CLASS_NAMES, df["map_per_class"].iloc[-1])),
        },
        "class_rankings_final": {
            "by_map50": sorted(
                zip(CLASS_NAMES, df["map50_per_class"].iloc[-1]),
                key=lambda x: x[1],
                reverse=True,
            ),
            "by_map": sorted(
                zip(CLASS_NAMES, df["map_per_class"].iloc[-1]),
                key=lambda x: x[1],
                reverse=True,
            ),
        },
    }

    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  ✅ Summary statistics saved: {summary_path}")

    # Print key findings
    print("\n📋 Key Findings:")
    print(
        f"   • Best overall mAP@0.5: {summary_stats['best_overall_performance']['map50']:.3f} (epoch {summary_stats['best_overall_performance']['epoch']})"
    )
    print(f"   • Final mAP@0.5: {summary_stats['final_performance']['map50']:.3f}")
    print(f"   • Final mAP@0.5:0.95: {summary_stats['final_performance']['map']:.3f}")
    print("   • Class rankings by final mAP@0.5:")
    for i, (class_name, score) in enumerate(
        summary_stats["class_rankings_final"]["by_map50"], 1
    ):
        print(f"     {i}. {class_name.capitalize()}: {score:.3f}")


def main():
    """Main analysis function."""
    args = parse_args()

    print("🔍 Bird Head Detection Checkpoint Analysis")
    print("=" * 50)

    # Validate inputs
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate fraction
    if not 0.1 <= args.val_fraction <= 1.0:
        print(
            f"❌ Validation fraction must be between 0.1 and 1.0, got {args.val_fraction}"
        )
        return

    # Find checkpoint files
    checkpoint_files = find_checkpoint_files(
        checkpoint_dir, args.start_epoch, args.end_epoch, args.epoch_step
    )

    if not checkpoint_files:
        print("❌ No checkpoint files found in the specified range!")
        return

    # Setup validation data
    if args.val_fraction < 1.0:
        data_config = create_validation_subset(
            args.data_config, args.val_fraction, output_dir
        )
    else:
        data_config = args.data_config

    print(f"📊 Using data config: {data_config}")
    print("⚙️  Evaluation settings:")
    print(f"   • Confidence threshold: {args.conf_thres}")
    print(f"   • IoU threshold: {args.iou_thres}")
    print(f"   • Validation fraction: {args.val_fraction:.1%}")

    # Setup cache
    cache_path = output_dir / "evaluation_cache.pkl"

    # Evaluate all checkpoints
    results = evaluate_all_checkpoints(checkpoint_files, data_config, args, cache_path)

    if not results:
        print("❌ No successful evaluations!")
        return

    # Create plots and save results
    print("\n📈 Generating analysis outputs...")
    create_plots(results, output_dir)
    save_results_summary(results, output_dir)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("🔍 Check the following files:")
    print("   • plots/overall_metrics.png - Overall performance trends")
    print("   • plots/per_class_map50.png - Per-class mAP@0.5 evolution")
    print("   • plots/comprehensive_analysis.png - Complete analysis")
    print("   • checkpoint_metrics.csv - Detailed metrics data")
    print("   • summary_statistics.json - Key performance insights")


if __name__ == "__main__":
    main()
