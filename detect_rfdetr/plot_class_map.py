#!/usr/bin/env python3
"""
Plot per-class mAP over epochs from training log JSON lines file.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log_data(log_file):
    """Load and parse the JSON lines log file for all metric combinations."""
    epochs = []
    data_dict = {
        "test_map@50:95": {},
        "test_map@50": {},
        "ema_test_map@50:95": {},
        "ema_test_map@50": {},
    }

    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            epoch = data.get("epoch", 0)
            epochs.append(epoch)

            # Extract regular test results
            test_results = data.get("test_results_json", {})
            test_class_map = test_results.get("class_map", [])

            # Extract EMA test results
            ema_test_results = data.get("ema_test_results_json", {})
            ema_test_class_map = ema_test_results.get("class_map", [])

            # Process regular test results
            for class_info in test_class_map:
                class_name = class_info["class"]

                # map@50:95
                if class_name not in data_dict["test_map@50:95"]:
                    data_dict["test_map@50:95"][class_name] = []
                data_dict["test_map@50:95"][class_name].append(class_info["map@50:95"])

                # map@50
                if class_name not in data_dict["test_map@50"]:
                    data_dict["test_map@50"][class_name] = []
                data_dict["test_map@50"][class_name].append(class_info["map@50"])

            # Process EMA test results
            for class_info in ema_test_class_map:
                class_name = class_info["class"]

                # map@50:95
                if class_name not in data_dict["ema_test_map@50:95"]:
                    data_dict["ema_test_map@50:95"][class_name] = []
                data_dict["ema_test_map@50:95"][class_name].append(
                    class_info["map@50:95"]
                )

                # map@50
                if class_name not in data_dict["ema_test_map@50"]:
                    data_dict["ema_test_map@50"][class_name] = []
                data_dict["ema_test_map@50"][class_name].append(class_info["map@50"])

    return epochs, data_dict


def plot_single_class_curve(epochs, values, color, class_name, ax):
    """Plot a single class curve with running maximum and annotation."""
    # Plot actual values
    ax.plot(
        epochs,
        values,
        marker="o",
        linewidth=2,
        markersize=4,
        color=color,
        label=class_name.capitalize(),
        alpha=0.8,
    )

    # Calculate and plot running maximum
    running_max = np.maximum.accumulate(values)
    ax.plot(epochs, running_max, linestyle="--", linewidth=2, color=color, alpha=0.6)

    # Add annotation for global maximum
    max_value = max(values)
    max_epoch_idx = values.index(max_value)
    max_epoch = epochs[max_epoch_idx]
    ax.annotate(
        f"{max_value:.2f}",
        xy=(max_epoch, max_value),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=9,
        color=color,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8
        ),
    )


def create_legends(ax):
    """Create the dual legend setup."""
    from matplotlib.lines import Line2D

    # Create main legend for classes
    legend1 = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.7),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title="Classes",
    )

    # Add legend for line styles
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-", label="Current mAP"),
        Line2D(
            [0], [0], color="gray", linewidth=2, linestyle="--", label="Running Max"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.3),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title="Line Types",
    )

    # Add the first legend back
    ax.add_artist(legend1)


def plot_class_map(epochs, class_data, metric_name, output_file, title_suffix):
    """Create a plot of per-class mAP over epochs."""
    # Set up the plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Define colors for different classes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot each class (excluding 'all' if present)
    class_names = [name for name in class_data.keys() if name != "all"]

    for i, class_name in enumerate(class_names):
        values = class_data[class_name]
        color = colors[i % len(colors)]
        plot_single_class_curve(epochs, values, color, class_name, ax)

    # Add overall mAP if available
    if "all" in class_data:
        all_values = class_data["all"]
        ax.plot(
            epochs,
            all_values,
            marker="s",
            linewidth=3,
            markersize=5,
            color="black",
            label="Overall mAP",
            alpha=0.9,
        )

        # Add running maximum for overall mAP
        all_running_max = np.maximum.accumulate(all_values)
        ax.plot(
            epochs,
            all_running_max,
            linestyle="--",
            linewidth=3,
            color="black",
            alpha=0.6,
        )

        # Add annotation for overall mAP global maximum
        max_value = max(all_values)
        max_epoch_idx = all_values.index(max_value)
        max_epoch = epochs[max_epoch_idx]
        ax.annotate(
            f"{max_value:.2f}",
            xy=(max_epoch, max_value),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=9,
            color="black",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="black",
                alpha=0.8,
            ),
        )

    # Customize the plot
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"Per-Class {metric_name} Progress Over Training Epochs{title_suffix}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Create legends
    create_legends(ax)

    # Set axis limits with some padding
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
    y_min = min([min(values) for values in class_data.values()])
    y_max = max([max(values) for values in class_data.values()])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Plot saved as {output_file}")

    # Close the figure to free memory
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot per-class mAP over epochs from training log JSON lines file."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="output/log.txt",
        help="Path to the log file (default: output/log.txt)",
    )
    args = parser.parse_args()

    # Path to the log file
    log_file = Path(args.log_file)

    if not log_file.exists():
        print(f"Error: Log file {log_file} not found!")
        return

    print(f"Loading data from {log_file}...")
    epochs, data_dict = load_log_data(log_file)

    print(f"Found {len(epochs)} epochs")

    # Get the directory where the log file is located
    log_dir = log_file.parent

    # Define plot configurations
    plot_configs = [
        ("test_map@50:95", "mAP@50:95", "test_map50_95.png", ""),
        ("test_map@50", "mAP@50", "test_map50.png", ""),
        ("ema_test_map@50:95", "mAP@50:95", "ema_test_map50_95.png", " (EMA)"),
        ("ema_test_map@50", "mAP@50", "ema_test_map50.png", " (EMA)"),
    ]

    # Create all plots
    for data_key, metric_name, output_file, title_suffix in plot_configs:
        if data_key in data_dict and data_dict[data_key]:
            print(f"\nCreating plot for {data_key}...")
            print(f"Classes: {list(data_dict[data_key].keys())}")
            # Save plot in the same directory as the log file
            output_path = log_dir / output_file
            plot_class_map(
                epochs, data_dict[data_key], metric_name, output_path, title_suffix
            )
        else:
            print(f"Warning: No data found for {data_key}")


if __name__ == "__main__":
    main()
