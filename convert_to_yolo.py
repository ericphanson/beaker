# This is a python script to convert the CUB 200 parts labels to yolo format.
# Converts to normalized YOLO cx, cy, w, h format and map img_id to file paths using images.txt, train_test_split.txt.

import os
import shutil
from pathlib import Path

import polars as pl
from PIL import Image
from tqdm import tqdm


def load_data():
    """Load all necessary data files"""
    data_dir = Path("data/CUB_200_2011")

    # Load part locations
    part_locs = pl.read_csv(
        data_dir / "parts/part_locs.txt",
        separator=" ",
        has_header=False,
        new_columns=["img_id", "part_id", "x", "y", "visible"],
    )

    # Load part names
    parts = pl.read_csv(
        data_dir / "parts/parts.txt",
        separator=" ",
        has_header=False,
        new_columns=["part_id", "part_name"],
    )

    # Load bounding boxes
    bboxes = pl.read_csv(
        data_dir / "bounding_boxes.txt",
        separator=" ",
        has_header=False,
        new_columns=["img_id", "x", "y", "width", "height"],
    )

    # Load image paths
    images = pl.read_csv(
        data_dir / "images.txt", separator=" ", has_header=False, new_columns=["img_id", "filepath"]
    )

    # Load train/test split
    train_test = pl.read_csv(
        data_dir / "train_test_split.txt",
        separator=" ",
        has_header=False,
        new_columns=["img_id", "is_training"],
    )

    return part_locs, parts, bboxes, images, train_test


def get_head_parts():
    """Define which parts constitute the bird head"""
    # Head-related parts: beak, crown, forehead, left eye, nape, right eye, throat
    return [2, 5, 6, 7, 10, 11, 15]


def calculate_head_bbox(img_parts, img_width, img_height):
    """Calculate bounding box for bird head based on visible head parts"""
    visible_parts = img_parts.filter(pl.col("visible") == 1)

    if len(visible_parts) == 0:
        return None

    # Get min/max coordinates of visible head parts
    x_coords = visible_parts["x"].to_list()
    y_coords = visible_parts["y"].to_list()

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add some padding around the parts (10% of the span in each direction)
    x_span = x_max - x_min
    y_span = y_max - y_min

    padding_x = max(x_span * 0.1, 10)  # At least 10 pixels
    padding_y = max(y_span * 0.1, 10)  # At least 10 pixels

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(img_width, x_max + padding_x)
    y_max = min(img_height, y_max + padding_y)

    # Convert to YOLO format (normalized cx, cy, w, h)
    width = x_max - x_min
    height = y_max - y_min
    cx = (x_min + x_max) / 2 / img_width
    cy = (y_min + y_max) / 2 / img_height
    w = width / img_width
    h = height / img_height

    return cx, cy, w, h


def get_image_dimensions(image_path):
    """Get image dimensions"""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except:
        return None, None


def convert_to_yolo():
    """Main conversion function"""
    print("Loading data...")
    part_locs, parts, bboxes, images, train_test = load_data()

    # Filter for head parts only
    head_part_ids = get_head_parts()
    head_parts = part_locs.filter(pl.col("part_id").is_in(head_part_ids))

    # Join with images and train/test data
    data = head_parts.join(images, on="img_id").join(train_test, on="img_id")

    # Create output directories
    os.makedirs("data/yolo/train/images", exist_ok=True)
    os.makedirs("data/yolo/train/labels", exist_ok=True)
    os.makedirs("data/yolo/val/images", exist_ok=True)
    os.makedirs("data/yolo/val/labels", exist_ok=True)

    # Process each unique image
    unique_images = data.select(["img_id", "filepath", "is_training"]).unique()
    total_images = len(unique_images)

    print(f"Processing {total_images} unique images...")

    processed = 0
    skipped = 0

    # Use tqdm for progress bar
    for row in tqdm(
        unique_images.iter_rows(named=True), total=total_images, desc="Converting images"
    ):
        img_id = row["img_id"]
        filepath = row["filepath"]
        is_training = row["is_training"]

        # Get image dimensions
        full_image_path = f"data/CUB_200_2011/images/{filepath}"
        img_width, img_height = get_image_dimensions(full_image_path)

        if img_width is None or img_height is None:
            tqdm.write(f"Warning: Could not load image {filepath}")
            skipped += 1
            continue

        # Get all head parts for this image
        img_parts = head_parts.filter(pl.col("img_id") == img_id)

        # Calculate head bounding box
        head_bbox = calculate_head_bbox(img_parts, img_width, img_height)

        if head_bbox is None:
            tqdm.write(f"Warning: No visible head parts for image {filepath}")
            skipped += 1
            continue

        # Determine output directory
        split_dir = "train" if is_training == 1 else "val"

        # Copy image to YOLO directory
        image_name = Path(filepath).name
        yolo_image_path = f"data/yolo/{split_dir}/images/{image_name}"
        os.makedirs(os.path.dirname(yolo_image_path), exist_ok=True)

        # Copy the image file
        try:
            shutil.copy2(full_image_path, yolo_image_path)
        except Exception as e:
            tqdm.write(f"Warning: Could not copy image {filepath}: {e}")
            skipped += 1
            continue

        # Create YOLO label file
        label_name = Path(image_name).stem + ".txt"
        label_path = f"data/yolo/{split_dir}/labels/{label_name}"

        # Write label (class 0 for bird head, followed by bbox coordinates)
        cx, cy, w, h = head_bbox
        with open(label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        processed += 1

    print("Conversion complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")

    # Create dataset YAML file
    yaml_content = """# Bird Head Detection Dataset
path: data/yolo
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['bird_head']  # class names
"""

    with open("data/yolo/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("Created dataset.yaml configuration file")


def main():
    """Entry point for the beaker-convert script."""
    convert_to_yolo()


if __name__ == "__main__":
    convert_to_yolo()
