# This is a python script to convert the CUB 200 parts labels to yolo format.
# Converts to normalized YOLO cx, cy, w, h format and map img_id to file paths using images.txt, train_test_split.txt.
# Supports 4 classes: bird (0), head (1), eye (2), beak (3)

import os
import shutil
from pathlib import Path

import polars as pl
from PIL import Image
from tqdm import tqdm

# Global data prefix for easier path management
DATA_PREFIX = "../data"


def load_data():
    """Load all necessary data files"""
    data_dir = Path(DATA_PREFIX) / "CUB_200_2011"

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
        data_dir / "images.txt",
        separator=" ",
        has_header=False,
        new_columns=["img_id", "filepath"],
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


def get_eye_parts():
    """Define which parts constitute eyes"""
    # Eye parts: left eye, right eye
    return [7, 11]


def get_beak_parts():
    """Define which parts constitute the beak"""
    # Beak part
    return [2]


def calculate_part_bbox(
    img_parts, img_width, img_height, padding_factor=0.1, min_padding=10
):
    """Calculate bounding box for a set of parts"""
    visible_parts = img_parts.filter(pl.col("visible") == 1)

    if len(visible_parts) == 0:
        return None

    # Get min/max coordinates of visible parts
    x_coords = visible_parts["x"].to_list()
    y_coords = visible_parts["y"].to_list()

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add padding around the parts
    x_span = x_max - x_min
    y_span = y_max - y_min

    padding_x = max(x_span * padding_factor, min_padding)
    padding_y = max(y_span * padding_factor, min_padding)

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


def calculate_bird_bbox(bbox_row, img_width, img_height):
    """Convert CUB bounding box to YOLO format"""
    x, y, width, height = (
        bbox_row["x"],
        bbox_row["y"],
        bbox_row["width"],
        bbox_row["height"],
    )

    # Convert to YOLO format (normalized cx, cy, w, h)
    cx = (x + width / 2) / img_width
    cy = (y + height / 2) / img_height
    w = width / img_width
    h = height / img_height

    return cx, cy, w, h


def get_image_dimensions(image_path):
    """Get image dimensions"""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def convert_to_yolo():
    """Main conversion function"""
    print("Loading data...")
    part_locs, parts, bboxes, images, train_test = load_data()

    # Join all data together
    full_data = images.join(train_test, on="img_id").join(bboxes, on="img_id")

    # Create output directories
    os.makedirs(f"{DATA_PREFIX}/yolo-4-class/train/images", exist_ok=True)
    os.makedirs(f"{DATA_PREFIX}/yolo-4-class/train/labels", exist_ok=True)
    os.makedirs(f"{DATA_PREFIX}/yolo-4-class/val/images", exist_ok=True)
    os.makedirs(f"{DATA_PREFIX}/yolo-4-class/val/labels", exist_ok=True)

    # Process each unique image
    total_images = len(full_data)
    print(f"Processing {total_images} unique images...")

    processed = 0
    skipped = 0

    # Use tqdm for progress bar
    for row in tqdm(
        full_data.iter_rows(named=True),
        total=total_images,
        desc="Converting images",
    ):
        img_id = row["img_id"]
        filepath = row["filepath"]
        is_training = row["is_training"]

        # Get image dimensions
        full_image_path = f"{DATA_PREFIX}/CUB_200_2011/images/{filepath}"
        img_width, img_height = get_image_dimensions(full_image_path)

        if img_width is None or img_height is None:
            tqdm.write(f"Warning: Could not load image {filepath}")
            skipped += 1
            continue

        # Determine output directory
        split_dir = "train" if is_training == 1 else "val"

        # Copy image to YOLO directory
        image_name = Path(filepath).name
        yolo_image_path = f"{DATA_PREFIX}/yolo-4-class/{split_dir}/images/{image_name}"
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
        label_path = f"{DATA_PREFIX}/yolo-4-class/{split_dir}/labels/{label_name}"

        labels = []

        # 1. Bird bounding box (class 0) - from CUB bounding boxes
        bird_bbox = calculate_bird_bbox(row, img_width, img_height)
        cx, cy, w, h = bird_bbox
        labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Get all parts for this image
        img_parts = part_locs.filter(pl.col("img_id") == img_id)

        # 2. Head bounding box (class 1) - aggregated from head parts
        head_part_ids = get_head_parts()
        head_parts = img_parts.filter(pl.col("part_id").is_in(head_part_ids))
        head_bbox = calculate_part_bbox(head_parts, img_width, img_height)
        if head_bbox is not None:
            cx, cy, w, h = head_bbox
            labels.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 3. Eye bounding boxes (class 2) - from eye parts
        eye_part_ids = get_eye_parts()
        for eye_part_id in eye_part_ids:
            eye_part = img_parts.filter(
                (pl.col("part_id") == eye_part_id) & (pl.col("visible") == 1)
            )
            if len(eye_part) > 0:
                eye_bbox = calculate_part_bbox(
                    eye_part, img_width, img_height, padding_factor=0.15, min_padding=5
                )
                if eye_bbox is not None:
                    cx, cy, w, h = eye_bbox
                    labels.append(f"2 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 4. Beak bounding box (class 3) - from beak part
        beak_part_ids = get_beak_parts()
        beak_parts = img_parts.filter(
            pl.col("part_id").is_in(beak_part_ids) & (pl.col("visible") == 1)
        )
        if len(beak_parts) > 0:
            beak_bbox = calculate_part_bbox(
                beak_parts, img_width, img_height, padding_factor=0.2, min_padding=5
            )
            if beak_bbox is not None:
                cx, cy, w, h = beak_bbox
                labels.append(f"3 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Write all labels to file
        with open(label_path, "w") as f:
            for label in labels:
                f.write(label + "\n")

        processed += 1

    print("Conversion complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")

    # Create dataset YAML file
    yaml_content = f"""# Bird Multi-class Detection Dataset
path: {DATA_PREFIX}/yolo-4-class
train: train/images
val: val/images

# Classes
nc: 4  # number of classes
names: ['bird', 'head', 'eye', 'beak']  # class names
"""

    with open(f"{DATA_PREFIX}/yolo-4-class/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("Created dataset.yaml configuration file")


def main():
    """Entry point for the beaker-convert script."""
    convert_to_yolo()


if __name__ == "__main__":
    convert_to_yolo()
