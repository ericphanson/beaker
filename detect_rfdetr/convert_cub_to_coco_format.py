#!/usr/bin/env python3
"""
Convert CUB-200-2011 dataset to COCO JSON format for object detection training.

The CUB-200-2011 dataset contains:
- images.txt: Image ID and file path
- image_class_labels.txt: Image ID and class label
- classes.txt: Class ID and class name
- bounding_boxes.txt: Image ID and bounding box coordinates
- train_test_split.txt: Image ID and train/test split indicator
- parts/part_locs.txt: Image ID, part ID, X, Y, visible (for parts detection)

This script converts the dataset to COCO format with proper annotations.
For parts mode, it creates 4 classes: bird (whole bird), head, eye, beak.
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, List, Tuple, Optional
import numpy as np


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


def get_crown_parts():
    """Define which parts constitute the crown"""
    # Crown part
    return [5]


def get_tail_parts():
    """Define which parts constitute the tail"""
    # Tail part
    return [14]


def calculate_orientation_angle(
    from_point: Tuple[float, float], to_point: Tuple[float, float]
) -> float:
    """
    Calculate orientation angle from one point to another relative to positive horizontal axis.

    Args:
        from_point: (x, y) starting point
        to_point: (x, y) ending point

    Returns:
        Angle in radians relative to positive horizontal axis, or NaN if points coincide
    """
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]

    # Return NaN if points coincide (dx=dy=0)
    if dx == 0 and dy == 0:
        return np.nan

    return np.arctan2(dy, dx)


class CUBtoCOCOConverter:
    def __init__(self, cub_root: str, output_dir: str):
        """
        Initialize the converter.

        Args:
            cub_root: Path to CUB-200-2011 dataset root directory
            output_dir: Directory to save COCO format JSON files
        """
        self.cub_root = Path(cub_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CUB dataset file paths
        self.images_file = self.cub_root / "images.txt"
        self.labels_file = self.cub_root / "image_class_labels.txt"
        self.classes_file = self.cub_root / "classes.txt"
        self.bbox_file = self.cub_root / "bounding_boxes.txt"
        self.split_file = self.cub_root / "train_test_split.txt"
        self.images_dir = self.cub_root / "images"

        # Parts dataset file
        self.parts_file = self.cub_root / "parts" / "part_locs.txt"

    def load_cub_data(self) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        """Load all CUB dataset files."""
        print("Loading CUB dataset files...")

        # Load images.txt: image_id image_path
        images = {}
        with open(self.images_file, "r") as f:
            for line in f:
                img_id, img_path = line.strip().split(" ", 1)
                images[int(img_id)] = img_path

        # Load image_class_labels.txt: image_id class_id
        labels = {}
        with open(self.labels_file, "r") as f:
            for line in f:
                img_id, class_id = line.strip().split()
                labels[int(img_id)] = int(class_id)

        # Load classes.txt: class_id class_name
        classes = {}
        with open(self.classes_file, "r") as f:
            for line in f:
                class_id, class_name = line.strip().split(" ", 1)
                classes[int(class_id)] = class_name

        # Load bounding_boxes.txt: image_id x y width height
        bboxes = {}
        with open(self.bbox_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                x, y, width, height = map(float, parts[1:5])
                bboxes[img_id] = [x, y, width, height]

        # Load train_test_split.txt: image_id is_training_image
        splits = {}
        with open(self.split_file, "r") as f:
            for line in f:
                img_id, is_train = line.strip().split()
                splits[int(img_id)] = int(is_train) == 1

        # Load parts data
        print("Loading parts annotations...")
        parts_data = {}
        if self.parts_file.exists():
            with open(self.parts_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    img_id, part_id, x, y, visible = (
                        int(parts[0]),
                        int(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        int(parts[4]),
                    )
                    if img_id not in parts_data:
                        parts_data[img_id] = {}
                    parts_data[img_id][part_id] = {"x": x, "y": y, "visible": visible}

        return images, labels, classes, bboxes, splits, parts_data

    def create_coco_categories(self, classes: Dict[int, str]) -> List[Dict]:
        """Create COCO categories for parts detection."""
        # Create 4-class categories for parts detection
        categories = [
            {"id": 0, "name": "bird", "supercategory": "bird"},
            {"id": 1, "name": "head", "supercategory": "bird"},
            {"id": 2, "name": "eye", "supercategory": "bird"},
            {"id": 3, "name": "beak", "supercategory": "bird"},
        ]
        return categories

    def create_parts_bounding_box(
        self, img_id: int, part_ids: List[int], parts_data: Dict, margin: int = 10
    ) -> Optional[List[float]]:
        """Create a bounding box around specified parts."""
        if img_id not in parts_data:
            return None

        # Collect visible parts coordinates
        coords = []
        for part_id in part_ids:
            if part_id in parts_data[img_id] and parts_data[img_id][part_id]["visible"]:
                coords.append(
                    [parts_data[img_id][part_id]["x"], parts_data[img_id][part_id]["y"]]
                )

        if not coords:
            return None

        # Calculate bounding box
        coords = np.array(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # Add margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        width = x_max - x_min + 2 * margin
        height = y_max - y_min + 2 * margin

        return [float(x_min), float(y_min), float(width), float(height)]

    def get_parts_center(
        self, img_id: int, part_ids: List[int], parts_data: Dict
    ) -> Optional[Tuple[float, float]]:
        """Get the center point of specified parts."""
        if img_id not in parts_data:
            return None

        # Collect visible parts coordinates
        coords = []
        for part_id in part_ids:
            if part_id in parts_data[img_id] and parts_data[img_id][part_id]["visible"]:
                coords.append(
                    [parts_data[img_id][part_id]["x"], parts_data[img_id][part_id]["y"]]
                )

        if not coords:
            return None

        # Calculate center
        coords = np.array(coords)
        center_x = float(np.mean(coords[:, 0]))
        center_y = float(np.mean(coords[:, 1]))

        return (center_x, center_y)

    def calculate_head_orientation(
        self, img_id: int, parts_data: Dict
    ) -> Optional[float]:
        """Calculate head orientation from center of eyes to center of beak."""
        # Get center of eyes
        eye_center = self.get_parts_center(img_id, get_eye_parts(), parts_data)
        if eye_center is None:
            return None

        # Get center of beak
        beak_center = self.get_parts_center(img_id, get_beak_parts(), parts_data)
        if beak_center is None:
            return None

        # Calculate angle from eyes to beak
        return calculate_orientation_angle(eye_center, beak_center)

    def calculate_bird_orientation(
        self, img_id: int, parts_data: Dict
    ) -> Optional[float]:
        """Calculate bird orientation from crown to tail."""
        # Get center of crown
        crown_center = self.get_parts_center(img_id, get_crown_parts(), parts_data)
        if crown_center is None:
            return None

        # Get center of tail
        tail_center = self.get_parts_center(img_id, get_tail_parts(), parts_data)
        if tail_center is None:
            return None

        # Calculate angle from crown to tail
        return calculate_orientation_angle(crown_center, tail_center)

    def get_image_info(self, img_id: int, img_path: str) -> Dict:
        """Get image information including dimensions."""
        full_img_path = self.images_dir / img_path

        try:
            with Image.open(full_img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not open image {full_img_path}: {e}")
            # Use default dimensions if image can't be opened
            width, height = 500, 375

        return {"id": img_id, "file_name": img_path, "width": width, "height": height}

    def create_coco_annotation(
        self,
        ann_id: int,
        img_id: int,
        class_id: int,
        bbox: List[float],
        img_width: int,
        img_height: int,
        orientation: Optional[float] = None,
    ) -> Dict:
        """Create a COCO annotation from CUB data."""
        x, y, width, height = bbox

        # Ensure bbox is within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))

        area = width * height

        annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": class_id,
            "bbox": [x, y, width, height],
            "area": area,
            "iscrowd": 0,
        }

        # Add orientation if provided and not NaN
        if orientation is not None and not np.isnan(orientation):
            annotation["orient"] = float(orientation)

        return annotation

    def convert_split(
        self,
        split_name: str,
        img_ids: List[int],
        images_data: Dict,
        labels_data: Dict,
        classes_data: Dict,
        bboxes_data: Dict,
        parts_data: Dict,
    ) -> Dict:
        """Convert a data split to COCO format."""
        print(f"Converting {split_name} split with {len(img_ids)} images...")

        # Create COCO structure
        coco_data = {
            "info": {
                "description": f"CUB-200-2011 {split_name} set in COCO format (parts mode with orientation)",
                "url": "http://www.vision.caltech.edu/visipedia/CUB-200-2011.html",
                "version": "1.0",
                "year": 2011,
                "contributor": "Wah et al.",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "categories": self.create_coco_categories(classes_data),
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img_id in img_ids:
            img_path = images_data[img_id]

            # Get image info
            img_info = self.get_image_info(img_id, img_path)
            coco_data["images"].append(img_info)

            # Create annotations for each part type
            annotations = []

            # 1. Bird (whole bird) - use original bounding box with orientation
            if img_id in bboxes_data:
                bbox = bboxes_data[img_id]
                bird_orientation = self.calculate_bird_orientation(img_id, parts_data)
                annotation = self.create_coco_annotation(
                    ann_id,
                    img_id,
                    0,  # category_id=0 for bird
                    bbox,
                    img_info["width"],
                    img_info["height"],
                    orientation=bird_orientation,
                )
                annotations.append(annotation)
                ann_id += 1

            # 2. Head parts with orientation
            head_bbox = self.create_parts_bounding_box(
                img_id, get_head_parts(), parts_data
            )
            if head_bbox:
                head_orientation = self.calculate_head_orientation(img_id, parts_data)
                annotation = self.create_coco_annotation(
                    ann_id,
                    img_id,
                    1,  # category_id=1 for head
                    head_bbox,
                    img_info["width"],
                    img_info["height"],
                    orientation=head_orientation,
                )
                annotations.append(annotation)
                ann_id += 1

            # 3. Eye parts (no orientation)
            eye_bbox = self.create_parts_bounding_box(
                img_id, get_eye_parts(), parts_data
            )
            if eye_bbox:
                annotation = self.create_coco_annotation(
                    ann_id,
                    img_id,
                    2,  # category_id=2 for eye
                    eye_bbox,
                    img_info["width"],
                    img_info["height"],
                )
                annotations.append(annotation)
                ann_id += 1

            # 4. Beak parts (no orientation)
            beak_bbox = self.create_parts_bounding_box(
                img_id, get_beak_parts(), parts_data
            )
            if beak_bbox:
                annotation = self.create_coco_annotation(
                    ann_id,
                    img_id,
                    3,  # category_id=3 for beak
                    beak_bbox,
                    img_info["width"],
                    img_info["height"],
                )
                annotations.append(annotation)
                ann_id += 1

            coco_data["annotations"].extend(annotations)

        return coco_data

    def convert(self):
        """Main conversion function."""
        print("Starting CUB to COCO conversion...")

        # Verify CUB dataset files exist
        required_files = [
            self.images_file,
            self.labels_file,
            self.classes_file,
            self.bbox_file,
            self.split_file,
        ]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required CUB file not found: {file_path}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.parts_file.exists():
            raise FileNotFoundError(f"Parts file not found: {self.parts_file}")

        # Load CUB data
        images_data, labels_data, classes_data, bboxes_data, splits_data, parts_data = (
            self.load_cub_data()
        )

        # Split images into train and test sets
        train_img_ids = [img_id for img_id, is_train in splits_data.items() if is_train]
        test_img_ids = [
            img_id for img_id, is_train in splits_data.items() if not is_train
        ]

        print(
            f"Found {len(train_img_ids)} training images and {len(test_img_ids)} test images"
        )
        print("Using parts mode: 4 classes (bird, head, eye, beak) with orientation")
        print(f"Total classes: {len(classes_data)}")

        # Convert train split
        train_coco = self.convert_split(
            "train",
            train_img_ids,
            images_data,
            labels_data,
            classes_data,
            bboxes_data,
            parts_data,
        )

        # Convert test split
        test_coco = self.convert_split(
            "test",
            test_img_ids,
            images_data,
            labels_data,
            classes_data,
            bboxes_data,
            parts_data,
        )

        # Save COCO format files
        suffix = "_parts"
        train_file = self.output_dir / f"cub_train{suffix}.json"
        test_file = self.output_dir / f"cub_test{suffix}.json"

        with open(train_file, "w") as f:
            json.dump(train_coco, f, indent=2)

        with open(test_file, "w") as f:
            json.dump(test_coco, f, indent=2)

        print("Conversion complete!")
        print(f"Train annotations saved to: {train_file}")
        print(f"Test annotations saved to: {test_file}")

        # Print summary statistics
        print("\nSummary:")
        print(f"- Training images: {len(train_coco['images'])}")
        print(f"- Training annotations: {len(train_coco['annotations'])}")
        print(f"- Test images: {len(test_coco['images'])}")
        print(f"- Test annotations: {len(test_coco['annotations'])}")
        print(f"- Total categories: {len(train_coco['categories'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CUB-200-2011 dataset to COCO format with parts and orientation"
    )
    parser.add_argument(
        "--cub_root",
        type=str,
        default="../data/CUB_200_2011",
        help="Path to CUB-200-2011 dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/coco_annotations",
        help="Directory to save COCO format JSON files",
    )

    args = parser.parse_args()

    # Verify CUB root directory exists
    if not os.path.exists(args.cub_root):
        print(f"Error: CUB dataset directory not found: {args.cub_root}")
        print("Please check the path and try again.")
        return

    # Create converter and run conversion
    converter = CUBtoCOCOConverter(args.cub_root, args.output_dir)
    try:
        converter.convert()
    except Exception as e:
        print(f"Error during conversion: {e}")
        return


if __name__ == "__main__":
    main()
