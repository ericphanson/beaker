import os
import shutil
import json
import random

# Create COCO dataset structure for RF-DETR
coco_dir = "../data/cub_coco_parts"

# RF-DETR expects train/, valid/, and test/ directories with _annotations.coco.json files
os.makedirs(f"{coco_dir}/train", exist_ok=True)
os.makedirs(f"{coco_dir}/valid", exist_ok=True)
os.makedirs(f"{coco_dir}/test", exist_ok=True)

print(f"Created COCO dataset structure at {coco_dir}")

# Copy training annotations
shutil.copy(
    "../data/coco_annotations/cub_train_parts.json",
    f"{coco_dir}/train/_annotations.coco.json",
)

print("Copied training annotation files")

# Create symlinks to CUB images to avoid duplicating data
cub_images_dir = "../data/CUB_200_2011/images"
for split in ["train", "valid", "test"]:
    link_path = f"{coco_dir}/{split}/cub_images"

    # Remove if exists
    if os.path.exists(link_path):
        os.unlink(link_path)

    # Create symlink
    os.symlink(os.path.abspath(cub_images_dir), link_path)

print("Created symlinks to CUB images")

# Load the original test/val annotations and split them
with open("../data/coco_annotations/cub_test_parts.json", "r") as f:
    original_data = json.load(f)

# Get all images and shuffle them
images = original_data["images"]
random.seed(42)  # For reproducible splits
random.shuffle(images)

# Split into validation (first half) and test (second half)
mid_point = len(images) // 2
val_images = images[:mid_point]
test_images = images[mid_point:]

print(
    f"Split original {len(images)} images into {len(val_images)} validation and {len(test_images)} test images"
)

# Create image ID sets for filtering annotations
val_image_ids = {img["id"] for img in val_images}
test_image_ids = {img["id"] for img in test_images}

# Filter annotations for each split
val_annotations = [
    ann for ann in original_data["annotations"] if ann["image_id"] in val_image_ids
]
test_annotations = [
    ann for ann in original_data["annotations"] if ann["image_id"] in test_image_ids
]

# Create validation dataset
val_data = original_data.copy()
val_data["images"] = val_images
val_data["annotations"] = val_annotations
val_data["info"]["description"] = (
    "CUB-200-2011 validation set in COCO format (parts mode)"
)

# Create test dataset
test_data = original_data.copy()
test_data["images"] = test_images
test_data["annotations"] = test_annotations
test_data["info"]["description"] = "CUB-200-2011 test set in COCO format (parts mode)"

# Fix image paths in annotations to match RF-DETR expectations and save
for split, data in [("train", None), ("valid", val_data), ("test", test_data)]:
    if split == "train":
        # Load and update train annotations
        ann_file = f"{coco_dir}/{split}/_annotations.coco.json"
        with open(ann_file, "r") as f:
            data = json.load(f)

    # Update image file names to include the symlink directory
    for img in data["images"]:
        img["file_name"] = f'cub_images/{img["file_name"]}'

    # Save the updated annotations
    ann_file = f"{coco_dir}/{split}/_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(data, f, indent=2)

    print(
        f'Updated {split} annotations - {len(data["images"])} images, {len(data["annotations"])} annotations'
    )

print("Dataset structure ready for RF-DETR!")
