# This is a python script to convert the CUB 200 parts labels to yolo format.
# Converts to normalized YOLO cx, cy, w, h format and map img_id to file paths using images.txt, train_test_split.txt.

# it uses numpy and polars

# file 1:
# data/CUB_200_2011/parts/part_locs.txt
# first line: `1 1 0.0 0.0 0 2.329`

# file 2:
# data/CUB_200_2011/parts/part_locs.txt
# first line: `1 back`

# file 3:
# data/CUB_200_2011/bounding_boxes.txt
# first line: `1 60.0 27.0 325.0 304.0`

# file 4:
# data/CUB_200_2011/image_class_labels.txt
# first line: `1 1`

# file 5:
# data/CUB_200_2011/images.txt
# first line: `1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg`
