from rfdetr import RFDETRNano

# Use 4 classes for parts detection: bird, head, eye, beak
model = RFDETRNano(num_classes=4)

# actually re-init head! Otherwise has 90 classes:
assert model.model.model.class_embed.out_features == 91

model.model.reinitialize_detection_head(5)
assert model.model.model.class_embed.out_features == 5

dataset_dir = "../data/cub_coco_parts"
model.train(dataset_dir=dataset_dir, epochs=1, output_dir="output_debug", num_workers=0)
