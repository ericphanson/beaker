from rfdetr import RFDETRNano

# Use 4 classes for parts detection: bird, head, eye, beak
model = RFDETRNano(num_classes=4)

dataset_dir = "../data/cub_coco_parts"
model.train(dataset_dir=dataset_dir, epochs=1, batch_size=8, grad_accum_steps=2)
