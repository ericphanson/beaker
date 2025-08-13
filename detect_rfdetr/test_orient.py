from rfdetr import RFDETRNano
import torch

model = RFDETRNano(num_classes=4, device="cpu", pretrain_weights=None)
model.model.reinitialize_detection_head(5)

x = torch.randn(1, 3, 800, 800)
y = model.model.model.forward(x)

y["pred_orient"]

model.get_train_config(dataset_dir="../data/cub_coco_parts")
