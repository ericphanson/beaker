from rfdetr import RFDETRNano
import torch

model = RFDETRNano(num_classes=4, device="cpu", pretrain_weights=None)
model.model.reinitialize_detection_head(5)

# output3 trained w detect head
checkpoint = torch.load(
    "../output3/checkpoint_best_regular.pth", map_location="cpu", weights_only=False
)
model.model.model.load_state_dict(checkpoint["model"], strict=True)


x = torch.randn(1, 3, 800, 800)
y = model.model.model.forward(x)

y["pred_orient"]

help(model.export())
