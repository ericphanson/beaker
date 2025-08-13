import torchlens as tl
import torch
from rfdetr import RFDETRNano

# Use 4 classes for parts detection: bird, head, eye, beak
model = RFDETRNano(num_classes=4)

x = torch.rand(1, 3, 640, 640)
y = torch.rand(1, 640, 640) < 0.5
model_history = tl.log_forward_pass(
    model.model.model, (x, y), vis_opt="rolled", vis_nesting_depth=1
)
print(model_history)
