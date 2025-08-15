from rfdetr import RFDETRMedium
import torch
import os
from rfdetr.deploy.export import (
    no_batch_norm,
    make_infer_image,
    export_onnx,
    onnx_simplify,
)


model = RFDETRMedium(
    num_classes=4, device="cpu", pretrain_weights=None, num_queries=100
)
model.model.reinitialize_detection_head(5)

# output4 trained w orient head, 0-based classes
checkpoint = torch.load(
    "../output5/checkpoint_best_regular.pth", map_location="cpu", weights_only=False
)
model.model.model.load_state_dict(checkpoint["model"], strict=True)


x = torch.randn(1, 3, 800, 800)
y = model.model.model.forward(x)


y["pred_orient"]
inner_model = model.model.model
n_parameters = sum(p.numel() for p in inner_model.parameters())
print(f"number of parameters: {n_parameters}")
n_backbone_parameters = sum(p.numel() for p in inner_model.backbone.parameters())
print(f"number of backbone parameters: {n_backbone_parameters}")
n_projector_parameters = sum(
    p.numel() for p in inner_model.backbone[0].projector.parameters()
)
print(f"number of projector parameters: {n_projector_parameters}")
n_backbone_encoder_parameters = sum(
    p.numel() for p in inner_model.backbone[0].encoder.parameters()
)
print(f"number of backbone encoder parameters: {n_backbone_encoder_parameters}")
n_transformer_parameters = sum(p.numel() for p in inner_model.transformer.parameters())
print(f"number of transformer parameters: {n_transformer_parameters}")

if model.get_model_config().layer_norm:
    no_batch_norm(inner_model)

device = "cpu"
inner_model.to(device)

infer_dir = "../../example.jpg"
res = model.model.resolution
print(f"Model resolution: {res}")

input_tensors = make_infer_image(
    infer_dir, shape=(res, res), batch_size=1, device=device
)
print(f"Input tensors shape: {input_tensors.shape}")
input_names = ["input"]
output_names = ["dets", "labels", "orients"]
dynamic_axes = None
# Run inner_model inference in pytorch mode
inner_model.eval().to(device)
input_tensors = input_tensors.to(device)
with torch.no_grad():
    outputs = inner_model(input_tensors)
    dets = outputs["pred_boxes"]
    labels = outputs["pred_logits"]
    orients = outputs["pred_orient"]
    print(
        f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, Orients: {orients.shape}"
    )
inner_model.cpu()
input_tensors = input_tensors.cpu()

# output_dir,
# model,
# input_names,
# input_tensors,
# output_names,
# dynamic_axes,
# backbone_only=False,
# verbose=True,
# opset_version=17,


# verbose = True
# training = False
# val_do_constant_folding = True
# operator_export_type= torch.onnx.utils._C_onnx.OperatorExportTypes.ONNX
# fixed_batch_size = 1
# output_names = ["dets", "labels", "orients"]
# graph, params_dict, torch_out = torch.onnx.utils._model_to_graph(
#                 inner_model,
#                 input_tensors,
#                 verbose,
#                 input_names,
#                 output_names,
#                 operator_export_type,
#                 val_do_constant_folding,
#                 fixed_batch_size=fixed_batch_size,
#                 training=training,
#                 dynamic_axes=dynamic_axes,
#             )

model.optimize_for_inference()
output_dir = "export_output"
os.makedirs(output_dir, exist_ok=True)

# export_model model.model.inference_model
export_model = model.model.inference_model
print(f"Export model resolution: {model._optimized_resolution}")
output_file = export_onnx(
    output_dir=output_dir,
    model=export_model,
    input_names=input_names,
    input_tensors=input_tensors,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    verbose=False,
)

# onnx_dir: str, input_names, input_tensors, force
output_file = onnx_simplify(output_file, input_names, input_tensors)
# if args.simplify:
# output_file = onnx_simplify(output_file, input_names, input_tensors, args)


# Run ONNX inference with `output_file`
print("Testing running inference")
import onnxruntime as ort

ort_session = ort.InferenceSession(output_file)

# Add batch dimension to input_tensors
input_tensors_batched = input_tensors.unsqueeze(0)
onnx_inputs = {
    name: tensor.cpu().numpy()
    for name, tensor in zip(input_names, input_tensors_batched)
}
outputs = ort_session.run(output_names, onnx_inputs)

# Add output names from ORT session
output_names = [output.name for output in ort_session.get_outputs()]
named_outputs = {name: output for name, output in zip(output_names, outputs)}

print(f"It works! Got outputs {named_outputs.keys()}")
