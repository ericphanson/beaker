from paq2piq.inference_model import InferenceModel
from paq2piq.model import RoIPoolModel
from pathlib import Path

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Wrapper that computes blockwise ROIs from the input tensor shape and
    calls the original model with [image, rois] so RoIPool is exercised
    inside the exported ONNX graph.

    The wrapper accepts a tensor of shape (B,3,H,W) and returns the same
    output as the original model.forward().
    """

    def __init__(self, model: nn.Module, blk_size=(20, 20), include_image=True):
        super().__init__()
        self.model = model
        self.blk_size = blk_size
        self.include_image = include_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        batch_size = x.size(0)
        H = x.size(2)
        W = x.size(3)

        bs_y, bs_x = self.blk_size

        # build 1D coordinate tensors
        y = torch.linspace(0, float(H), steps=bs_y + 1, device=x.device)
        x_lin = torch.linspace(0, float(W), steps=bs_x + 1, device=x.device)

        boxes = []
        if self.include_image:
            boxes.append(
                torch.stack(
                    [
                        torch.tensor(0.0, device=x.device),
                        torch.tensor(0.0, device=x.device),
                        torch.tensor(float(W), device=x.device),
                        torch.tensor(float(H), device=x.device),
                    ]
                )
            )

        for n in range(bs_y):
            for m in range(bs_x):
                x1 = x_lin[m]
                y1 = y[n]
                x2 = x_lin[m + 1]
                y2 = y[n + 1]
                boxes.append(torch.stack([x1, y1, x2, y2]))

        boxes = torch.stack(boxes).float()

        # replicate per-batch and flatten to (batch_size * n_rois, 4)
        boxes = (
            boxes.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1).view(-1, 4)
        )

        # call the original model with [image, rois]
        return self.model([x, boxes])


def export_onnx(model_path: Path, onnx_path: Path, img_size=(224, 224)):
    inf = InferenceModel(RoIPoolModel(), model_path)
    model = inf.model
    model.eval()

    wrapper = ExportWrapper(model, blk_size=inf.blk_size, include_image=True)

    # export on CPU for portability
    wrapper.to("cpu")

    dummy = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="RoIPoolModel-fit.10.bs.120.pth",
        help="path to model state dict",
    )
    p.add_argument("--onnx", default="RoIPoolModel.onnx", help="output ONNX path")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    args = p.parse_args()

    export_onnx(Path(args.model), Path(args.onnx), img_size=(args.height, args.width))
    print(f"Exported ONNX model to {args.onnx}")
