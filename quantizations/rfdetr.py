from src.quantizations.quantizer import quantize_model
import pathlib

quantize_model(
    pathlib.Path("output/models/rfdetr.sim.onnx"),
    pathlib.Path("output/quantized"),
    "dynamic",
)
