import shutil
from src.quantizations.quantizer import quantize_model
import pathlib

shutil.copy(
    "../detect_rfdetr/onnx_export/export_output/inference_model.sim.onnx",
    "output/models/rfdetr.sim.onnx",
)

quantize_model(
    pathlib.Path("output/models/rfdetr.sim.onnx"),
    pathlib.Path("output/quantized"),
    "dynamic",
)
