import shutil
from src.quantizations.quantizer import quantize_model
import pathlib

shutil.copy(
    "../quality_model/RoIPoolModel.onnx",
    "output/models/quality.onnx",
)

output_path = quantize_model(
    pathlib.Path("output/models/quality.onnx"),
    pathlib.Path("output/quantized"),
    "dynamic",
)

shutil.move(output_path, "output/quantized/quality-dynamic-int8.onnx")
