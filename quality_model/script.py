from paq2piq.inference_model import InferenceModel
from paq2piq.model import RoIPoolModel
from pathlib import Path


model = InferenceModel(RoIPoolModel(), Path("RoIPoolModel-fit.10.bs.120.pth"))
