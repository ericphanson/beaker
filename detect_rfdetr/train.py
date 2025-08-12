import comet_ml
import os

from rfdetr import RFDETRNano

api_key = os.getenv("COMET_API_KEY")

project_name = os.getenv("COMET_PROJECT_NAME", "bird-head-detector")
workspace = os.getenv("COMET_WORKSPACE")

experiment = comet_ml.Experiment(
    api_key=api_key,
    project_name=project_name,
    workspace=workspace,
)

# Use 4 classes for parts detection: bird, head, eye, beak
model = RFDETRNano(num_classes=4)

dataset_dir = "../data/cub_coco_parts"
model.train(dataset_dir=dataset_dir, epochs=100)
