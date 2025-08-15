import comet_ml
import os

from rfdetr import RFDETRMedium

api_key = os.getenv("COMET_API_KEY")

project_name = os.getenv("COMET_PROJECT_NAME", "bird-head-detector")
workspace = os.getenv("COMET_WORKSPACE")

experiment = comet_ml.Experiment(
    api_key=api_key,
    project_name=project_name,
    workspace=workspace,
)

output_dir = "output5"

if os.path.isdir(output_dir):
    raise Exception("Output dir already exists")

# resolution = 560
# Use 4 classes for parts detection: bird, head, eye, beak
model = RFDETRMedium(
    num_classes=4, class_names=["bird", "head", "eye", "beak"], num_queries=100
)

# actually re-init head! Otherwise has 90 classes:
assert model.model.model.class_embed.out_features == 91

model.model.reinitialize_detection_head(5)
assert model.model.model.class_embed.out_features == 5

dataset_dir = "../data/cub_coco_parts"
# assert model.get_model_config().resolution == resolution
model.train(dataset_dir=dataset_dir, epochs=100, output_dir=output_dir)
