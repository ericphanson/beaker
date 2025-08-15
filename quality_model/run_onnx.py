import onnxruntime as ort
from pathlib import Path
from PIL import Image
import numpy as np

from paq2piq.common import Transform


def run_image_through_onnx(
    onnx_path: Path,
    image_path: Path,
    height: int = 224,
    width: int = 224,
    blk_size=(20, 20),
):
    # load and preprocess
    img = Image.open(image_path).convert("RGB")
    # resize with PIL to the exported model size
    img_resized = img.resize((width, height), resample=Image.BILINEAR)
    transform = Transform().val_transform
    t = transform(img_resized)  # Tensor C,H,W
    input_tensor = t.unsqueeze(0).numpy().astype(np.float32)  # N,C,H,W

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_tensor})
    out = outputs[0]  # numpy array, shape (B, N)

    # convert to same postprocessing as InferenceModel.predict
    out0 = out[0]
    global_score = float(out0[0])
    local_scores = np.reshape(out0[1:], blk_size)
    return {"global_score": global_score, "local_scores": local_scores}


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Run an image through the exported RoIPool ONNX model"
    )
    p.add_argument(
        "--onnx", default="quality-dynamic-int8.onnx", help="path to ONNX model"
    )
    p.add_argument("--image", default="../example.jpg", help="path to input image")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    image_path = Path(args.image)
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}. Run script.py to export it first.")
        sys.exit(1)
    if not image_path.exists():
        print(f"Image not found at {image_path}")
        sys.exit(1)

    out = run_image_through_onnx(
        onnx_path, image_path, height=args.height, width=args.width
    )
    print("Global score:", out["global_score"])
    print("Local scores shape:", out["local_scores"].shape)
    print("Done")
