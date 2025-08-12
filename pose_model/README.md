# README

## Steps

### Get the checkpoint
Download the checkpoint from `https://drive.google.com/file/d/1yELGKtpIc9gBCQmwMqZ-6J-GQ7l6q1KY/view?usp=drive_link` saving it to vhr_birdpose_s_add.pth. This is provided by the paper author's https://github.com/LuoXishuang0712/VHR-BirdPose/tree/main?tab=readme-ov-file#pretrained.

You can validate it with `uv run python summarize_checkpoint.py`.

### Export to ONNX
```sh
uv run python export_vhr_birdpose.py --aggressive-quantization --fp16
```

### Run
Place bird photos in "birds".

Generate crops:
```sh
beaker detect --crop bird --output-dir birds-crops -v --confidence 0.8 birds
```

Run VHR pose detect:
```sh
uv run python batch_inference.py --input-dir birds-crops --output-dir birds-crops-output --save-visualizations --backend coreml
```

### Citation
Any usage of this model should cite the paper:
```bibtex
@Article{
    he2023vhrbirdpose,
    AUTHOR = {He, Runang and Wang, Xiaomin and Chen, Huazhen and Liu, Chang},
    TITLE = {VHR-BirdPose: Vision Transformer-Based HRNet for Bird Pose Estimation with Attention Mechanism},
    JOURNAL = {Electronics},
    VOLUME = {12},
    YEAR = {2023},
    NUMBER = {17},
    ARTICLE-NUMBER = {3643},
}
```
I am not affiliated with the authors.
