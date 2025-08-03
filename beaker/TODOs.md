# TODOs

- [x] support input as directory. Each image inside should be processed independently/identically. Only use coreml when on a directory with enough inputs and ensure we only do the model loading once (is this ort session reuse? not sure). The model loading is much slower on coreml but inference is 2x faster. Bake in heuristics but allow overrides with `--device` or similar.
- [x] update to ORT v2-rc. This will need detailed steps. This should allow us to ship pyke supplied binaries easily.
- [ ] support raw image inputs, including panasonic rw2. We should convert these to JPEG, run the yolo model, obtain the bounding boxes, then go back to the raw input and crop that to produce DNG raw output. Normally we try to match output image format to input, but if it's a proprietary raw format we will just do DNG. Use crates dng for output writing, rawloader for input reading, and image for cropping.
- [x] hide all printing unless we pass `--verbose`. This is a top-level flag, not a head-model specific one. We should be silent unless `--verbose` is passed. Then we can include the same printing as now. Additionally, we should tend to use `--verbose` in tests/CI so we can see what is going on.
- [ ] support stdin when `-` is passed as the input image name
- [x] add back version command
- [ ] pipeline subcommand. Used as `beaker pipeline --steps cutout,head --output-dir out_dir path/*.jpg`. This should have a help like
```
Usage: beaker pipeline [OPTIONS] --steps <STEPS> <IMAGES_OR_DIRS>...

Options:
  --steps <STEPS>
      Comma-separated list of processing steps to apply (e.g., cutout,head)
  --output-dir <OUTPUT_DIR>
      Global output directory
  --device <DEVICE>
      Inference device (auto, cpu, coreml)
  --no-metadata
  --verbose
  --shared-options ...
      (Or allow step-specific override syntax like: cutout:--alpha-matting)
```
