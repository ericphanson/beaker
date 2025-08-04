# TODOs

## New commands

- does this photo have a bird and where (YOLO bird object detection)
- wings/beaks etc (detect other parts besides head using CUB dataset)
- species classification

## Improve existing commands
- [ ] support raw image inputs, including panasonic rw2. We should convert these to JPEG, run the yolo model, obtain the bounding boxes, then go back to the raw input and crop that to produce DNG raw output. Normally we try to match output image format to input, but if it's a proprietary raw format we will just do DNG. Use crates dng for output writing, rawloader for input reading, and image for cropping.
- [ ] support stdin when `-` is passed as the input image name
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
