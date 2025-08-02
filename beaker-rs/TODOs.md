# TODOs

- [ ] support input as directory. Each image inside should be processed independently/identically.
- [ ] update to ORT v2-rc. This will need detailed steps. This should allow us to ship pyke supplied binaries easily.
- [ ] support raw image inputs, including panasonic rw2. We should convert these to JPEG, run the yolo model, obtain the bounding boxes, then go back to the raw input and crop that to produce DNG raw output. Normally we try to match output image format to input, but if it's a proprietary raw format we will just do DNG.
- [ ] hide all printing unless we pass `--verbose`. This is a top-level flag, not a head-model specific one. We should be silent unless `--verbose` is passed. Then we can include the same printing as now.
- [ ] support stdin when `-` is passed as the input image name
