# Quality model

## Source code

paq2piq comes from:

- repo: https://github.com/baidut/paq2piq
- commit: 48c91e844e9f7a768f6ddcfc744dddfdd1160fea
- license: MIT

## Pretrained model

Supplied by paq2piq, `RoIPoolModel-fit.10.bs.120.pth`.

## Steps

1. download model
2. `uv run python export.py` - creates `RoIPoolModel.onnx`
3. `uv run run_onnx.py` - check onnx model works
4. in `../quantizations`, run `quality.py` to quantize
5. Can use this with `run_onnx.py` to check results
