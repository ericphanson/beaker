# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""Build a test set: a random sample of images plus degraded copies of each.

Every variant is saved as PNG, so the degradation is the only difference
between a copy and its original.

`--skip` takes a later slice of the same shuffle, which gives a second set
that shares no image with the first.

usage: uv run make_variants.py <images_root> <out_root> [--n 200] [--seed 0] [--skip 0]
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BLUR_SIGMAS = [1, 2, 4, 8]
VARIANTS = ["orig"] + [f"blur{s}" for s in BLUR_SIGMAS] + ["jpeg10", "noise25"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images_root", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", type=int, default=0)
    args = p.parse_args()

    paths = sorted(args.images_root.rglob("*.jpg"))
    random.Random(args.seed).shuffle(paths)
    paths = paths[args.skip : args.skip + args.n]

    for v in VARIANTS:
        (args.out_root / v).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed + args.skip)
    widths = []
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        widths.append(img.width)
        name = f"img{i:04d}.png"
        img.save(args.out_root / "orig" / name)
        for s in BLUR_SIGMAS:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=s))
            blurred.save(args.out_root / f"blur{s}" / name)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10)
        Image.open(buf).convert("RGB").save(args.out_root / "jpeg10" / name)
        arr = np.asarray(img, dtype=np.float32)
        noisy = np.clip(arr + rng.normal(0, 25, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(args.out_root / "noise25" / name)

    (args.out_root / "sources.txt").write_text("\n".join(str(p) for p in paths) + "\n")
    print(f"wrote {len(paths)} images x {len(VARIANTS)} variants to {args.out_root}")
    print(f"median width: {int(np.median(widths))} px")


if __name__ == "__main__":
    main()
