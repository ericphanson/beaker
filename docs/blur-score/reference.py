# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""Python reference for beaker's blur scores, used to choose and check the formula.

It re-implements the grid score used up to beaker 0.5 and the relative measure
used from 0.6 (beaker/src/blur_detection.rs), then prints:

  1. an ablation of the grid score, changing one design choice at a time
     (the "lowcon" column is the original at half contrast, where 0.50 is ideal)
  2. how the relative measure responds to its two blur widths
  3. (with --beaker-out) agreement between this file and beaker's own output,
     for output from either an old or a new beaker
  4. (with --photo) the relative measure on one photo at several working sizes

For every score here, higher means blurrier.

usage: uv run reference.py <test_set> [--beaker-out DIR] [--photo PATH]
"""

import argparse
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Grid score (beaker <= 0.5)
TAU, BETA, BIAS112 = 0.02, 1.2, 1.25
# Relative measure (beaker >= 0.6)
DENOISE_SIGMA, REBLUR_SIGMA, RATIO_SHARP = 1.0, 2.0, 0.15


def gray_of(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sobel_g2(g):
    """Squared Sobel gradient per pixel, edges replicated."""
    p = np.pad(g, 1, mode="edge")
    left = p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
    right = p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
    up = p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
    down = p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
    return (right - left) ** 2 + (down - up) ** 2


def gauss(g, sigma):
    """Separable Gaussian blur truncated at 3 sigma, edges replicated."""
    if sigma <= 0:
        return g
    r = round(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    h, w = g.shape
    p = np.pad(g, r, mode="edge")
    rows = sum(kw * p[i : i + h, :] for i, kw in enumerate(k))
    return sum(kw * rows[:, i : i + w] for i, kw in enumerate(k))


def cell_means(x, n=20):
    h, w = x.shape
    rb = [(i * h) // n for i in range(n + 1)]
    cb = [(i * w) // n for i in range(n + 1)]
    return np.array(
        [
            [x[rb[i] : rb[i + 1], cb[j] : cb[j + 1]].mean() for j in range(n)]
            for i in range(n)
        ]
    )


def relative_ratio(gray, denoise=DENOISE_SIGMA, reblur=REBLUR_SIGMA):
    """Share of edge energy that survives re-blurring (0..1)."""
    base = gauss(gray, denoise)
    energy = float(sobel_g2(base).sum())
    energy_reblurred = float(sobel_g2(gauss(base, reblur)).sum())
    eps = 1e-7 * gray.size
    return min((energy_reblurred + eps) / (energy + eps), 1.0)


def blur_score_from_ratio(ratio):
    return 100.0 * float(np.clip((ratio - RATIO_SHARP) / (1.0 - RATIO_SHARP), 0.0, 1.0))


def grid_scores(img):
    """The grid score of beaker <= 0.5 and single-change variants of it."""
    g224 = gray_of(img.resize((224, 224), Image.LANCZOS))
    t224 = cell_means(sobel_g2(g224))
    g112 = g224.reshape(112, 2, 112, 2).mean(axis=(1, 3))
    t112 = cell_means(sobel_g2(g112))
    # Rust takes element len/2 of the sorted values (the upper median)
    upper_median = lambda t: float(np.sort(t.ravel())[t.size // 2])  # noqa: E731
    ratio = float(
        np.clip(upper_median(t112) / max(upper_median(t224), 1e-12), 0.05, 0.80)
    )
    p224 = (TAU / (t224 + TAU)) ** BETA
    tau112 = TAU * ratio * BIAS112
    p112 = (tau112 / (t112 + tau112)) ** BETA
    fused = 1 - (1 - p224) * (1 - p112)

    gray = gray_of(img)
    native_cells = cell_means(sobel_g2(gray))
    # Rows after the first three change one thing relative to the row named in the label
    return {
        "grid score (beaker <= 0.5)": fused.mean(),
        "  224 px scale only": p224.mean(),
        "  112 px scale only": p112.mean(),
        "sharpest 10% of cells": np.percentile(fused, 10),
        "sharpest 10% of cells, 224 px scale only": -np.log(
            np.percentile(t224, 90) + 1e-12
        ),
        "  same, at native resolution": -np.log(
            np.percentile(native_cells, 90) + 1e-12
        ),
        "relative measure, native (beaker >= 0.6)": relative_ratio(gray),
    }


def auc_blurrier(orig, degraded):
    """Chance a degraded image scores higher than a different original.

    Both arrays are in the same image order; pairs of an original with its own
    degraded copy are left out.
    """
    o, d = np.asarray(orig), np.asarray(degraded)
    blurrier = (d[None, :] > o[:, None]) + 0.5 * (d[None, :] == o[:, None])
    return blurrier[~np.eye(len(o), dtype=bool)].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test_set", type=Path)
    p.add_argument("--beaker-out", type=Path)
    p.add_argument("--photo", type=Path)
    args = p.parse_args()

    variants = [
        d.name
        for d in sorted(args.test_set.iterdir())
        if d.is_dir() and d.name != "orig"
    ]
    names = sorted(f.name for f in (args.test_set / "orig").glob("*.png"))
    widths = [(dn, rb) for dn in (0.0, 0.5, 1.0, 2.0) for rb in (1.0, 2.0, 4.0)]

    # One pass over the images; keep only the scores. "lowcon" is each original with
    # its contrast halved: sharpness is unchanged, so a blur score should not move.
    def load(v, n):
        if v == "lowcon":
            img = Image.open(args.test_set / "orig" / n).convert("RGB")
            return ImageEnhance.Contrast(img).enhance(0.5)
        return Image.open(args.test_set / v / n)

    on_disk = ["orig", *variants]
    variants.append("lowcon")
    grid, ratio = {}, {}
    for v in ["orig", *variants]:
        grid[v], ratio[v] = [], []
        for n in names:
            img = load(v, n)
            grid[v].append(grid_scores(img))
            gray = gray_of(img)
            ratio[v].append({c: relative_ratio(gray, *c) for c in widths})
    header = f"{'':<44}" + "".join(f"{v:>9}" for v in variants)

    print(f"1. ablation of the grid score, n = {len(names)}")
    keys = list(grid["orig"][0])
    print("AUC: chance a degraded image scores blurrier than a different original")
    print(header)
    for k in keys:
        o = [r[k] for r in grid["orig"]]
        cells = (auc_blurrier(o, [r[k] for r in grid[v]]) for v in variants)
        print(f"{k:<44}" + "".join(f"{c:>9.2f}" for c in cells))
    print(
        "paired: share of images whose degraded copy scores blurrier than the original"
    )
    print(header)
    for k in keys:
        o = np.array([r[k] for r in grid["orig"]])
        cells = ((np.array([r[k] for r in grid[v]]) > o).mean() for v in variants)
        print(f"{k:<44}" + "".join(f"{c:>9.0%}" for c in cells))
    o = np.array([r[keys[0]] for r in grid["orig"]])
    print(
        f"grid score: spread between originals (std) = {o.std():.3f}; median rise per image:"
    )
    for v in [v for v in variants if v.startswith("blur")]:
        rise = np.median(np.array([r[keys[0]] for r in grid[v]]) - o)
        print(f"   {v}: {rise:.3f}")

    print("\n2. relative measure: AUC by (denoise, re-blur) width in px")
    print(f"{'denoise':>8}{'re-blur':>8}" + "".join(f"{v:>9}" for v in variants))
    for c in widths:
        o = [r[c] for r in ratio["orig"]]
        cells = (auc_blurrier(o, [r[c] for r in ratio[v]]) for v in variants)
        print(f"{c[0]:>8}{c[1]:>8}" + "".join(f"{x:>9.2f}" for x in cells))
    chosen = (DENOISE_SIGMA, REBLUR_SIGMA)
    print(f"raw ratio at the chosen widths {chosen}:")
    print(f"{'variant':<9}{'min':>7}{'p5':>7}{'median':>8}{'p95':>7}{'max':>7}")
    for v, rows in ratio.items():
        x = np.array([r[chosen] for r in rows])
        q = np.percentile(x, [5, 50, 95])
        print(
            f"{v:<9}{x.min():>7.2f}{q[0]:>7.2f}{q[1]:>8.2f}{q[2]:>7.2f}{x.max():>7.2f}"
        )

    if args.beaker_out:
        # beaker <= 0.5 wrote the grid score (0..1); later versions write the relative measure
        mine, theirs, versions = [], [], set()
        for v in on_disk:
            for i, n in enumerate(names):
                toml = args.beaker_out / v / n.replace(".png", ".beaker.toml")
                q = tomllib.loads(toml.read_text())["quality"]
                version = q["execution"]["beaker_version"]
                versions.add(version)
                theirs.append(q["global_blur_score"])
                if tuple(int(x) for x in version.split(".")[:2]) < (0, 6):
                    mine.append(grid[v][i][keys[0]])
                else:
                    mine.append(blur_score_from_ratio(ratio[v][i][chosen]))
        diff = np.abs(np.array(mine) - np.array(theirs))
        r = np.corrcoef(mine, theirs)[0, 1]
        print(f"\n3. agreement with global_blur_score from beaker {sorted(versions)}")
        print(
            f"   {len(mine)} images: r = {r:.5f}, max abs difference = {diff.max():.4f}"
        )

    if args.photo:
        photo = Image.open(args.photo).convert("RGB")
        w, h = photo.size
        print(f"\n4. {args.photo.name} ({w}x{h}): blur applied at native size,")
        print("   blur score computed at several working sizes")
        sizes = [None, w // 2, w // 4, w // 8]
        labels = ["native"] + [f"{s} px wide" for s in sizes[1:]]
        print(f"{'blur sigma (px)':<17}" + "".join(f"{lab:>14}" for lab in labels))
        for sigma in (0, 1, 2, 4, 8, 16):
            im = (
                photo
                if sigma == 0
                else photo.filter(ImageFilter.GaussianBlur(radius=sigma))
            )
            row = []
            for s in sizes:
                small = (
                    im if s is None else im.resize((s, round(h * s / w)), Image.LANCZOS)
                )
                row.append(blur_score_from_ratio(relative_ratio(gray_of(small))))
            print(f"{sigma:<17}" + "".join(f"{x:>14.1f}" for x in row))


if __name__ == "__main__":
    main()
