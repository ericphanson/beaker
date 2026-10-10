# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Summarise beaker's global quality scores over a test set from make_variants.py.

Expects <out_root>/<variant>/*.beaker.toml, written by e.g.

    beaker quality <test_set>/<variant> --metadata --output-dir <out_root>/<variant>

For each score and each degradation it prints:

  paired  share of images whose degraded copy scores worse than the original
  AUC     chance that a degraded image scores worse than a different, original
          image (0.5 is a coin flip, 1.0 is perfect separation)

usage: uv run evaluate.py <out_root>
"""

import sys
import tomllib
from pathlib import Path

import numpy as np

# score -> True if a higher value means a better image
SCORES = {
    "global_paq2piq_score": True,
    "global_blur_score": False,
    "global_quality_score": True,
}
ORDER = ["orig", "blur1", "blur2", "blur4", "blur8", "jpeg10", "noise25"]


def load(variant_dir):
    rows = {}
    for f in sorted(variant_dir.glob("*.beaker.toml")):
        rows[f.name] = tomllib.loads(f.read_text())["quality"]
    return rows


def auc_worse(orig, degraded):
    """Chance a degraded image has a lower value than a different original.

    Both arrays are in the same image order; pairs of an original with its own
    degraded copy are left out.
    """
    o, d = np.asarray(orig), np.asarray(degraded)
    worse = (d[None, :] < o[:, None]) + 0.5 * (d[None, :] == o[:, None])
    return worse[~np.eye(len(o), dtype=bool)].mean()


def main():
    out_root = Path(sys.argv[1])
    data = {v: load(out_root / v) for v in ORDER if (out_root / v).is_dir()}
    names = sorted(data["orig"])
    versions = {
        q.get("execution", {}).get("beaker_version") for q in data["orig"].values()
    }
    print(f"n = {len(names)} images, beaker_version = {sorted(versions)}")

    for score, higher_is_better in SCORES.items():
        sign = 1 if higher_is_better else -1
        orig = np.array([data["orig"][n][score] for n in names])
        print(f"\n== {score} ==")
        print(f"{'variant':<9}{'p5':>8}{'median':>8}{'p95':>8}{'paired':>9}{'AUC':>7}")
        for v, rows in data.items():
            x = np.array([rows[n][score] for n in names])
            p5, med, p95 = np.percentile(x, [5, 50, 95])
            line = f"{v:<9}{p5:>8.2f}{med:>8.2f}{p95:>8.2f}"
            if v != "orig":
                paired = (sign * x < sign * orig).mean()
                line += f"{paired:>9.1%}{auc_worse(sign * orig, sign * x):>7.2f}"
            print(line)


if __name__ == "__main__":
    main()
