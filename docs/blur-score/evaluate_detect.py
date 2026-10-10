# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Summarise per-detection quality for the head class over a test set.

Expects <out_root>/<variant>/*.beaker.toml, written by e.g.

    beaker detect <test_set>/<variant> --metadata --refine-detection-quality \
        --output-dir <out_root>/<variant>

Uses the highest-confidence head in each image; heads are matched between an
original and its blurred copy by image, not by box. The crop scores come from
`quality_refined`, where beaker re-runs quality on a padded crop of each box
of at least 64x64 px.

usage: uv run evaluate_detect.py <out_root>
"""

import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

ORDER = ["orig", "blur1", "blur2", "blur4", "blur8"]
CROP_SCORES = {"crop_paq2piq_global": True, "crop_blur_score": False}


def top_head(path):
    dets = tomllib.loads(path.read_text()).get("detect", {}).get("detections", [])
    heads = [d for d in dets if d.get("class_name") == "head" and "quality" in d]
    return max(heads, key=lambda d: d["confidence"]) if heads else None


def main():
    out_root = Path(sys.argv[1])
    data = {}
    for v in ORDER:
        if (out_root / v).is_dir():
            files = sorted((out_root / v).glob("*.beaker.toml"))
            data[v] = {f.name: top_head(f) for f in files}
    names = sorted(data["orig"])
    print(f"n = {len(names)} images; one head (highest confidence) per image\n")

    print(
        f"{'variant':<8}{'heads':>6}{'good':>6}{'unknown':>9}{'bad':>5}"
        f"{'refined':>9}{'crop paq2piq':>14}{'crop blur':>11}"
    )
    for v, rows in data.items():
        heads = [h for h in rows.values() if h]
        c = Counter(h["quality"]["triage_decision"] for h in heads)
        # beaker refines only boxes of at least 64x64 px
        refined = [h["quality_refined"] for h in heads if "quality_refined" in h]
        med = [
            np.median([r[k] for r in refined]) if refined else np.nan
            for k in CROP_SCORES
        ]
        print(
            f"{v:<8}{len(heads):>6}{c['good']:>6}{c['unknown']:>9}{c['bad']:>5}"
            f"{len(refined):>9}{med[0]:>14.1f}{med[1]:>11.1f}"
        )

    print("\nTop head of the original against top head of its blurred copy:")
    print(
        f"{'variant':<8}{'pairs':>6}{'good stays good':>17}"
        f"{'refined pairs':>15}{'crop paq2piq worse':>20}{'crop blur worse':>17}"
    )
    for v in [v for v in data if v != "orig"]:
        pairs = [
            (data["orig"][n], data[v][n])
            for n in names
            if data["orig"][n] and data[v].get(n)
        ]
        goods = [(a, b) for a, b in pairs if a["quality"]["triage_decision"] == "good"]
        stays = np.mean([b["quality"]["triage_decision"] == "good" for _, b in goods])
        refined = [
            (a["quality_refined"], b["quality_refined"])
            for a, b in pairs
            if "quality_refined" in a and "quality_refined" in b
        ]
        worse = []
        for key, higher_is_better in CROP_SCORES.items():
            sign = 1 if higher_is_better else -1
            worse.append(np.mean([sign * b[key] < sign * a[key] for a, b in refined]))
        print(
            f"{v:<8}{len(pairs):>6}{stays:>17.1%}"
            f"{len(refined):>15}{worse[0]:>20.1%}{worse[1]:>17.1%}"
        )


if __name__ == "__main__":
    main()
