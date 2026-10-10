# The blur score

This note describes the blur score that `beaker quality` reports from version 0.6.0, why the earlier score was replaced, and how the replacement was chosen and tested. The scripts in this folder reproduce every table.

## What the score is

`global_blur_score` runs from 0 to 100. 0 means sharp and 100 means no fine detail is left.

It is computed on the full-resolution image:

1. Convert the image to grey.
2. Blur it lightly (Gaussian, 1 px) to suppress pixel noise.
3. Sum the squared Sobel gradient over all pixels. Call this the edge energy `E`.
4. Blur the result of step 2 again (Gaussian, 2 px) and sum the edge energy again, giving `E_reblurred`.
5. Take the ratio `r = E_reblurred / E`.
6. Report `100 * (r - 0.15) / 0.85`, clamped to 0..100.

A sharp image loses most of its edge energy in step 4, so `r` is small. An image that is already blurry has little fine detail to lose, so `r` is close to 1. `E` and `E_reblurred` both grow with the amount and the contrast of the edges in the scene, so the ratio depends on sharpness much more than on content.

An image with no edges at all (a single flat colour) gets `r = 1` and a score of 100.

The code is `relative_blur_ratio` and `blur_score_from_ratio` in [blur_detection.rs](../../beaker/src/blur_detection.rs). [reference.py](reference.py) holds the same formula in Python.

Typical values, from 200 test images about 500 px wide (the test sets are described below):

| Image | 5th percentile | Median | 95th percentile |
|---|---|---|---|
| Original | 10.9 | 26.8 | 42.6 |
| Blurred, 1 px | 24.2 | 39.5 | 54.2 |
| Blurred, 2 px | 48.7 | 60.8 | 71.2 |
| Blurred, 4 px | 76.9 | 82.7 | 87.6 |
| Blurred, 8 px | 92.5 | 94.2 | 96.0 |

### Fields that use it

- `quality.global_blur_score`: the score above.
- `quality.global_quality_score`: `global_paq2piq_score * max(1 - alpha * global_blur_score / 100, 0.2)`, with `alpha` 0.7 by default. The formula is unchanged from 0.5 apart from the division by 100; the blur score that goes into it is new.
- `quality_refined.crop_blur_score` and `crop_final_quality_score` on each detection, when `beaker detect` runs with `--refine-detection-quality`. These are the same two scores computed on a crop of the detection box.

### Fields that do not use it

`local_fused_probability` and `local_blur_weights` are 20×20 grids. They still come from the earlier grid measure, described in the next section, and so do the heatmaps and the per-detection `roi_blur_probability_mean` and `roi_blur_weight_mean`. The `--beta` and `--tau` options of `beaker quality` shape these grids and no longer affect any global score.

## The earlier score and what was wrong with it

Up to 0.5, the blur score was the mean of the `local_fused_probability` grid. That grid is built as follows. The image is resized to 224×224 and divided into 20×20 cells. For each cell the mean squared Sobel gradient (the Tenengrad measure) is compared with a fixed threshold `tau`, giving a "blur probability" `(tau / (t + tau)) ^ beta` that is high when the cell has little edge energy. The same is done on a 112×112 copy, and the two probabilities are combined.

On the test described below, this score always rose when an image was blurred, but it separated blurred images from sharp ones poorly when the images were different photos. With a 2 px blur, a blurred image scored blurrier than a different, unblurred image only 61% of the time, where 50% is chance.

To find out why, [reference.py](reference.py) re-implements the grid score in Python and changes one design choice at a time. The re-implementation agrees with beaker 0.5.0 to within 0.002 (on the old 0..1 scale) on all 1,400 images of the first test set.

Each cell of the table is the chance that a degraded image scores blurrier than a different original image (the AUC, defined under Test method). The last column uses each original at half contrast. Halving the contrast does not change sharpness, so a blur score should not move and 0.50 is the ideal value there.

| Score | Blur 1 px | Blur 2 px | Blur 4 px | Blur 8 px | JPEG q10 | Noise | Half contrast |
|---|---|---|---|---|---|---|---|
| Grid score (beaker 0.5) | 0.53 | 0.61 | 0.77 | 0.94 | 0.47 | 0.33 | 0.76 |
| 224 px scale only | 0.54 | 0.62 | 0.77 | 0.93 | 0.46 | 0.22 | 0.72 |
| 112 px scale only | 0.51 | 0.55 | 0.65 | 0.81 | 0.47 | 0.39 | 0.73 |
| Sharpest 10% of cells | 0.63 | 0.82 | 0.97 | 0.99 | 0.50 | 0.48 | 0.95 |
| Sharpest 10% of cells, 224 px scale only | 0.67 | 0.88 | 0.98 | 1.00 | 0.50 | 0.49 | 0.95 |
| The same at native resolution | 0.83 | 0.97 | 0.99 | 1.00 | 0.54 | 0.41 | 0.91 |
| Relative measure, native resolution (beaker 0.6) | 0.81 | 0.99 | 1.00 | 1.00 | 0.50 | 0.32 | 0.50 |

"Sharpest 10% of cells" replaces the mean over the grid with its 10th percentile. The next row does the same on the 224 px edge energies alone, and the row after that computes those cell energies on the full-size image instead of the 224 px resize, changing nothing else.

The table points to five causes:

1. **The score compared edge energy with a fixed threshold.** The grid score of the 200 unblurred images had a standard deviation of 0.20 (on its 0..1 scale). A 2 px blur raised an image's score by a median of 0.065. Differences between scenes were larger than the effect of blur. Contrast alone moved it too: every half-contrast copy scored blurrier than its original.
2. **It averaged all 400 cells.** A sharp subject in front of a smooth background is mostly low-energy cells and scored as blurry. Using the sharpest tenth of the cells raised the 2 px result from 0.61 to 0.82.
3. **It worked on a 224 px resize.** The resize removes fine detail before anything is measured. Moving the same calculation to native resolution raised the 1 px result from 0.67 to 0.83 and the 2 px result from 0.88 to 0.97, on images only 500 px wide. The loss grows with image size (see the camera-resolution photo below).
4. **The second scale did not help with blur.** The 224 px scale alone gave the same blur results as the fused score, to within 0.01.
5. **Noise counted as detail.** beaker 0.5.0 scored the noisy copy as sharper than its original in 199 of the 200 images.

## How the replacement was chosen

Causes 1 and 2 both come from comparing edge energy with an absolute threshold. A ratio of two edge energies from the same image needs no threshold. Blurring the image again and comparing edge energy before and after is one way to form such a ratio; the same idea is behind the no-reference blur metric of Crete et al. (2007). The ratio asks how much of the image's own edge energy is fine detail.

Fixing causes 2 and 3 without a ratio also works for blur: the "same at native resolution" row separates blurred images about as well as the relative measure. It is still an absolute measure, though, and its last column is 0.91: a sharp, low-contrast photo scores as blurry. The relative measure is at 0.50 there. That is the reason for preferring it. The comparison between these two rows was made after the relative measure had been chosen, and it is reported because it is the closest alternative.

The two blur widths, 1 px and 2 px, were picked before any test was run. The table below was computed afterwards to check that the choice does not sit on a narrow optimum. Cells are AUC on the first set.

| Denoise (px) | Re-blur (px) | Blur 1 px | Blur 2 px | Blur 4 px | Noise |
|---|---|---|---|---|---|
| 0 | 1 | 0.97 | 1.00 | 1.00 | 0.11 |
| 0 | 2 | 0.93 | 1.00 | 1.00 | 0.17 |
| 0.5 | 2 | 0.90 | 1.00 | 1.00 | 0.20 |
| 1 | 1 | 0.88 | 1.00 | 1.00 | 0.25 |
| **1** | **2** | **0.81** | **0.99** | **1.00** | **0.32** |
| 1 | 4 | 0.73 | 0.97 | 1.00 | 0.37 |
| 2 | 2 | 0.66 | 0.93 | 1.00 | 0.43 |

Less denoising detects a 1 px blur better and rewards noise more. The widths were left at 1 and 2. The test sets contain no real sensor noise, so they give no basis for trading noise robustness away.

The constant 0.15 in step 6 sets which ratio maps to a score of 0. Across the 1,400 images of the first test set the ratio ranged from 0.17 to 0.97, so 0.15 puts the sharpest images just above 0. The upper end needs no constant, because the ratio does not exceed 1.

## Test method

**Images.** 200 images drawn at random from CUB-200-2011, median width 500 px. [make_variants.py](make_variants.py) saves each one with six degraded copies: Gaussian blur of 1, 2, 4 and 8 px, JPEG compression at quality 10, and added Gaussian noise (standard deviation 25 on the 0..255 scale). All copies are stored as PNG, so the degradation is the only difference from the original.

**Two sets.** The first set (`--seed 0`) was used to diagnose the old score and to choose the new one, including its three constants. The second set (`--seed 0 --skip 200`, the next 200 images of the same shuffle) shares no image with the first. It was scored once, after code review and after the full test suite passed on the final code.

**Statistics.** Two numbers are reported for each score and degradation:

- *Paired*: the share of images whose degraded copy scores worse than its own original. This asks whether the score moves in the right direction.
- *AUC*: the chance that a degraded image scores worse than a different, original image. 0.5 is a coin flip and 1.0 is perfect separation. Pairs of an original with its own copy are left out. This asks whether the score can rank different photos, which is what a quality score is used for.

[evaluate.py](evaluate.py) computes both from beaker's `.beaker.toml` output.

## Results

### Global blur score

| Set | | Blur 1 px | Blur 2 px | Blur 4 px | Blur 8 px | JPEG q10 | Noise |
|---|---|---|---|---|---|---|---|
| First | AUC, beaker 0.5.0 | 0.53 | 0.61 | 0.77 | 0.94 | 0.47 | 0.33 |
| First | AUC, beaker 0.6.0 | 0.81 | 0.99 | 1.00 | 1.00 | 0.50 | 0.32 |
| Second | AUC, beaker 0.5.0 | 0.53 | 0.60 | 0.76 | 0.93 | 0.48 | 0.33 |
| Second | AUC, beaker 0.6.0 | 0.84 | 1.00 | 1.00 | 1.00 | 0.50 | 0.30 |
| First | Paired, beaker 0.5.0 | 100% | 100% | 100% | 100% | 5% | 0.5% |
| First | Paired, beaker 0.6.0 | 100% | 100% | 100% | 100% | 53% | 0% |
| Second | Paired, beaker 0.5.0 | 100% | 100% | 100% | 100% | 5% | 1% |
| Second | Paired, beaker 0.6.0 | 100% | 100% | 100% | 100% | 56% | 0% |

The held-out second set gives the same picture as the first. Its score ranges also match: unblurred images had a median of 25.9 (5th to 95th percentile 11.5 to 40.1), and the 2 px copies a median of 59.2.

The Rust implementation and [reference.py](reference.py) agree on all 2,800 images of the two sets, with a largest difference below 0.0001 points.

### Combined quality score

`global_quality_score` multiplies the PaQ-2-PiQ score by a weight that falls as the blur score rises. Cells are AUC; PaQ-2-PiQ alone is shown for comparison.

| Set | Score | Blur 1 px | Blur 2 px | Blur 4 px | Blur 8 px | JPEG q10 | Noise |
|---|---|---|---|---|---|---|---|
| First | PaQ-2-PiQ alone | 0.65 | 0.94 | 1.00 | 1.00 | 0.70 | 0.75 |
| First | Quality score, beaker 0.5.0 | 0.56 | 0.73 | 0.95 | 1.00 | 0.51 | 0.39 |
| First | Quality score, beaker 0.6.0 | 0.83 | 0.99 | 1.00 | 1.00 | 0.58 | 0.45 |
| Second | PaQ-2-PiQ alone | 0.64 | 0.92 | 1.00 | 1.00 | 0.68 | 0.74 |
| Second | Quality score, beaker 0.5.0 | 0.56 | 0.71 | 0.94 | 1.00 | 0.51 | 0.39 |
| Second | Quality score, beaker 0.6.0 | 0.86 | 1.00 | 1.00 | 1.00 | 0.58 | 0.44 |

With the new blur score, the quality score separates blurred images better than PaQ-2-PiQ does alone. For compression and noise it is weaker than PaQ-2-PiQ alone. The blur term carries most of the variation between images, and it does not penalise either degradation: compression leaves the blur score unchanged on average, and noise lowers it. With noise, the quality score of beaker 0.6.0 rated the noisy copy below its original in 28% of the first set and 24.5% of the second. Lowering `--alpha` gives PaQ-2-PiQ more weight.

### A camera-resolution photo

One 5184×3888 photo (a converted camera raw file, not part of the repository) was blurred at native resolution, and the score was computed on the full image and on downsized copies. `reference.py --photo` produces this table.

| Blur (px, native) | Native | 2592 px wide | 1296 px wide | 648 px wide |
|---|---|---|---|---|
| 0 | 21.9 | 23.1 | 23.2 | 17.7 |
| 1 | 34.8 | 27.3 | 24.2 | 17.9 |
| 2 | 58.9 | 37.9 | 27.1 | 18.6 |
| 4 | 84.0 | 60.4 | 36.4 | 21.2 |
| 8 | 94.4 | 82.9 | 57.2 | 30.2 |

At native resolution a 1 px blur moves the score by 13 points. On a copy 1296 px wide the same blur moves it by 1 point. This is why the score is computed at full resolution. beaker 0.6.0 gives 21.9 for the unblurred photo, the same as the reference.

### Per-detection crop score

`beaker detect --refine-detection-quality` computes the same score on a padded crop around bird and head boxes of at least 64×64 px. [evaluate_detect.py](evaluate_detect.py) takes the highest-confidence head in each image and compares it with the highest-confidence head in each blurred copy of that image.

| Set | | Original | Blur 1 px | Blur 2 px | Blur 4 px | Blur 8 px |
|---|---|---|---|---|---|---|
| First | Heads found | 200 | 198 | 189 | 156 | 59 |
| First | Heads with a crop score | 124 | 153 | 181 | 150 | 59 |
| First | Median `crop_blur_score` | 30.0 | 43.1 | 62.6 | 83.2 | 94.2 |
| First | Pairs with a crop score in both | | 123 | 122 | 110 | 50 |
| First | Crop score worse in the blurred copy | | 100% | 100% | 100% | 100% |
| Second | Heads found | 199 | 199 | 190 | 167 | 61 |
| Second | Heads with a crop score | 139 | 162 | 179 | 160 | 61 |
| Second | Median `crop_blur_score` | 30.4 | 43.5 | 62.4 | 83.1 | 94.4 |
| Second | Pairs with a crop score in both | | 139 | 136 | 128 | 55 |
| Second | Crop score worse in the blurred copy | | 100% | 100% | 100% | 100% |

The medians in each column are over the heads that have a crop score in that column, which are not the same heads from column to column. The paired rows compare like with like.

The `triage_decision` field on each detection is not based on the blur score. Of the heads rated `good` in the original, between 78% and 97% were still rated `good` in the blurred copy, depending on the set and the blur. That is a separate, open issue.

### Cost

Wall time of the whole `beaker quality` command, on an Apple M5 Max with `--device cpu`, repeated five to seven times with a spread of 0.01 s on the single photo:

| Input | beaker 0.5.0 | beaker 0.6.0 |
|---|---|---|
| One 5184×3888 JPEG | 0.26 s | 0.35 s |
| 200 PNGs, median 500×375 | 2.9 s | 3.5 s |

The measure runs on one thread. It holds two extra buffers of 4 bytes per pixel each while it runs, about 160 MB for a 20 megapixel image.

## Limits

- The blur in these tests is synthetic and Gaussian, which a re-blur ratio suits well. Real defocus and motion blur were not tested, and there are no human sharpness ratings behind any number here.
- Only one camera-resolution photo was tested.
- Noise counts as fine detail: the noisy copy scored as sharper than its original in every test image. Heavy JPEG compression moved the score up about as often as down. PaQ-2-PiQ responds to both.
- The score is in pixels of the image as given. The same photo saved at half size scores differently, so scores are comparable between images of similar resolution.
- Transparency is ignored. In an image with an alpha channel, such as a cutout, the edge between the subject and the transparent area counts as sharp detail and lowers the score.
- A nearly featureless image holds little besides pixel noise and rounding steps, so it can score as sharp. Only an exactly flat image scores 100.
- Unblurred images still spread from about 11 to 43, because scenes differ in how their edge energy divides between fine and coarse detail.
- The 20×20 grids, the heatmaps and the per-detection `roi_*` blur values still use the grid measure, with the weaknesses listed above.

## Reproducing the tables

The scripts declare their dependencies inline, so `uv run` needs no setup. `$CUB` is the `images` folder of CUB-200-2011. Run the same commands with `second` for the held-out set, and with a beaker 0.5.0 binary for the comparison rows.

```bash
# Test sets: first (development) and second (held-out)
uv run make_variants.py $CUB sets/first --seed 0
uv run make_variants.py $CUB sets/second --seed 0 --skip 200

# Global scores
for v in orig blur1 blur2 blur4 blur8 jpeg10 noise25; do
  beaker quality sets/first/$v --metadata --output-dir out/first/$v --device cpu
done
uv run evaluate.py out/first

# Ablation, width sensitivity, agreement with beaker, camera-resolution photo
uv run reference.py sets/first --beaker-out out/first --photo photo.jpg

# Per-detection crop scores
for v in orig blur1 blur2 blur4 blur8; do
  beaker detect sets/first/$v --metadata --refine-detection-quality \
    --output-dir det/first/$v --device cpu
done
uv run evaluate_detect.py det/first

# Cost
time beaker quality photo.jpg --metadata --output-dir /tmp/timing --device cpu
time beaker quality sets/first/orig --metadata --output-dir /tmp/timing --device cpu
```
