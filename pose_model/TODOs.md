### 1. Missing detector-centric **affine crop**

* **How VHR-BirdPose was trained**
  A bird detector first produced a bounding box. The dataloader then built an *affine transform* that:
    1. Recentred the bird (`center = [x + w/2, y + h/2]`).
    2. Expanded the box by a constant factor (≈ 1.25×) so the whole head is visible.
    3. Warped the square to the fixed network input size (256 × 256) with `cv2.warpAffine`.
  This guarantees that at inference each key-point spans 6–15 pixels on the 64 × 64 heat-maps.

* **What your pipeline does**
  Down-scales the *entire* 4000 × 3000 park photo to 256 × 256, then pads it. The bird’s head shrinks to a handful of pixels, leaving almost no signal for the heat-map decoder.

* **Why accuracy collapses**
  Key-point detectors are extremely scale-sensitive: PCK drops > 15 pp when the head diameter falls below ~25 input pixels. Without the crop, the network is simply off-distribution.

---

### 2. **ImageNet mean / std normalisation omitted**

* **Expected by the checkpoint**
  Inputs were standardised with channel-wise statistics:
  ```
  mean = [0.485, 0.456, 0.406] × 255
  std  = [0.229, 0.224, 0.225] × 255
  ```
  after RGB conversion. During training the optimiser learned weights under that distribution.

* **Current behaviour**
  Pixels are merely divided by 255, so the mean shifts from ~0 to ~-2 in the normalised space and the dynamic range shrinks by ~4× relative to what BatchNorm and LayerNorm layers expect.

* **Observed effect**
  Heat-map activations become flat (confidence ≈ 0.2 everywhere), the arg-max is noisy, and PCK drops another 5–10 pp even if the crop size were correct.

---

### 3. **Incorrect coordinate unwarping**

* **Ground truth pipeline**
  The same affine matrix `M` used for cropping is inverted (`M⁻¹`) and applied to each heat-map peak:
  $begin:math:display$
  p_\\text{orig} = M^{-1} \\,[x_\\text{heat}\\,·s,\\,y_\\text{heat}\\,·s,\\,1]^⊤,
  $end:math:display$
  where *s* is the stride (4 for 256 → 64). This recovers sub-pixel-accurate positions in the *original* image frame regardless of padding, aspect ratio, or detector location.

* **Current heuristic**
  Assumes a fixed stride of 4, subtracts padding offsets derived from the resized dimensions, and scales by the preprocessing scale factor. Any small rounding error in the padding or a non-square box skews the final coordinates; rotation in the affine (e.g., a tilted crop) is ignored completely.

* **Practical fallout**
  Even when the network predicts a perfect heat-map, the decoded point can land several dozen pixels away—often outside the head—making qualitative results look like the model “missed” when it was the math that was wrong.
