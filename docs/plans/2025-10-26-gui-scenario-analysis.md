# GUI Quality Triage: Real-time Slider UX Analysis

**Date:** 2025-10-26
**Context:** Quality slider tuning for folder of 100 images

## Scenario

**GUI Layout:**
- Main view: Top-ranked image with heatmap overlay
- Sidebar: 3-5 thumbnail previews of other high-quality images
- Controls: Sliders for ALPHA, BETA, TAU_TEN_224, MIN_WEIGHT, etc.

**User Action:**
User adjusts a slider (e.g., BETA from 1.2 to 1.5)

**Required Response:**
1. Recompute quality scores for all 100 images
2. Re-rank images to find new top image
3. Display heatmap for (potentially new) top image
4. Update thumbnails for 3-5 sidebar images

**Target:** Sub-50ms response for "instant" feel

---

## Timing Analysis

### Measured Performance (from benchmarks)

**Per-image breakdown:**
- Preprocessing: 24-31ms (parameter-independent)
- ONNX inference: 29-31ms (parameter-independent)
- Blur detection: 2.1-2.3ms (parameter-dependent)
- Postprocessing: <0.1ms (parameter-dependent)

**Heatmap generation:**
- 13 PNG files (224×224): 35-38ms
- Per-file estimate: ~2.8ms per PNG

### Critical Insight

**Heuristic recomputation from cache:**
- Parameters only affect blur mapping and final combination
- Raw Tenengrad values (t224, t112) are parameter-independent
- Recomputing probabilities/weights from cached Tenengrad: **<0.1ms per image**
- **100 images × 0.1ms = 10ms total** ← INSTANT!

### The Challenge

Heatmaps ARE parameter-dependent:
```rust
// Blur probability depends on BETA, TAU_TEN_224, P_FLOOR
let p = (tau / (t + tau)).powf(BETA);

// Weights depend on ALPHA, MIN_WEIGHT
let w = (1.0 - ALPHA * p).clamp(MIN_WEIGHT, 1.0);
```

So we **cannot** pre-cache heatmap images. They must be re-rendered when sliders change.

---

## Solution: Multi-Level Caching with Smart Visualization

### Three-Level Cache Architecture

**Level 1: Expensive ML Operations (Parameter-Independent)**
```rust
struct Level1Cache {
    // Per image
    preprocessed_tensor: Array4<f32>,     // 224×224×3 RGB tensor
    onnx_output: Paq2PiqOutput,            // 401 values (1 global + 400 local)
    blur_raw: BlurRawData,                 // Raw Tenengrad: t224, t112, median
}

// Memory: ~600KB per image × 100 = 60MB total
// Computed: On initial load (57-70ms per image)
// Invalidated: Never (unless file changes)
```

**Level 2: Heuristic Scores (Parameter-Dependent, Cheap)**
```rust
struct Level2Cache {
    // Computed from Level 1 on every slider change
    blur_probability: Array2<f32>,         // 20×20 grid
    blur_weights: Array2<f32>,             // 20×20 grid
    quality_score: f32,                    // Final score for ranking
}

// Memory: ~2KB per image × 100 = 200KB total
// Computed: On slider change (<0.1ms per image, 10ms for 100)
// Invalidated: When any parameter changes
```

**Level 3: Rendered Visualizations (Parameter-Dependent, Moderate Cost)**
```rust
struct Level3Cache {
    // Only for currently visible images
    heatmap_pfused: ImageBuffer,           // Fused blur probability heatmap
    heatmap_weights: ImageBuffer,          // Blur weights heatmap
    overlay_pfused: ImageBuffer,           // Heatmap overlaid on original
}

// Memory: ~600KB per visible image × 10 = 6MB typical
// Computed: On demand when image visible (5-10ms per image)
// Invalidated: When parameters change or image not visible
```

---

## Optimal User Flow

### When User Adjusts Slider

**Phase 1: Instant Scoring (10ms)**
```
1. Recompute Level 2 cache for all 100 images
   - Compute blur probabilities from cached raw Tenengrad
   - Compute weights from probabilities
   - Compute final quality scores
   - Time: <0.1ms × 100 = 10ms

2. Re-rank images by quality score
   - Time: <1ms (simple sort)
```

**Phase 2: Main View Update (5-10ms)**
```
3. Check if top image changed
   - If same image: Check Level 3 cache
     - Cache hit: Display immediately (0ms)
     - Cache miss: Render heatmaps (5-10ms)
   - If different image: Render new top image's heatmaps (5-10ms)

4. Display main view
   - Update image if changed
   - Update heatmap overlay
```

**Phase 3: Sidebar Update (0-20ms, optional)**
```
5. Update sidebar thumbnails

   Option A: Show original thumbnails only
   - Time: 0ms (thumbnails pre-cached from originals)
   - Sufficient if just showing ranking

   Option B: Show small heatmap overlays
   - Render 3-5 small heatmaps (64×64 or 128×128)
   - Time: ~3-4ms each × 5 = 15-20ms
   - Better for visual triage
```

**Total Response Time:**
- Minimum: 10ms (scoring) + 0ms (cached viz) = **10ms**
- Typical: 10ms (scoring) + 5-10ms (main viz) = **15-20ms**
- Maximum: 10ms (scoring) + 10ms (main viz) + 20ms (sidebar) = **40ms**

All well under 50ms threshold for "instant" feel!

---

## Progressive Rendering Strategy

For best UX, use progressive updates:

```javascript
async function onSliderChange(newValue) {
    // Phase 1: Immediate (10ms)
    const newScores = recomputeAllScores(newValue);  // Level 2 cache
    const newRanking = sortByScore(newScores);

    // Update UI immediately - user sees ranking change
    updateImageList(newRanking);
    // UI shows loading indicator for heatmap

    // Phase 2: Very fast (5-10ms)
    const topImage = newRanking[0];
    if (topImage !== currentTopImage) {
        currentTopImage = topImage;
        showLoadingOverlay();
    }

    const heatmap = await renderHeatmap(topImage, newValue);  // Level 3 cache
    updateMainView(topImage, heatmap);
    hideLoadingOverlay();

    // Phase 3: Background (15-20ms), can be async
    setTimeout(() => {
        updateSidebarThumbnails(newRanking.slice(1, 6), newValue);
    }, 0);
}
```

**User Experience:**
1. Slider moves → ranking updates instantly (10ms) ← User sees immediate feedback
2. Main heatmap updates (15-20ms) ← Barely perceptible delay
3. Thumbnails fade in (30-40ms) ← Smooth progressive enhancement

---

## Heatmap Rendering Optimization

### Problem with Current Debug Code

Current `--debug-dump-images` writes 13 PNG files (35-38ms):
- 6 heatmaps: t224, t112, p224, p112, pfused, weights
- 6 overlays: overlay_* versions
- 1 normalized input: image.png

**Issues:**
1. Too many images (only need 2-3 for GUI)
2. Writes to disk (slower than in-memory)
3. Full resolution (could use smaller for thumbnails)

### Optimized Rendering for GUI

**Extract rendering from I/O:**
```rust
pub fn render_heatmap_to_buffer(
    data: &Array2<f32>,
    colormap: ColorMap,
    size: (u32, u32),  // Allow resizing for thumbnails
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    // Render heatmap to in-memory buffer
    // No disk I/O
    // ~2-3ms for 224×224
    // ~1ms for 64×64 thumbnail
}

pub fn render_overlay_to_buffer(
    original: &DynamicImage,
    heatmap: &Array2<f32>,
    alpha: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    // Composite heatmap over original image
    // ~3-4ms for 224×224
}
```

**For GUI, render only what's needed:**
- Main view: `pfused` heatmap + overlay (~7-8ms)
- Sidebar thumbnails: Small `pfused` overlays (~3-4ms each)

**Total rendering time:**
- Main: 7-8ms
- 5 thumbnails: 5 × 4ms = 20ms
- Total: 27-28ms

Combined with scoring (10ms): **37-38ms total** - still instant!

---

## Memory Budget

### Level 1 Cache (Parameter-Independent)

Per image:
- Preprocessed tensor: 224×224×3×4 = 600KB
- ONNX output: 401×4 = 1.6KB
- Blur raw: ~1KB (two 20×20 grids + metadata)
- **Total: ~602KB per image**

For 100 images: **60MB** - very reasonable for modern systems

### Level 2 Cache (Parameter-Dependent, All Images)

Per image:
- Blur probabilities: 20×20×4 = 1.6KB
- Blur weights: 20×20×4 = 1.6KB
- Quality score: 4 bytes
- **Total: ~3.2KB per image**

For 100 images: **320KB** - negligible

### Level 3 Cache (Rendered Visualizations, Visible Only)

Per image (assuming 2-3 heatmaps at 224×224 RGBA):
- Each heatmap: 224×224×4 = 200KB
- 3 heatmaps: 600KB
- **Total: ~600KB per visible image**

Cache policy: Keep last 10 visible images = **6MB** - negligible

**Total Memory Budget:** 60MB + 0.3MB + 6MB = **~66MB** for 100 images

Scales linearly: 1000 images = **~660MB** - still reasonable

---

## Implementation Recommendation

### Phase 1: Core Caching (Week 1)

1. **Refactor blur detection to separate computation from rendering**
   - Extract `compute_blur_probabilities()` from raw Tenengrad
   - Extract `compute_blur_weights()` from probabilities
   - Make parameters explicit arguments (not constants)

2. **Implement Level 1 & 2 caching**
   - Cache structure for parameter-independent data
   - Fast recomputation from cache on slider change
   - Benchmark: Confirm <10ms for 100 images

3. **Basic heatmap rendering**
   - Extract `render_heatmap_to_buffer()` from debug code
   - Remove disk I/O
   - Render only essential visualizations (pfused, weights)

### Phase 2: GUI Integration (Week 2)

4. **Implement Level 3 visualization cache**
   - LRU cache for rendered heatmaps
   - Invalidate on parameter change
   - Lazy rendering (only visible images)

5. **Progressive rendering UI**
   - Immediate ranking update
   - Main view heatmap (with loading indicator)
   - Async sidebar thumbnail updates

6. **Polish and optimization**
   - Benchmark end-to-end response time
   - Optimize thumbnail rendering (smaller sizes)
   - Add smooth transitions

---

## Expected Performance

### Initial Load (One Time)

**Sequential:**
- 100 images × 60ms = **6 seconds**

**Parallelized (8 cores):**
- 100 images ÷ 8 = 12.5 images per core
- 12.5 × 60ms = **750ms**

**With progress bar:** Feels fast and responsive

### Slider Adjustment (Every Change)

**Best case (top image unchanged, heatmap cached):**
- Score recomputation: 10ms
- Heatmap from cache: 0ms
- **Total: 10ms** ← Imperceptible

**Typical case (top image changed):**
- Score recomputation: 10ms
- Main heatmap render: 7-8ms
- **Total: 17-18ms** ← Feels instant

**Worst case (top image changed + sidebar updates):**
- Score recomputation: 10ms
- Main heatmap: 7-8ms
- 5 sidebar thumbnails: 20ms
- **Total: 37-38ms** ← Still feels instant (under 50ms threshold)

---

## Alternative: Heatmap-Free UI

If heatmap rendering becomes a bottleneck, consider:

**Minimalist UI:**
- Show images sorted by quality score
- Display score as number/bar
- Show heatmap ONLY for selected image on demand
- Update scores instantly (10ms)
- Render heatmap only when user clicks image (7-8ms)

**Benefits:**
- Even faster response (10ms always)
- Lower memory usage
- Simpler implementation

**Tradeoffs:**
- Less visual feedback
- Requires explicit interaction to see heatmaps

---

## Conclusion

**The key insight:** Separating parameter-dependent computation into two phases:

1. **Cheap computation** (10ms): Recompute quality scores from cached raw data
2. **Moderate rendering** (7-30ms): Render heatmaps only for visible images

This gives us **17-40ms end-to-end response** which feels instant.

**Recommended approach:**
- Use Level 1-3 caching strategy
- Render heatmaps to in-memory buffers (not PNG files)
- Render only what's visible (main + 3-5 thumbnails)
- Use progressive rendering for smooth UX

**Result:** Instant-feeling quality triage UI with real-time parameter tuning!
