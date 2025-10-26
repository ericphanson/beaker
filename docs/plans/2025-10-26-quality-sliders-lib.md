# Quality Algorithm GUI Tuning: Library Architecture Proposal

**Date:** 2025-10-26
**Status:** Proposal
**Author:** Claude (Analysis Agent)

## Executive Summary

This proposal analyzes architectural approaches for enabling real-time GUI slider-based tuning of the quality assessment algorithm without rerunning expensive ML inference. We evaluate four architectural approaches ranging from simple caching to sophisticated incremental computation frameworks.

**Performance Characteristics:**
- **Total processing time:** 57-70ms per image on CPU
- **Preprocessing:** 24-31ms (~42% of total) - Resize to 224x224 + normalize
- **ONNX inference:** 29-31ms (~47% of total) - INT8 quantized model
- **Blur detection:** 2.1-2.3ms (~3% of total) - Multi-scale Sobel on 224x224
- **Postprocessing:** <0.1ms (<0.1% of total) - Score combination

**Caching Impact:**
- **Initial load (1000 images):** 60-70 seconds sequential, 7-9 seconds parallelized (8 cores)
- **Slider adjustment with caching:** <0.1 seconds
- **Slider adjustment without caching:** 60-70 seconds (full recomputation)
- **Speedup from caching:** 600-700x

**Recommendation:** Start with **Approach 2 (Staged Computation with Simple Caching)** for immediate implementation, with **Approach 3 (Hybrid Staged + Salsa)** as a future enhancement if complex multi-parameter dependencies emerge.

---

## Problem Analysis

### Current Architecture

The quality pipeline executes three stages sequentially:

```
Input Image (any size)
    ↓
[1] Preprocessing (24-31ms) ← EXPENSIVE, PARAMETER-INDEPENDENT
    ├─ Resize to 224×224
    └─ Normalize to [0,1] RGB
    ↓
[2] Multi-Scale Blur Detection (2.1-2.3ms) ← FAST, PARAMETER-DEPENDENT
    ├─ Sobel filtering on 224×224
    ├─ Downsample to 112×112
    ├─ Sobel filtering on 112×112
    ├─ Multi-scale fusion (uses BETA, TAU_TEN_224, P_FLOOR)
    └─ Output: 20×20 blur maps (w20, p20, t224)
    ↓
[3] ONNX Model Inference (29-31ms) ← EXPENSIVE, PARAMETER-INDEPENDENT
    ├─ Input: 224×224 RGB tensor
    └─ Output: 401 values (1 global + 400 local)
    ↓
[4] Heuristic Postprocessing (<0.1ms) ← NEGLIGIBLE, PARAMETER-DEPENDENT
    ├─ Combine paq2piq + blur scores (uses ALPHA, MIN_WEIGHT)
    ├─ Apply tunable parameters
    └─ Generate final quality scores

Total: 57-70ms per image on CPU
```

### Tunable Parameters

Located in `beaker/src/blur_detection.rs:6-24`:

| Parameter | Default | Purpose | Usage Stage |
|-----------|---------|---------|-------------|
| `ALPHA` | 0.7 | Blur weight coefficient | Postprocessing |
| `MIN_WEIGHT` | 0.2 | Minimum blur weight clamp | Postprocessing |
| `TAU_TEN_224` | 0.02 | Tenengrad threshold | Blur detection |
| `BETA` | 1.2 | Probability mapping power | Blur detection |
| `P_FLOOR` | 0.0 | Blur probability floor | Blur detection |
| `S_REF` | 96.0 | Reference detection size | Per-detection analysis |
| `COV_REF` | 4.0 | Reference grid coverage | Per-detection analysis |
| `CORE_RATIO` | 0.60 | Core vs ring ratio | Per-detection analysis |
| `ROI_SAMPLES` | 8 | Bilinear samples per axis | Per-detection analysis |
| `GAUSS_SIGMA_NATIVE` | 1.0 | Denoise sigma | Per-detection analysis |

### Performance Profile

Benchmark timings on example images (1280x960-1280px) using CPU:

**Per-Stage Breakdown:**
- **Preprocessing:** 24-31ms (~42% of total) - Resize to 224x224 + normalize
- **Blur Detection:** 2.1-2.3ms (~3% of total) - Multi-scale Sobel + Tenengrad on 224x224
- **ONNX Inference:** 29-31ms (~47% of total) - INT8 quantized model
- **Postprocessing:** 0.01-0.06ms (<0.1% of total) - Combine scores
- **Total:** 57-70ms per image

**Benchmark Results:**
```
example.jpg (1280x960):
  Run 1: Preprocess 24.24ms | Blur 2.16ms | ONNX 30.51ms | Post 0.03ms | Total 63ms
  Run 2: Preprocess 25.27ms | Blur 2.32ms | ONNX 28.82ms | Post 0.06ms | Total 63ms

example-2-birds.jpg (1280x1280):
  Run 3: Preprocess 31.25ms | Blur 2.25ms | ONNX 28.99ms | Post 0.03ms | Total 70ms
```

**Key Observations:**
1. **Preprocessing and ONNX dominate cost** - Together account for ~90% of processing time
2. **Blur detection is surprisingly fast** - Only 2ms because it operates on 224x224 resized images, not original resolution
3. **Postprocessing is negligible** - Less than 0.1ms, essentially free
4. **Preprocessing cost varies with input size** - Larger images take longer to resize to 224x224

**Debug Visualization Cost:**

When `--debug-dump-images` flag is enabled, the algorithm generates 13 heatmap visualizations:
- Heatmaps: t224, t112, p224, p112, pfused, weights (6 images)
- Overlays: overlay_t224, overlay_t112, overlay_p224, overlay_p112, overlay_pfused, overlay_weights (6 images)
- Normalized input: image.png (1 image)

**Performance with Debug Images:**
- **Debug image generation:** 35-38ms (writes 13 PNG files)
- **Total with debug images:** 99-109ms per image (~40-50ms overhead)
- **For GUI use:** Debug images can be generated on-demand or cached alongside quality results

### GUI Tuning Requirements

**User Experience Goals:**
1. Adjust sliders and see results update within 50ms ("instant" feel)
2. Handle folders of 100-1000 images
3. Support undo/redo of parameter changes
4. Enable A/B comparison of parameter sets
5. Export tuned parameter configurations

**Technical Requirements:**
- Cache expensive operations (preprocessing, ONNX inference)
- Recompute only affected results when parameters change
- Support batch recomputation when changing folders
- Memory-efficient caching for large image sets
- Render heatmap visualizations on demand

### Real-Time GUI Scenario Analysis

**See detailed analysis:** [`docs/plans/2025-10-26-gui-scenario-analysis.md`](./2025-10-26-gui-scenario-analysis.md)

**Scenario:** 100 images, user adjusts slider, GUI shows top image with heatmap + sidebar thumbnails

**Key Performance Insights:**

With three-level caching strategy:

**Level 1: Expensive ML (Parameter-Independent)**
- Cache: Preprocessed tensors + ONNX outputs + raw Tenengrad
- Cost: 57-70ms per image on initial load
- Memory: ~600KB per image (60MB for 100 images)

**Level 2: Heuristics (Parameter-Dependent, Cheap)**
- Compute: Quality scores from Level 1 cache
- Cost: **<0.1ms per image × 100 = 10ms total** ← Instant!
- Memory: ~3KB per image (300KB for 100 images)

**Level 3: Visualization (Parameter-Dependent, On-Demand)**
- Render: Heatmaps for visible images only (top + 3-5 thumbnails)
- Cost: 7-8ms for main view, 3-4ms per thumbnail
- Memory: ~600KB per visible image (~6MB for 10 cached)

**Total Response Time:**
- Best case: 10ms (scores only, cached heatmap)
- Typical: 17-18ms (scores + main heatmap)
- Worst case: 37-38ms (scores + main + 5 thumbnails)

All well under 50ms threshold for "instant" feel!

**Critical Insight:** Heatmaps ARE parameter-dependent (use BETA, ALPHA, etc.), so we must:
1. Separate computation from visualization
2. Render heatmaps to in-memory buffers (not PNG files)
3. Render only visible images, cache the results
4. Use progressive rendering (update ranking instantly, then heatmaps)

---

## Approach 1: Simple Cache with Manual Invalidation

### Architecture

Store expensive computation results in a simple HashMap cache, invalidate manually when parameters affecting those computations change.

```rust
/// Cache key for expensive operations
#[derive(Hash, Eq, PartialEq, Clone)]
struct ImageCacheKey {
    image_path: PathBuf,
    image_md5: String,  // Detect file changes
}

/// Cached intermediate results
struct QualityCache {
    /// ONNX model outputs: global + 20×20 local paq2piq scores
    paq2piq_cache: HashMap<ImageCacheKey, Paq2PiqResult>,

    /// Multi-scale blur detection: raw Tenengrad values
    blur_raw_cache: HashMap<ImageCacheKey, BlurRawResult>,
}

#[derive(Clone)]
struct Paq2PiqResult {
    global_score: f32,
    local_grid: [[u8; 20]; 20],
}

#[derive(Clone)]
struct BlurRawResult {
    /// Raw 20×20 Tenengrad scores (before parameter-dependent mapping)
    t224: [[f32; 20]; 20],
    t112: [[f32; 20]; 20],
    /// Median Tenengrad for adaptive thresholding
    median_t224: f32,
}

/// Fast parameter-dependent recomputation layer
struct QualityHeuristics {
    params: QualityParams,
}

impl QualityHeuristics {
    fn compute_from_cache(
        &self,
        paq: &Paq2PiqResult,
        blur_raw: &BlurRawResult,
    ) -> QualityResult {
        // Recompute blur probabilities with current BETA, TAU_TEN_224, P_FLOOR
        let blur_prob = self.tenengrad_to_probability(&blur_raw.t224, blur_raw.median_t224);

        // Recompute weights with current ALPHA, MIN_WEIGHT
        let weights = self.compute_weights(&blur_prob);

        // Combine paq + blur with current parameters
        let global_quality = self.combine_scores(paq.global_score, blur_prob);

        QualityResult { /* ... */ }
    }

    fn tenengrad_to_probability(&self, t: &[[f32; 20]; 20], median: f32) -> [[f32; 20]; 20] {
        // Uses: self.params.tau_ten_224, self.params.beta, self.params.p_floor
        // Fast computation: ~1ms for 20×20 grid
        // ...
    }
}
```

### Parameter Categorization

**Tier 1: Postprocessing-only (cache fully reusable)**
- `ALPHA`, `MIN_WEIGHT` → Only affect final score combination
- **Recompute cost:** <1ms (pure arithmetic on cached values)

**Tier 2: Blur mapping (partial cache reuse)**
- `BETA`, `TAU_TEN_224`, `P_FLOOR` → Affect Tenengrad→probability mapping
- **Cache reuse:** Raw Tenengrad values (t224, t112)
- **Recompute cost:** 1-2ms (apply new sigmoid mapping)

**Tier 3: Detection-specific (only affects per-detection analysis)**
- `S_REF`, `COV_REF`, `CORE_RATIO`, `ROI_SAMPLES`, `GAUSS_SIGMA_NATIVE`
- **Cache reuse:** Full image quality results unchanged
- **Recompute cost:** Only when used with detection pipeline

### Cache Management

```rust
impl QualityCache {
    /// Load or compute cached results
    async fn get_or_compute(&mut self, image_path: &Path) -> Result<CachedResults> {
        let key = self.compute_cache_key(image_path)?;

        let paq = match self.paq2piq_cache.get(&key) {
            Some(cached) => cached.clone(),
            None => {
                let result = self.run_onnx_inference(image_path).await?;
                self.paq2piq_cache.insert(key.clone(), result.clone());
                result
            }
        };

        let blur_raw = match self.blur_raw_cache.get(&key) {
            Some(cached) => cached.clone(),
            None => {
                let result = self.compute_blur_raw(image_path).await?;
                self.blur_raw_cache.insert(key.clone(), result.clone());
                result
            }
        };

        Ok(CachedResults { paq, blur_raw })
    }

    /// Clear cache when switching folders or detecting file changes
    fn invalidate_all(&mut self) {
        self.paq2piq_cache.clear();
        self.blur_raw_cache.clear();
    }
}
```

### Integration with GUI

```rust
/// GUI-facing quality service
struct QualityService {
    cache: QualityCache,
    heuristics: QualityHeuristics,
    current_folder: Vec<PathBuf>,
}

impl QualityService {
    /// Initial load: populate cache for all images
    async fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        self.cache.invalidate_all();
        self.current_folder = paths;

        // Parallel cache population
        for path in &self.current_folder {
            self.cache.get_or_compute(path).await?;
        }
        Ok(())
    }

    /// Fast recomputation when sliders change
    fn update_parameters(&mut self, new_params: QualityParams) -> Result<Vec<QualityResult>> {
        self.heuristics.params = new_params;

        // Fast recompute from cache: 1-2ms per image
        let results: Vec<_> = self.current_folder.iter()
            .map(|path| {
                let key = self.cache.compute_cache_key(path)?;
                let cached = self.cache.get_cached(&key)?;
                self.heuristics.compute_from_cache(&cached.paq, &cached.blur_raw)
            })
            .collect::<Result<_>>()?;

        Ok(results)
    }
}
```

### Pros

1. **Simplicity:** Straightforward HashMap-based caching with manual management
2. **Predictable:** Explicit control over what's cached and when
3. **Minimal dependencies:** No external frameworks required
4. **Easy debugging:** Clear cache hit/miss behavior
5. **Good enough:** For simple parameter tuning, manual invalidation works well
6. **Fast implementation:** 1-2 days of development

### Cons

1. **Manual invalidation:** Must track parameter→cache dependencies manually
2. **No automatic optimization:** Can't detect unchanged intermediate results
3. **Memory management:** Must manually decide cache eviction strategy
4. **No dependency tracking:** If we add cross-parameter dependencies, complexity explodes
5. **Limited scalability:** Hard to extend if we add complex parameter interactions
6. **Brittle:** Easy to introduce cache invalidation bugs

### Implementation Estimate

- **Core caching infrastructure:** 4-6 hours
- **Parameter-based recomputation:** 2-3 hours
- **GUI integration:** 2-3 hours
- **Testing and validation:** 2-3 hours
- **Total:** 1-2 days

### When to Choose This

- **Small scope:** 2-5 tunable parameters with simple dependencies
- **Prototype phase:** Need quick validation of GUI tuning concept
- **Resource-constrained:** Cannot add external dependencies

---

## Approach 2: Staged Computation with Simple Caching

### Architecture

Separate the pipeline into explicit stages with different cache invalidation rules. Use the `cached` crate for automatic memoization with TTL and capacity limits.

```rust
use cached::proc_macro::cached;
use cached::{Cached, TimedCache};

/// Stage 1: ONNX inference (no parameters, cache forever until file changes)
#[cached(
    type = "cached::UnboundCache<String, Paq2PiqResult>",
    create = "{ cached::UnboundCache::new() }",
    convert = r#"{ format!("{:?}:{}", _path, _md5) }"#
)]
fn stage1_onnx_inference(_path: &Path, _md5: &str, image: &DynamicImage) -> Result<Paq2PiqResult> {
    // Run ONNX model: 29-31ms
    // Output: global + 20×20 local paq2piq scores
}

/// Stage 2: Raw blur detection (no parameters, cache forever)
#[cached(
    type = "cached::UnboundCache<String, BlurRawResult>",
    create = "{ cached::UnboundCache::new() }",
    convert = r#"{ format!("{:?}:{}", _path, _md5) }"#
)]
fn stage2_blur_raw(_path: &Path, _md5: &str, image: &DynamicImage) -> Result<BlurRawResult> {
    // Run multi-scale Sobel + Tenengrad: ~2ms (includes all Sobel filtering)
    // Output: raw Tenengrad values WITHOUT parameter-dependent transformations
}

/// Stage 3: Parameter-dependent blur mapping (fast, cache with param hash)
#[cached(
    type = "cached::SizedCache<String, BlurMappedResult>",
    create = "{ cached::SizedCache::with_size(1000) }",
    convert = r#"{ format!("{}_{}_{}", _blur_raw_hash, _params.beta, _params.tau_ten_224) }"#
)]
fn stage3_blur_mapping(
    _blur_raw_hash: u64,
    blur_raw: &BlurRawResult,
    _params: &BlurParams,
) -> BlurMappedResult {
    // Apply BETA, TAU_TEN_224, P_FLOOR: <0.1ms (simple math on 20x20 grid)
    // Output: 20×20 blur probabilities and weights
}

/// Stage 4: Final combination (instant, cache with all param hash)
#[cached(
    type = "cached::SizedCache<String, QualityResult>",
    create = "{ cached::SizedCache::with_size(1000) }",
    convert = r#"{ format!("{}_{}_{}_{}", _paq_hash, _blur_hash, _params.alpha, _params.min_weight) }"#
)]
fn stage4_combine(
    _paq_hash: u64,
    paq: &Paq2PiqResult,
    _blur_hash: u64,
    blur: &BlurMappedResult,
    _params: &CombineParams,
) -> QualityResult {
    // Combine scores with ALPHA, MIN_WEIGHT: <0.01ms (trivial arithmetic)
}
```

### Parameter Dependency Graph

```
                    [Image File]
                         |
            ┌────────────┴────────────┐
            ↓                         ↓
    [Stage 1: ONNX]          [Stage 2: Blur Raw]
       (no params)               (no params)
            |                         |
            |                         ↓
            |            [Stage 3: Blur Mapping]
            |               (BETA, TAU, P_FLOOR)
            |                         |
            └────────────┬────────────┘
                         ↓
                [Stage 4: Combine]
               (ALPHA, MIN_WEIGHT)
                         ↓
                  [Final Result]
```

### Cache Management Strategy

```rust
/// Manages staged caching with smart invalidation
pub struct StagedQualityPipeline {
    /// Stage 1 & 2: Never invalidate unless file changes
    file_cache: FileBasedCache,

    /// Stage 3 & 4: Invalidate when relevant params change
    param_cache: ParameterBasedCache,

    /// Current parameter set
    params: QualityParams,
}

impl StagedQualityPipeline {
    /// Update parameters and invalidate only affected stages
    pub fn update_parameters(&mut self, new_params: QualityParams) {
        let old_params = &self.params;

        // Check which stages need invalidation
        let invalidate_stage3 =
            new_params.beta != old_params.beta ||
            new_params.tau_ten_224 != old_params.tau_ten_224 ||
            new_params.p_floor != old_params.p_floor;

        let invalidate_stage4 =
            new_params.alpha != old_params.alpha ||
            new_params.min_weight != old_params.min_weight;

        if invalidate_stage3 {
            self.param_cache.clear_stage3();
            // Stage 4 depends on stage 3, so clear it too
            self.param_cache.clear_stage4();
        } else if invalidate_stage4 {
            self.param_cache.clear_stage4();
        }

        self.params = new_params;
    }

    /// Process single image through pipeline
    pub fn process_image(&self, path: &Path) -> Result<QualityResult> {
        let md5 = calculate_md5(path)?;
        let image = image::open(path)?;

        // Stage 1 & 2: Run expensive ops once, cache forever
        let paq = stage1_onnx_inference(path, &md5, &image)?;
        let blur_raw = stage2_blur_raw(path, &md5, &image)?;

        // Stage 3: Fast mapping with current params (cached)
        let blur_raw_hash = self.hash_blur_raw(&blur_raw);
        let blur_params = self.params.blur_params();
        let blur_mapped = stage3_blur_mapping(blur_raw_hash, &blur_raw, &blur_params)?;

        // Stage 4: Final combination (cached)
        let paq_hash = self.hash_paq(&paq);
        let blur_hash = self.hash_blur_mapped(&blur_mapped);
        let combine_params = self.params.combine_params();
        stage4_combine(paq_hash, &paq, blur_hash, &blur_mapped, &combine_params)
    }

    /// Batch process folder with progress reporting
    pub fn process_folder(&self, paths: &[PathBuf], progress: impl Fn(usize)) -> Vec<QualityResult> {
        paths.par_iter()  // Use rayon for parallelism
            .enumerate()
            .map(|(i, path)| {
                let result = self.process_image(path)?;
                progress(i + 1);
                Ok(result)
            })
            .collect()
    }
}
```

### Memory Management

```rust
/// Configuration for cache sizes and eviction
pub struct CacheConfig {
    /// Stage 1 & 2: Unlimited cache (rely on folder change to clear)
    max_file_cache_entries: Option<usize>,  // None = unlimited

    /// Stage 3 & 4: LRU cache with size limit
    max_param_cache_entries: usize,  // Default: 1000

    /// Enable disk-based caching for Stage 1 & 2
    disk_cache_dir: Option<PathBuf>,
}

impl StagedQualityPipeline {
    /// Optional: Serialize expensive results to disk
    fn cache_to_disk(&self, path: &Path, paq: &Paq2PiqResult, blur: &BlurRawResult) -> Result<()> {
        if let Some(cache_dir) = &self.config.disk_cache_dir {
            let cache_file = cache_dir.join(format!("{}.cache", path.file_name()?));
            let data = CachedData { paq: paq.clone(), blur: blur.clone() };
            fs::write(cache_file, serde_json::to_vec(&data)?)?;
        }
        Ok(())
    }

    /// Load from disk cache if available
    fn load_from_disk(&self, path: &Path) -> Option<CachedData> {
        let cache_dir = self.config.disk_cache_dir.as_ref()?;
        let cache_file = cache_dir.join(format!("{}.cache", path.file_name().ok()?));
        let data = fs::read(cache_file).ok()?;
        serde_json::from_slice(&data).ok()
    }
}
```

### GUI Integration Pattern

```rust
/// Reactive GUI service
pub struct QualityGuiService {
    pipeline: StagedQualityPipeline,
    current_results: Vec<(PathBuf, QualityResult)>,
}

impl QualityGuiService {
    /// Called when user opens a folder (expensive: runs Stage 1 & 2 for all images)
    pub async fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<Vec<QualityResult>> {
        self.current_results.clear();

        // Show progress bar during expensive operations
        let results = self.pipeline.process_folder(&paths, |i| {
            println!("Processing {}/{}", i, paths.len());
        });

        self.current_results = paths.into_iter().zip(results).collect();
        Ok(self.current_results.iter().map(|(_, r)| r.clone()).collect())
    }

    /// Called when user adjusts sliders (fast: only Stage 3 & 4)
    pub fn update_sliders(&mut self, new_params: QualityParams) -> Result<Vec<QualityResult>> {
        self.pipeline.update_parameters(new_params);

        // Recompute all images with new params (1-2ms each)
        let results: Vec<_> = self.current_results.iter()
            .map(|(path, _)| self.pipeline.process_image(path))
            .collect::<Result<_>>()?;

        // Update stored results
        for (i, result) in results.iter().enumerate() {
            self.current_results[i].1 = result.clone();
        }

        Ok(results)
    }
}
```

### Pros

1. **Clean separation:** Each stage has explicit responsibilities and caching rules
2. **Automatic memoization:** `cached` crate handles memoization boilerplate
3. **Smart invalidation:** Only recompute affected stages when parameters change
4. **Ergonomic:** Proc macros make caching transparent
5. **Flexible:** Easy to add new stages or adjust caching strategies
6. **Testable:** Each stage can be tested independently
7. **Memory efficient:** LRU caching for parameter-dependent stages
8. **Disk caching support:** Can persist expensive results across sessions

### Cons

1. **Proc macro overhead:** Compile times slightly increased
2. **Hash collisions:** Must carefully design cache keys to avoid false hits
3. **Manual stage design:** Must explicitly split algorithm into stages
4. **Memory usage:** Multiple cache layers consume more memory
5. **Debugging complexity:** Cache behavior can be opaque with proc macros
6. **Limited dependency tracking:** Still manual tracking of param→stage relationships

### Implementation Estimate

- **Stage separation refactoring:** 4-6 hours
- **Cache key design and hashing:** 2-3 hours
- **Invalidation logic:** 3-4 hours
- **GUI service integration:** 3-4 hours
- **Disk caching (optional):** 2-3 hours
- **Testing and optimization:** 4-6 hours
- **Total:** 2-3 days

### When to Choose This

- **Medium complexity:** 5-10 tunable parameters with clear stage boundaries
- **Production-ready:** Need robust caching with automatic management
- **Growing scope:** Expect to add more parameters and stages over time
- **Standard dependencies:** `cached` crate is lightweight and well-maintained

---

## Approach 3: Hybrid Staged + Salsa

### Architecture

Combine staged computation (Approach 2) for the expensive ML operations with Salsa for fine-grained incremental computation of the heuristics layer. This provides the best of both worlds: simple caching for expensive operations and sophisticated dependency tracking for complex parameter interactions.

```rust
use salsa::Database;

/// Salsa database for quality computation
#[salsa::db]
pub trait QualityDatabase: salsa::Database {
    /// Input: Image file metadata (never changes during parameter tuning)
    #[salsa::input]
    fn image_file(&self, path: PathBuf) -> ImageFileInput;

    /// Input: Current parameter set (changes when sliders adjust)
    #[salsa::input]
    fn quality_params(&self) -> QualityParams;
}

/// Image file input (immutable during tuning session)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageFileInput {
    pub path: PathBuf,
    pub md5: String,
    pub width: u32,
    pub height: u32,
}

/// Parameter set (mutable via sliders)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualityParams {
    // Blur mapping params
    pub beta: OrderedFloat<f32>,
    pub tau_ten_224: OrderedFloat<f32>,
    pub p_floor: OrderedFloat<f32>,

    // Combination params
    pub alpha: OrderedFloat<f32>,
    pub min_weight: OrderedFloat<f32>,

    // Detection params
    pub s_ref: OrderedFloat<f32>,
    pub cov_ref: OrderedFloat<f32>,
    pub core_ratio: OrderedFloat<f32>,
}

/// Expensive operations: NOT tracked by Salsa, manually cached
pub struct ExpensiveCache {
    paq_cache: HashMap<String, Paq2PiqResult>,
    blur_cache: HashMap<String, BlurRawResult>,
}

impl ExpensiveCache {
    fn get_paq(&mut self, input: &ImageFileInput) -> Result<Paq2PiqResult> {
        let key = format!("{}:{}", input.path.display(), input.md5);
        if let Some(cached) = self.paq_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Run ONNX: 29-31ms
        let result = run_onnx_inference(&input.path)?;
        self.paq_cache.insert(key, result.clone());
        Ok(result)
    }

    fn get_blur_raw(&mut self, input: &ImageFileInput) -> Result<BlurRawResult> {
        let key = format!("{}:{}", input.path.display(), input.md5);
        if let Some(cached) = self.blur_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Run blur detection: ~2ms
        let result = compute_blur_raw(&input.path)?;
        self.blur_cache.insert(key, result.clone());
        Ok(result)
    }
}

/// Salsa-tracked functions for fast parameter-dependent computations
#[salsa::tracked]
fn blur_probability_map(
    db: &dyn QualityDatabase,
    image: ImageFileInput,
) -> BlurProbabilityMap {
    // Get cached expensive result (NOT tracked by Salsa)
    let blur_raw = db.expensive_cache().get_blur_raw(&image).unwrap();

    // Get current params (tracked by Salsa)
    let params = db.quality_params();

    // Fast computation: <0.1ms
    // Salsa will only rerun this if params.beta/tau_ten_224/p_floor change
    compute_blur_probability(&blur_raw, params.beta, params.tau_ten_224, params.p_floor)
}

#[salsa::tracked]
fn blur_weights(
    db: &dyn QualityDatabase,
    image: ImageFileInput,
) -> BlurWeights {
    // Depends on blur_probability_map
    let prob_map = blur_probability_map(db, image);
    let params = db.quality_params();

    // Fast computation: <0.01ms
    // Salsa will only rerun if prob_map or params.alpha/min_weight change
    compute_weights(&prob_map, params.alpha, params.min_weight)
}

#[salsa::tracked]
fn quality_result(
    db: &dyn QualityDatabase,
    image: ImageFileInput,
) -> QualityResult {
    // Get cached expensive result
    let paq = db.expensive_cache().get_paq(&image).unwrap();

    // Get intermediate results (tracked by Salsa)
    let blur_prob = blur_probability_map(db, image);
    let blur_weights = blur_weights(db, image);

    // Combine everything
    combine_quality(paq, blur_prob, blur_weights)
}

/// Extension trait to access expensive cache
pub trait QualityDatabaseExt: QualityDatabase {
    fn expensive_cache(&self) -> &ExpensiveCache;
    fn expensive_cache_mut(&mut self) -> &mut ExpensiveCache;
}
```

### Dependency Tracking Example

When a slider changes, Salsa automatically determines minimal recomputation:

```
User adjusts BETA slider:
    ↓
Salsa detects: quality_params() input changed
    ↓
Salsa checks dependents:
    - blur_probability_map() reads params.beta → INVALIDATE
    - blur_weights() reads blur_probability_map → INVALIDATE
    - quality_result() reads blur_probability_map → INVALIDATE
    ↓
User requests quality_result():
    ↓
Salsa recomputes chain:
    1. blur_probability_map() → <0.1ms (uses cached blur_raw)
    2. blur_weights() → <0.01ms (uses fresh prob_map)
    3. quality_result() → <0.01ms (combines everything)
    ↓
Total: <0.1ms per image

User adjusts ALPHA slider:
    ↓
Salsa detects: quality_params() input changed
    ↓
Salsa checks dependents:
    - blur_probability_map() does NOT read alpha → KEEP CACHED
    - blur_weights() reads params.alpha → INVALIDATE
    - quality_result() reads blur_weights → INVALIDATE
    ↓
Salsa recomputes:
    1. blur_probability_map() → CACHE HIT (unchanged)
    2. blur_weights() → <0.01ms (uses cached prob_map)
    3. quality_result() → <0.01ms
    ↓
Total: <0.01ms per image (saved ~0.1ms by keeping prob_map)
```

### GUI Service with Salsa

```rust
/// GUI service with Salsa database
pub struct SalsaQualityService {
    db: QualityDatabaseImpl,
    current_images: Vec<ImageFileInput>,
}

impl SalsaQualityService {
    /// Load folder: populate expensive cache
    pub async fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        self.current_images.clear();

        for path in paths {
            let md5 = calculate_md5(&path)?;
            let (width, height) = image::image_dimensions(&path)?;

            let input = ImageFileInput { path: path.clone(), md5, width, height };

            // Pre-populate expensive cache
            self.db.expensive_cache_mut().get_paq(&input)?;
            self.db.expensive_cache_mut().get_blur_raw(&input)?;

            // Register as Salsa input
            self.db.set_image_file(path, input.clone());
            self.current_images.push(input);
        }

        Ok(())
    }

    /// Update parameters: Salsa handles invalidation automatically
    pub fn update_sliders(&mut self, new_params: QualityParams) -> Vec<QualityResult> {
        // Update Salsa input
        self.db.set_quality_params(new_params);

        // Query results: Salsa only recomputes changed dependencies
        self.current_images.iter()
            .map(|input| quality_result(&self.db, input.clone()))
            .collect()
    }

    /// Undo/redo: trivial with Salsa
    pub fn set_parameters(&mut self, params: QualityParams) {
        self.db.set_quality_params(params);
    }
}
```

### Advanced: Multi-Parameter Interactions

Salsa shines when parameters have complex dependencies:

```rust
/// Example: Detection-aware quality with size priors
#[salsa::tracked]
fn detection_quality(
    db: &dyn QualityDatabase,
    image: ImageFileInput,
    bbox: BBox,
) -> DetectionQuality {
    let base_quality = quality_result(db, image);
    let params = db.quality_params();

    // Size prior depends on S_REF
    let size_prior = compute_size_prior(&bbox, params.s_ref);

    // Coverage prior depends on COV_REF
    let coverage_prior = compute_coverage_prior(&bbox, params.cov_ref);

    // Combine (Salsa tracks all dependencies)
    DetectionQuality {
        base_quality,
        size_prior,
        coverage_prior,
        final_score: base_quality.global_score * size_prior * coverage_prior,
    }
}
```

If `S_REF` changes, Salsa only recomputes `size_prior` and `final_score`, keeping `base_quality` and `coverage_prior` cached.

### Pros

1. **Automatic optimization:** Salsa minimizes recomputation without manual tracking
2. **Fine-grained dependencies:** Tracks individual field access (params.alpha vs params.beta)
3. **Undo/redo trivial:** Just set params to previous values, Salsa handles everything
4. **Scales with complexity:** Adding new parameters and dependencies is straightforward
5. **Early cutoff:** If a parameter change produces the same intermediate result, downstream computations reuse cache
6. **Thread-safe:** Salsa databases are thread-safe by design
7. **Debugging tools:** Salsa provides query inspection and profiling
8. **Hybrid approach:** Expensive ops stay simple, complex heuristics get full Salsa benefits

### Cons

1. **Learning curve:** Salsa has unique concepts (database, tracked functions, interning)
2. **Heavyweight dependency:** Salsa is a substantial framework (though well-maintained)
3. **Serialization complexity:** Salsa data structures require care with Clone/Eq/Hash
4. **Compile time overhead:** Salsa macros increase compile times
5. **Over-engineering risk:** May be overkill if parameter interactions remain simple
6. **Mixed caching layers:** Hybrid approach has conceptual overhead (Salsa + manual cache)
7. **Debugging complexity:** Two caching systems (Salsa + ExpensiveCache) to reason about

### Implementation Estimate

- **Salsa database setup:** 4-6 hours
- **Expensive cache integration:** 3-4 hours
- **Tracked function implementation:** 6-8 hours
- **GUI service with Salsa:** 4-6 hours
- **Testing and optimization:** 6-8 hours
- **Total:** 3-5 days

### When to Choose This

- **Complex dependencies:** 10+ parameters with intricate interactions
- **Long-term project:** Expect continued development and parameter additions
- **Advanced features:** Need undo/redo, A/B comparison, parameter presets
- **Performance critical:** Every millisecond matters for large image batches
- **Team familiarity:** Team has experience with incremental computation frameworks

---

## Approach 4: Full Salsa with Interned Computations

### Architecture

Use Salsa for all computation stages, including expensive operations. Store expensive results as interned values in the Salsa database, relying on Salsa's durability system to avoid recomputation.

```rust
#[salsa::db]
pub trait QualityDatabase: salsa::Database {
    /// Input: Image file (with durability: high)
    #[salsa::input]
    fn image_file(&self, path: PathBuf) -> ImageFile;

    /// Input: Parameters (with durability: low)
    #[salsa::input]
    fn quality_params(&self) -> QualityParams;
}

/// Image file metadata
#[salsa::interned]
pub struct ImageFile {
    pub path: PathBuf,
    pub md5: String,
}

/// Expensive operations as tracked functions with high durability
#[salsa::tracked(durability = High)]
fn onnx_inference(
    db: &dyn QualityDatabase,
    image: ImageFile,
) -> Paq2PiqResult {
    // Salsa will cache this aggressively due to High durability
    // Even across parameter changes, this won't recompute
    let img = load_image(db, image.path(db));
    run_onnx_inference(&img)
}

#[salsa::tracked(durability = High)]
fn blur_raw_computation(
    db: &dyn QualityDatabase,
    image: ImageFile,
) -> BlurRawResult {
    // High durability: Salsa knows this is expensive and stable
    let img = load_image(db, image.path(db));
    compute_blur_raw(&img)
}

/// Fast operations with low durability
#[salsa::tracked]
fn blur_probability_map(
    db: &dyn QualityDatabase,
    image: ImageFile,
) -> BlurProbabilityMap {
    let blur_raw = blur_raw_computation(db, image);
    let params = db.quality_params();
    compute_blur_probability(&blur_raw, params)
}

#[salsa::tracked]
fn final_quality(
    db: &dyn QualityDatabase,
    image: ImageFile,
) -> QualityResult {
    let paq = onnx_inference(db, image);
    let blur_prob = blur_probability_map(db, image);
    let params = db.quality_params();
    combine_quality(paq, blur_prob, params)
}
```

### Durability System

Salsa's durability system categorizes inputs by stability:

- **High durability:** Inputs that rarely change (image files)
- **Low durability:** Inputs that change frequently (parameters)

When a low-durability input changes, Salsa only invalidates functions that transitively depend on it, keeping high-durability results cached.

```rust
impl SalsaQualityService {
    pub fn new() -> Self {
        let mut db = QualityDatabaseImpl::default();

        // Set durability for expensive operations
        db.salsa_runtime_mut().set_revision_guard(|revision| {
            // Expensive operations are High durability
            // Parameter-dependent operations are Low durability
        });

        Self { db, current_images: Vec::new() }
    }

    pub fn update_sliders(&mut self, new_params: QualityParams) {
        // Update low-durability input
        self.db.set_quality_params(new_params);

        // Salsa automatically:
        // 1. Keeps onnx_inference() and blur_raw_computation() cached (High durability)
        // 2. Invalidates blur_probability_map() and final_quality() (depend on params)
        // 3. Recomputes only invalidated functions on next query
    }
}
```

### Persistence Across Sessions

```rust
/// Optional: Serialize Salsa database to disk
impl SalsaQualityService {
    pub fn save_session(&self, path: &Path) -> Result<()> {
        // Save expensive computation results
        let snapshot = SalsaSnapshot {
            onnx_cache: self.extract_onnx_cache(),
            blur_cache: self.extract_blur_cache(),
            params: self.db.quality_params(),
        };

        fs::write(path, serde_json::to_vec(&snapshot)?)?;
        Ok(())
    }

    pub fn load_session(&mut self, path: &Path) -> Result<()> {
        let snapshot: SalsaSnapshot = serde_json::from_slice(&fs::read(path)?)?;

        // Restore cached results
        self.restore_onnx_cache(snapshot.onnx_cache);
        self.restore_blur_cache(snapshot.blur_cache);
        self.db.set_quality_params(snapshot.params);

        Ok(())
    }
}
```

### Pros

1. **Unified framework:** All computation goes through Salsa, no mixed approaches
2. **Durability optimization:** Automatic separation of stable vs volatile computations
3. **Full dependency tracking:** Salsa tracks everything, including expensive ops
4. **Persistence support:** Can serialize/deserialize Salsa databases
5. **Revision tracking:** Built-in support for undo/redo via revisions
6. **Profiling tools:** Salsa provides query profiling and debugging
7. **Theoretical optimality:** Salsa minimizes recomputation across all stages

### Cons

1. **Highest complexity:** Most conceptual overhead of all approaches
2. **Heavy dependency:** Salsa becomes core to the entire quality system
3. **Memory overhead:** Salsa stores all intermediate results in database
4. **Serialization challenges:** Must carefully handle large Salsa structures
5. **Opaque behavior:** Harder to predict and debug cache behavior
6. **Over-engineering:** Likely overkill for this use case
7. **Long-running operations:** Salsa not designed for 100ms+ operations (blocking)
8. **Lock contention:** Expensive ops hold Salsa locks, blocking other queries

### Implementation Estimate

- **Full Salsa architecture:** 8-12 hours
- **Durability system setup:** 4-6 hours
- **Expensive operation integration:** 6-8 hours
- **Persistence and session management:** 6-8 hours
- **Testing and debugging:** 8-12 hours
- **Total:** 5-7 days

### When to Choose This

- **Research project:** Exploring advanced incremental computation techniques
- **Highly dynamic:** Expect frequent changes to computation graph structure
- **Persistent sessions:** Need to save/load entire computation state
- **Already using Salsa:** Project already depends on Salsa for other features

---

## Comparative Analysis

### Performance Comparison

| Approach | Initial Load (1000 images) | Slider Adjust (Tier 1) | Slider Adjust (Tier 2) | No Cache (Full Rerun) |
|----------|---------------------------|------------------------|------------------------|-----------------------|
| **Approach 1** | 60-70s | <0.1s | ~2s | 60-70s |
| **Approach 2** | 60-70s | <0.1s | <0.1s | 60-70s |
| **Approach 3** | 60-70s | <0.1s | <0.1s | 60-70s |
| **Approach 4** | 60-70s | <0.1s | <0.1s | 60-70s |

**Notes:**
- **Base performance:** 57-70ms per image × 1000 images = 60-70 seconds
- **Tier 1 params:** ALPHA, MIN_WEIGHT (postprocessing only, <0.1ms to recompute)
- **Tier 2 params:** BETA, TAU_TEN_224, P_FLOOR (blur mapping, ~0.1ms to recompute)
- **Parallelization:** Initial load can be reduced to 7-9 seconds with 8 CPU cores
- **Caching benefit:** 600-700x speedup for slider adjustments
- **Key insight:** All approaches have similar performance since expensive operations (preprocessing + ONNX) are parameter-independent

### Complexity Comparison

| Aspect | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|--------|------------|------------|------------|------------|
| **LOC** | +200-300 | +300-400 | +500-700 | +700-1000 |
| **Dependencies** | None | `cached` | `cached`, `salsa` | `salsa` |
| **Conceptual** | Simple | Medium | Medium-High | High |
| **Maintainability** | Good | Good | Medium | Medium |
| **Extensibility** | Poor | Good | Excellent | Excellent |
| **Debugging** | Easy | Easy | Medium | Hard |

### Feature Comparison

| Feature | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|---------|------------|------------|------------|------------|
| **Basic caching** | ✓ | ✓ | ✓ | ✓ |
| **Auto invalidation** | ✗ | Partial | ✓ | ✓ |
| **Fine-grained deps** | ✗ | ✗ | ✓ | ✓ |
| **Undo/redo** | Manual | Manual | Easy | Easy |
| **Disk persistence** | Manual | Manual | Medium | Easy |
| **Parallel processing** | ✓ | ✓ | ✓ | Limited |
| **Memory efficiency** | ✓ | ✓ | ✓ | ✗ |
| **Early cutoff** | ✗ | ✗ | ✓ | ✓ |

---

## Recommendation

### Primary Recommendation: Approach 2 (Staged Computation)

**Performance Context:**
- **Processing time:** 57-70ms per image on CPU
- **Initial load (1000 images):** 60-70 seconds sequential, 7-9 seconds parallelized
- **Caching benefit:** 600-700x speedup for slider adjustments (<0.1s vs 60-70s)
- **Cost distribution:** Preprocessing (42%) + ONNX (47%) are expensive and parameter-independent; blur (3%) and postprocessing (<1%) are cheap and parameter-dependent

**Why Approach 2:**

1. **Right complexity level:** Balances simplicity with robustness for 5-10 parameters
2. **Proven patterns:** Staged computation is well-understood and maintainable
3. **Fast implementation:** 2-3 days to production-ready code
4. **Lightweight dependency:** `cached` crate is minimal and well-maintained
5. **Room to grow:** Easy to add more stages or convert to Approach 3 later
6. **Natural fit:** The quality algorithm has clear stage boundaries (preprocess → expensive ops → cheap heuristics)
7. **Predictable performance:** Explicit caching rules make performance easy to reason about
8. **All approaches perform similarly:** Since expensive operations are parameter-independent, sophisticated dependency tracking (Approach 3/4) provides minimal benefit

### When to Upgrade to Approach 3

Consider upgrading if:
- **Parameter count grows beyond 10** with complex interactions
- **Need advanced features:** Undo/redo, A/B comparison, parameter presets
- **Performance critical:** Shaving off every millisecond matters
- **Complex cross-parameter dependencies emerge**

### Avoid Approach 4 Unless...

Only consider if:
- Team has deep Salsa expertise
- Project already uses Salsa extensively
- Research/experimental context

### Quick Start with Approach 1

For rapid prototyping:
1. Start with Approach 1 to validate GUI tuning concept (1-2 days)
2. Gather user feedback on which parameters are most important
3. Refactor to Approach 2 for production (2-3 days)
4. Keep option to upgrade to Approach 3 if complexity grows

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal:** Validate GUI tuning concept with Approach 1

1. **Refactor constants to struct** (2 hours)
   - Extract `blur_detection.rs` constants into `QualityParams` struct
   - Thread parameters through existing functions

2. **Implement basic caching** (4 hours)
   - Create `QualityCache` with HashMap for paq2piq and blur results
   - Add MD5-based cache keys

3. **Build minimal GUI service** (4 hours)
   - `load_folder()` → populate cache
   - `update_parameters()` → recompute from cache

4. **Proof-of-concept GUI** (8 hours)
   - Simple egui sliders for ALPHA, BETA, TAU_TEN_224
   - Display quality results in grid
   - Measure and display recomputation time

### Phase 2: Production Implementation (Week 2)

**Goal:** Refactor to Approach 2 for robustness

1. **Stage separation** (6 hours)
   - Split `quality_processing.rs` into clear stage functions
   - Design intermediate data structures

2. **Integrate `cached` crate** (4 hours)
   - Add `#[cached]` macros to expensive functions
   - Implement cache key generation

3. **Smart invalidation** (4 hours)
   - Implement `update_parameters()` with selective invalidation
   - Add parameter tier classification

4. **Memory management** (3 hours)
   - Configure LRU cache sizes
   - Add optional disk caching for Stage 1 & 2

5. **Testing and optimization** (6 hours)
   - Benchmark cache hit rates
   - Test with large image sets (1000+ images)
   - Profile memory usage

### Phase 3: Advanced Features (Week 3+)

**Optional enhancements:**

1. **Undo/redo** (4 hours)
   - Parameter history stack
   - Quick navigation between parameter sets

2. **Parameter presets** (3 hours)
   - Save/load named parameter configurations
   - Export tuned parameters to config file

3. **A/B comparison** (4 hours)
   - Side-by-side view of two parameter sets
   - Diff highlighting for changed parameters

4. **Batch export** (3 hours)
   - Apply tuned parameters to new folder
   - Parallel processing with progress tracking

5. **Upgrade to Approach 3 if needed** (3-5 days)
   - Integrate Salsa for complex parameter interactions
   - Keep simple cache for expensive operations

---

## Technical Debt and Future Considerations

### Known Limitations

1. **Parameter ranges:** Current proposal doesn't address parameter validation or UI range limits
2. **Persistence:** No specification for saving cache across application restarts
3. **Concurrency:** Limited discussion of thread safety for cache access
4. **Memory bounds:** Need policy for maximum cache size (important for large folders)

### Future Enhancements

1. **GPU acceleration:** ONNX inference on GPU (requires different caching strategy)
2. **Progressive loading:** Load and cache images lazily as user scrolls
3. **Differential quality:** Show how parameter changes affect specific images
4. **Smart visualization:** Cache only visible heatmaps, render to in-memory buffers
5. **Auto-tuning:** ML-based parameter suggestion for specific image types
6. **Heatmap optimization:** Extract rendering logic from debug code for faster GUI updates

### Migration Path

If starting with Approach 2 and later needing Approach 3:

1. Keep cache key generation and MD5 logic
2. Extract stage functions (already done in Approach 2)
3. Add Salsa as dependency
4. Wrap heuristic functions in `#[salsa::tracked]`
5. Keep expensive cache layer unchanged
6. Gradually convert stages to Salsa queries

**Estimated migration effort:** 2-3 days (much less than building Approach 3 from scratch)

---

## Conclusion

For enabling GUI slider-based quality algorithm tuning:

1. **Start simple:** Use **Approach 1** for 1-2 day prototype to validate concept
2. **Production implementation:** Refactor to **Approach 2** for 2-3 day robust solution
3. **Scale up:** Upgrade to **Approach 3** only if complexity demands it (3-5 days)
4. **Avoid premature optimization:** Approach 4 is overkill for this use case

The staged computation approach (Approach 2) hits the sweet spot of:
- Fast enough (<20ms typical slider response for 100 images)
- Simple enough (2-3 days implementation)
- Robust enough (automatic memoization, LRU caching)
- Extensible enough (easy to add stages or upgrade to Salsa)

### Key Architectural Principles

**1. Three-Level Caching Strategy** (See [GUI Scenario Analysis](./2025-10-26-gui-scenario-analysis.md)):
- **Level 1:** Cache expensive ML (preprocessing, ONNX, raw Tenengrad) - parameter-independent
- **Level 2:** Recompute cheap heuristics on every slider change (~10ms for 100 images)
- **Level 3:** Render visualizations on-demand for visible images only (~7-30ms)

**2. Separate Computation from Visualization:**
- Quality scoring must be instant (10ms for 100 images)
- Heatmaps can lag slightly (17-38ms total with progressive rendering)
- This separation enables instant feedback while visualizations load

**3. Progressive Rendering:**
- Update ranking immediately (user sees instant response)
- Render main heatmap next (17-18ms total)
- Update sidebar thumbnails in background (37-38ms total)

**Result:** 17-38ms end-to-end response time - well under 50ms threshold for "instant" feel!

This separation of concerns enables fast GUI responsiveness without over-engineering the caching layer.
