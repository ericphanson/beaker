# Quality Library Architecture: Code Implementation Plan

**Date:** 2025-10-26
**Goal:** Design library architecture that naturally supports high-performance frontends (GUI, CLI, API)

## Current State Analysis

### Current Code Structure

**`quality_processing.rs`:**
- Mixes preprocessing, inference, and parameter application
- Returns final `QualityResult` with all computed values
- No separation of cacheable layers

**`blur_detection.rs`:**
- `blur_weights_from_nchw()` mixes raw computation with parameter application
- Uses global constants for parameters (ALPHA, BETA, etc.)
- Combines computation with visualization (debug heatmaps)

### Problems with Current Design

1. **Monolithic computation:** Cannot cache intermediate results
2. **Hardcoded parameters:** Constants prevent runtime tuning
3. **Mixed concerns:** Computation and visualization coupled
4. **No layer separation:** Can't selectively recompute based on what changed

## Target Architecture: Layered Quality Assessment with Transparent Caching

### Core Principles

> **Separate what changes from what doesn't**
>
> **Cache transparently - frontends shouldn't implement caching**
>
> **Breaking changes are fine - optimize for best design**

The library provides three natural layers with built-in transparent caching:

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Raw Computation (Parameter-Independent)│
│  - Preprocessing                                 │
│  - ONNX inference                                │
│  - Raw Tenengrad scores                         │
│  Cost: 57-70ms | Cache: Forever                 │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Layer 2: Heuristic Scoring (Parameter-Dependent)│
│  - Blur probability mapping                      │
│  - Weight computation                            │
│  - Final quality scores                          │
│  Cost: <0.1ms | Cache: Until params change      │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Layer 3: Visualization (Parameter-Dependent)     │
│  - Heatmap rendering                             │
│  - Overlay composition                           │
│  - Colormap application                          │
│  Cost: 7-30ms | Cache: Until params/view changes│
└─────────────────────────────────────────────────┘
```

### Transparent Caching Architecture

The library implements transparent multi-level caching. Frontends just call functions; caching happens automatically.

**Design Goals:**
1. **Zero boilerplate** - Frontends don't implement caching logic
2. **Intelligent invalidation** - Cache invalidates automatically when inputs change
3. **Configurable** - Memory limits, disk persistence, TTL
4. **Thread-safe** - Concurrent access from GUI threads
5. **Observable** - Cache hits/misses for debugging

**Cache Manager:**

```rust
/// Central cache manager (singleton or passed to functions)
pub struct QualityCache {
    /// Level 1: Raw computation cache (in-memory + disk)
    raw_cache: Arc<RwLock<LruCache<ImageKey, QualityRawData>>>,
    raw_disk_cache: Option<DiskCache<ImageKey, QualityRawData>>,

    /// Level 2: Score cache (in-memory only, cheap to recompute)
    score_cache: Arc<RwLock<HashMap<(ImageKey, QualityParams), QualityScores>>>,

    /// Level 3: Visualization cache (in-memory, LRU)
    viz_cache: Arc<RwLock<LruCache<(ImageKey, QualityParams, HeatmapStyle), QualityVisualization>>>,

    /// ONNX session (shared across threads)
    session: Arc<Session>,

    /// Configuration
    config: CacheConfig,
}

/// Cache key for images (content-addressed)
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct ImageKey {
    path: PathBuf,
    content_hash: u64,  // Hash of image bytes
}

/// Cache configuration
pub struct CacheConfig {
    /// Max memory for Level 1 cache (default: 1GB)
    max_raw_cache_mb: usize,

    /// Max memory for Level 3 cache (default: 100MB)
    max_viz_cache_mb: usize,

    /// Disk cache directory (default: ~/.cache/beaker/quality)
    disk_cache_dir: Option<PathBuf>,

    /// Enable disk persistence for Level 1
    enable_disk_cache: bool,

    /// Level 2 cache size (default: unlimited, cheap to recompute)
    max_score_cache_entries: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_raw_cache_mb: 1024,
            max_viz_cache_mb: 100,
            disk_cache_dir: Some(dirs::cache_dir()?.join("beaker/quality")),
            enable_disk_cache: true,
            max_score_cache_entries: None,
        }
    }
}
```

**Usage Pattern (Transparent to Frontend):**

```rust
// Frontend code - caching is automatic!
let cache = QualityCache::new(CacheConfig::default())?;

// First call: Cache miss, computes everything (~60ms)
let scores = cache.assess_quality("image1.jpg", &params)?;

// Second call with same params: Cache hit (<1ms)
let scores = cache.assess_quality("image1.jpg", &params)?;

// Changed params: Level 1 cached, Level 2 recomputed (<0.1ms)
let new_params = QualityParams { alpha: 0.8, ..params };
let scores = cache.assess_quality("image1.jpg", &new_params)?;

// Visualization: Renders only if not cached
let viz = cache.visualize("image1.jpg", &params, &style)?;
```

**Cache Invalidation Strategy:**

```rust
impl QualityCache {
    /// Automatically invalidates based on changes
    fn assess_quality(
        &self,
        image_path: impl AsRef<Path>,
        params: &QualityParams,
    ) -> Result<QualityScores> {
        let key = self.compute_image_key(image_path.as_ref())?;

        // Level 1: Check raw cache
        let raw = self.get_or_compute_raw(&key)?;

        // Level 2: Check score cache
        let cache_key = (key.clone(), params.clone());
        {
            let cache = self.score_cache.read().unwrap();
            if let Some(scores) = cache.get(&cache_key) {
                return Ok(scores.clone());  // Cache hit!
            }
        }

        // Cache miss: Compute scores from raw data
        let scores = QualityScores::compute(&raw, params);

        // Store in cache
        {
            let mut cache = self.score_cache.write().unwrap();
            cache.insert(cache_key, scores.clone());
        }

        Ok(scores)
    }

    /// Get raw data from cache or compute
    fn get_or_compute_raw(&self, key: &ImageKey) -> Result<QualityRawData> {
        // Check in-memory cache
        {
            let cache = self.raw_cache.read().unwrap();
            if let Some(raw) = cache.get(key) {
                return Ok(raw.clone());
            }
        }

        // Check disk cache
        if let Some(disk_cache) = &self.raw_disk_cache {
            if let Ok(Some(raw)) = disk_cache.get(key) {
                // Populate in-memory cache
                let mut cache = self.raw_cache.write().unwrap();
                cache.put(key.clone(), raw.clone());
                return Ok(raw);
            }
        }

        // Cache miss: Compute from scratch
        let img = image::open(&key.path)?;
        let raw = QualityRawData::compute_from_image(
            &img,
            &self.session,
            "quality-model-v1".to_string(),
        )?;

        // Store in both caches
        {
            let mut cache = self.raw_cache.write().unwrap();
            cache.put(key.clone(), raw.clone());
        }

        if let Some(disk_cache) = &self.raw_disk_cache {
            disk_cache.put(key, &raw)?;
        }

        Ok(raw)
    }

    /// Clear Level 2 cache (when parameters change globally)
    pub fn invalidate_scores(&self) {
        let mut cache = self.score_cache.write().unwrap();
        cache.clear();
    }

    /// Clear Level 3 cache (when visualization style changes)
    pub fn invalidate_visualizations(&self) {
        let mut cache = self.viz_cache.write().unwrap();
        cache.clear();
    }
}
```

**Content-Addressed Caching:**

```rust
impl QualityCache {
    /// Compute cache key from image path
    /// Uses content hash to detect file changes
    fn compute_image_key(&self, path: &Path) -> Result<ImageKey> {
        let bytes = std::fs::read(path)?;
        let content_hash = hash_bytes(&bytes);

        Ok(ImageKey {
            path: path.to_path_buf(),
            content_hash,
        })
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
```

**Disk Cache Implementation:**

```rust
/// Persistent disk cache for expensive computation results
pub struct DiskCache<K, V> {
    cache_dir: PathBuf,
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V> DiskCache<K, V>
where
    K: Hash,
    V: Serialize + for<'de> Deserialize<'de>,
{
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            cache_dir,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let path = self.key_to_path(key);
        if !path.exists() {
            return Ok(None);
        }

        let bytes = std::fs::read(&path)?;
        let value: V = bincode::deserialize(&bytes)?;
        Ok(Some(value))
    }

    pub fn put(&self, key: &K, value: &V) -> Result<()> {
        let path = self.key_to_path(key);
        let bytes = bincode::serialize(value)?;
        std::fs::write(&path, bytes)?;
        Ok(())
    }

    fn key_to_path(&self, key: &K) -> PathBuf {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.cache_dir.join(format!("{:016x}.cache", hash))
    }
}
```

---

## Data Structure Design

### Layer 1: Raw Computation Results

```rust
/// Parameter-independent computation results
/// These are EXPENSIVE to compute but NEVER change for a given image
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityRawData {
    /// Image metadata
    pub input_width: u32,
    pub input_height: u32,

    /// Preprocessed image tensor (224x224x3, normalized)
    /// Store this to avoid re-preprocessing
    #[serde(skip)]
    pub preprocessed_tensor: Array4<f32>,

    /// ONNX model outputs (parameter-independent)
    pub paq2piq_global: f32,           // Global quality score (0-100)
    pub paq2piq_local: [[u8; 20]; 20], // 20x20 local quality grid

    /// Raw blur detection (parameter-independent)
    pub tenengrad_224: [[f32; 20]; 20],  // Raw Tenengrad at 224x224
    pub tenengrad_112: [[f32; 20]; 20],  // Raw Tenengrad at 112x112
    pub median_tenengrad_224: f32,       // Median for adaptive thresholding
    pub scale_ratio: f32,                // 112/224 scale ratio

    /// Provenance
    pub model_version: String,
    pub computed_at: std::time::SystemTime,
}

impl QualityRawData {
    /// Compute from image (expensive: ~60ms)
    pub fn compute_from_image(
        img: &DynamicImage,
        session: &Session,
        model_version: String,
    ) -> Result<Self> {
        // Preprocessing
        let preprocessed_tensor = preprocess_image_for_quality(img)?;

        // ONNX inference
        let (paq2piq_global, paq2piq_local) =
            run_onnx_inference(session, &preprocessed_tensor)?;

        // Raw blur detection (without parameters)
        let blur_raw = compute_raw_tenengrad(&preprocessed_tensor)?;

        Ok(Self {
            input_width: img.width(),
            input_height: img.height(),
            preprocessed_tensor,
            paq2piq_global,
            paq2piq_local,
            tenengrad_224: blur_raw.t224,
            tenengrad_112: blur_raw.t112,
            median_tenengrad_224: blur_raw.median_224,
            scale_ratio: blur_raw.scale_ratio,
            model_version,
            computed_at: std::time::SystemTime::now(),
        })
    }

    /// Serialize for caching (exclude tensor to save space)
    pub fn to_cache_bytes(&self) -> Result<Vec<u8>> {
        // Serialize everything except preprocessed_tensor
        bincode::serialize(self)
    }

    /// Deserialize from cache
    pub fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
    }
}

/// Memory footprint: ~3KB without tensor, ~600KB with tensor
```

### Layer 2: Heuristic Parameters & Scores

```rust
/// Tunable parameters for quality heuristics
/// These are what users adjust via GUI sliders
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityParams {
    /// Blur probability mapping
    pub beta: f32,           // Default: 1.2, Range: [0.5, 2.0]
    pub tau_ten_224: f32,    // Default: 0.02, Range: [0.001, 0.1]
    pub p_floor: f32,        // Default: 0.0, Range: [0.0, 0.1]

    /// Weight computation
    pub alpha: f32,          // Default: 0.7, Range: [0.0, 1.0]
    pub min_weight: f32,     // Default: 0.2, Range: [0.0, 0.5]

    /// Per-detection parameters (optional)
    pub s_ref: f32,          // Default: 96.0
    pub cov_ref: f32,        // Default: 4.0
    pub core_ratio: f32,     // Default: 0.60
}

impl Default for QualityParams {
    fn default() -> Self {
        Self {
            beta: 1.2,
            tau_ten_224: 0.02,
            p_floor: 0.0,
            alpha: 0.7,
            min_weight: 0.2,
            s_ref: 96.0,
            cov_ref: 4.0,
            core_ratio: 0.60,
        }
    }
}

/// Parameter-dependent quality scores
/// These are CHEAP to compute from QualityRawData
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityScores {
    /// Final quality score (combining paq2piq and blur)
    pub final_score: f32,

    /// Component scores (for analysis)
    pub paq2piq_score: f32,
    pub blur_score: f32,      // Global blur probability (0-1)

    /// Intermediate results (20x20 grids)
    pub blur_probability: [[f32; 20]; 20],  // Fused blur probability
    pub blur_weights: [[f32; 20]; 20],      // Weights (1 - alpha*P)

    /// Parameters used to compute these scores
    pub params: QualityParams,
}

impl QualityScores {
    /// Compute from raw data and parameters (cheap: <0.1ms)
    pub fn compute(raw: &QualityRawData, params: &QualityParams) -> Self {
        // Apply parameters to raw Tenengrad to get probabilities
        let (p224, p112) = apply_tenengrad_params(
            &raw.tenengrad_224,
            &raw.tenengrad_112,
            raw.median_tenengrad_224,
            raw.scale_ratio,
            params,
        );

        // Fuse probabilities
        let blur_probability = fuse_probabilities(&p224, &p112);

        // Compute weights
        let blur_weights = compute_weights(&blur_probability, params);

        // Global blur score
        let blur_score = blur_probability.iter()
            .flat_map(|row| row.iter())
            .sum::<f32>() / 400.0;

        // Final combined score
        let w_mean = (1.0 - params.alpha * blur_score)
            .clamp(params.min_weight, 1.0);
        let final_score = raw.paq2piq_global * w_mean;

        Self {
            final_score,
            paq2piq_score: raw.paq2piq_global,
            blur_score,
            blur_probability,
            blur_weights,
            params: params.clone(),
        }
    }
}

/// Memory footprint: ~3KB
```

### Layer 3: Visualization Data

```rust
/// Heatmap rendering options
#[derive(Clone, Debug)]
pub struct HeatmapStyle {
    pub colormap: ColorMap,
    pub alpha: f32,          // Overlay transparency
    pub size: (u32, u32),    // Target size (can be smaller for thumbnails)
}

#[derive(Clone, Copy, Debug)]
pub enum ColorMap {
    Viridis,
    Plasma,
    Inferno,
    Turbo,
    Grayscale,
}

/// Visualization layer (parameter-dependent, rendered on-demand)
pub struct QualityVisualization {
    /// Heatmap images (rendered to in-memory buffers)
    pub blur_probability_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    pub blur_weights_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    pub tenengrad_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,

    /// Overlay on original image
    pub blur_overlay: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl QualityVisualization {
    /// Render heatmaps from scores (moderate: 7-30ms depending on what's rendered)
    pub fn render(
        raw: &QualityRawData,
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<Self> {
        Ok(Self {
            blur_probability_heatmap: Some(
                render_heatmap_to_buffer(&scores.blur_probability, style)?
            ),
            blur_weights_heatmap: Some(
                render_heatmap_to_buffer(&scores.blur_weights, style)?
            ),
            tenengrad_heatmap: Some(
                render_heatmap_to_buffer(&raw.tenengrad_224, style)?
            ),
            blur_overlay: None, // Render lazily if needed
        })
    }

    /// Render only blur probability heatmap (fast: ~3-4ms)
    pub fn render_blur_only(
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        render_heatmap_to_buffer(&scores.blur_probability, style)
    }
}

/// Memory footprint: ~600KB per rendered heatmap at 224x224
```

---

## Core Library Functions

### Refactored `blur_detection.rs`

**Current function:**
```rust
pub fn blur_weights_from_nchw(
    x: &Array4<f32>,
    out_dir: Option<PathBuf>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, f32)
```

**New structure:**

```rust
// ============================================================================
// Layer 1: Parameter-Independent Computation
// ============================================================================

/// Raw Tenengrad computation results (parameter-independent)
#[derive(Clone, Debug)]
pub struct RawTenengradData {
    pub t224: Array2<f32>,      // 20x20 Tenengrad scores at 224x224
    pub t112: Array2<f32>,      // 20x20 Tenengrad scores at 112x112
    pub median_224: f32,        // Median for adaptive thresholding
    pub scale_ratio: f32,       // Scale ratio (112/224)
}

/// Compute raw Tenengrad scores (expensive: ~2ms)
/// This is parameter-independent - compute once, cache forever
pub fn compute_raw_tenengrad(x: &Array4<f32>) -> Result<RawTenengradData> {
    let gray224 = nchw_to_gray_224(x);
    let t224 = tenengrad_mean_grid_20(&gray224);

    let gray112 = downsample_2x_gray_f32(&gray224);
    let t112 = tenengrad_mean_grid_20(&gray112);

    // Compute median and scale ratio
    let mut v224: Vec<f32> = t224.iter().copied().collect();
    v224.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_224 = v224[v224.len() / 2].max(1e-12);

    let mut v112: Vec<f32> = t112.iter().copied().collect();
    v112.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_112 = v112[v112.len() / 2];

    let scale_ratio = if median_224 > 0.0 {
        (median_112 / median_224).clamp(0.05, 0.80)
    } else {
        0.25
    };

    Ok(RawTenengradData {
        t224,
        t112,
        median_224,
        scale_ratio,
    })
}

// ============================================================================
// Layer 2: Parameter-Dependent Computation
// ============================================================================

/// Apply parameters to raw Tenengrad to get blur probabilities (cheap: <0.1ms)
pub fn apply_tenengrad_params(
    t224: &Array2<f32>,
    t112: &Array2<f32>,
    median_224: f32,
    scale_ratio: f32,
    params: &QualityParams,
) -> (Array2<f32>, Array2<f32>) {
    const EPS_T: f32 = 1e-12;
    const BIAS112: f32 = 1.25;

    // Apply parameters to 224 Tenengrad
    let p224 = t224.mapv(|t| {
        let tau = params.tau_ten_224.max(EPS_T);
        let p = (tau / (t + tau)).powf(params.beta);
        (p + params.p_floor).min(1.0)
    });

    // Apply parameters to 112 Tenengrad
    let tau112 = params.tau_ten_224 * scale_ratio * BIAS112;
    let p112 = t112.mapv(|t| {
        let tau = tau112.max(EPS_T);
        let p = (tau / (t + tau)).powf(params.beta);
        (p + params.p_floor).min(1.0)
    });

    (p224, p112)
}

/// Fuse two probability maps (probabilistic OR)
pub fn fuse_probabilities(
    p224: &Array2<f32>,
    p112: &Array2<f32>,
) -> Array2<f32> {
    let mut p = Array2::<f32>::zeros((20, 20));
    ndarray::Zip::from(&mut p)
        .and(p224)
        .and(p112)
        .for_each(|p_elem, &a, &b| {
            *p_elem = 1.0 - (1.0 - a) * (1.0 - b);
        });
    p
}

/// Compute blur weights from probabilities
pub fn compute_weights(
    blur_probability: &Array2<f32>,
    params: &QualityParams,
) -> Array2<f32> {
    blur_probability.mapv(|p| {
        (1.0 - params.alpha * p).clamp(params.min_weight, 1.0)
    })
}

// ============================================================================
// Layer 3: Visualization
// ============================================================================

/// Render heatmap to in-memory buffer (moderate: ~3-4ms)
pub fn render_heatmap_to_buffer(
    data: &Array2<f32>,
    style: &HeatmapStyle,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    assert_eq!(data.shape(), &[20, 20]);

    let (width, height) = style.size;
    let mut img = ImageBuffer::new(width, height);

    // Upscale 20x20 to target size with bilinear interpolation
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 / width as f32) * 20.0;
            let v = (y as f32 / height as f32) * 20.0;

            let value = bilinear_sample(data, u, v);
            let color = apply_colormap(value, style.colormap);

            img.put_pixel(x, y, color);
        }
    }

    Ok(img)
}

/// Render overlay (composite heatmap over original image)
pub fn render_overlay(
    original: &DynamicImage,
    heatmap_data: &Array2<f32>,
    style: &HeatmapStyle,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let heatmap = render_heatmap_to_buffer(heatmap_data, style)?;

    // Resize heatmap to match original image size
    let heatmap_resized = image::imageops::resize(
        &heatmap,
        original.width(),
        original.height(),
        image::imageops::FilterType::Lanczos3,
    );

    // Composite with alpha blending
    composite_with_alpha(&original.to_rgba8(), &heatmap_resized, style.alpha)
}
```

---

## Frontend Usage Patterns

### CLI Usage (With Transparent Caching)

```rust
// Command: beaker quality image.jpg --metadata

pub fn quality_command(config: QualityConfig) -> Result<()> {
    // Initialize cache (automatic disk persistence)
    let cache_config = CacheConfig {
        enable_disk_cache: !config.no_cache,
        ..Default::default()
    };
    let cache = QualityCache::new(cache_config)?;

    let params = config.params.unwrap_or_default();

    for image_path in &config.sources {
        // Transparent caching - all layers handled automatically
        let scores = cache.assess_quality(image_path, &params)?;

        // Write metadata
        if config.metadata {
            write_metadata(image_path, &scores)?;
        }

        // Optional: Generate debug visualizations
        if config.debug_dump_images {
            let style = HeatmapStyle::default();
            let viz = cache.visualize(image_path, &params, &style)?;
            save_debug_images(image_path, &viz)?;
        }

        println!("Quality: {:.1}", scores.final_score);
    }

    // Print cache statistics if verbose
    if config.verbose {
        let stats = cache.stats();
        println!("\nCache Statistics:");
        println!("  Level 1 hits: {}/{}", stats.l1_hits, stats.l1_total);
        println!("  Level 2 hits: {}/{}", stats.l2_hits, stats.l2_total);
        println!("  Level 3 hits: {}/{}", stats.l3_hits, stats.l3_total);
    }

    Ok(())
}
```

**Benefits:**
- Automatic disk caching - second run is instant for Level 1
- Same user experience as current
- Can add `--params` flag to tune from CLI
- Cache invalidates automatically when files change

### GUI Usage (High-Performance with Transparent Caching)

```rust
// GUI quality triage with 100 images

pub struct QualityGuiState {
    /// Transparent cache manager (handles all 3 levels automatically)
    cache: Arc<QualityCache>,

    /// Current parameters (adjustable via sliders)
    params: QualityParams,

    /// Image paths in current folder
    paths: Vec<PathBuf>,

    /// Current ranking (path, score)
    ranked: Vec<(PathBuf, f32)>,

    /// Heatmap style
    style: HeatmapStyle,
}

impl QualityGuiState {
    /// Initialize with cache configuration
    pub fn new() -> Result<Self> {
        let cache_config = CacheConfig {
            max_raw_cache_mb: 1024,      // 1GB for Level 1
            max_viz_cache_mb: 100,       // 100MB for Level 3
            enable_disk_cache: true,     // Persist across sessions
            ..Default::default()
        };

        Ok(Self {
            cache: Arc::new(QualityCache::new(cache_config)?),
            params: QualityParams::default(),
            paths: Vec::new(),
            ranked: Vec::new(),
            style: HeatmapStyle::default(),
        })
    }

    /// Load folder: Parallel computation with progress
    pub async fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        self.paths = paths;

        // Parallel warm-up of Level 1 cache
        // (Subsequent calls will hit disk/memory cache)
        use rayon::prelude::*;
        self.paths.par_iter().try_for_each(|path| {
            // Transparent caching - library handles everything
            let _ = self.cache.assess_quality(path, &self.params)?;
            Ok::<_, anyhow::Error>(())
        })?;

        // Compute initial ranking
        self.recompute_ranking()?;

        Ok(())
    }

    /// Slider changed: Transparent cache handles invalidation
    pub fn update_params(&mut self, new_params: QualityParams) -> Result<()> {
        self.params = new_params;
        // Cache automatically recomputes Level 2, keeps Level 1
        self.recompute_ranking()?;
        Ok(())
    }

    /// Recompute ranking (instant: ~10ms for 100 images)
    fn recompute_ranking(&mut self) -> Result<()> {
        // Transparent caching: Level 1 cached, Level 2 recomputed
        self.ranked = self.paths.par_iter()
            .filter_map(|path| {
                let scores = self.cache.assess_quality(path, &self.params).ok()?;
                Some((path.clone(), scores.final_score))
            })
            .collect();

        self.ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(())
    }

    /// Get ranked images
    pub fn get_ranked_images(&self) -> &[(PathBuf, f32)] {
        &self.ranked
    }

    /// Get visualization (transparent caching, renders on-demand)
    pub fn get_visualization(&self, path: &Path) -> Result<QualityVisualization> {
        // Transparent caching: checks Level 3 cache, renders if needed
        self.cache.visualize(path, &self.params, &self.style)
    }

    /// Batch render for visible images (thumbnails)
    pub fn get_visualizations_batch(
        &self,
        paths: &[&Path],
    ) -> Vec<Result<QualityVisualization>> {
        use rayon::prelude::*;
        paths.par_iter()
            .map(|path| self.cache.visualize(path, &self.params, &self.style))
            .collect()
    }
}

// GUI event loop
async fn on_slider_change(state: &mut QualityGuiState, new_params: QualityParams) {
    // Phase 1: Instant (10ms) - cache recomputes scores automatically
    state.update_params(new_params)?;
    let ranked = state.get_ranked_images();

    // Update UI immediately - user sees ranking change
    ui.update_image_list(ranked);

    // Phase 2: Fast (7-8ms) - cache checks Level 3, renders if needed
    let top_image = &ranked[0].0;
    let viz = state.get_visualization(top_image)?;
    ui.update_main_view(top_image, viz);

    // Phase 3: Background (15-20ms) - parallel rendering with cache
    tokio::spawn(async move {
        let thumbnail_paths: Vec<_> = ranked.iter()
            .skip(1)
            .take(5)
            .map(|(p, _)| p.as_path())
            .collect();

        let vizs = state.get_visualizations_batch(&thumbnail_paths);
        for (path, viz) in thumbnail_paths.iter().zip(vizs) {
            if let Ok(viz) = viz {
                ui.update_thumbnail(path, viz);
            }
        }
    });
}
```

**Benefits:**
- Zero cache management boilerplate - library handles everything
- Thread-safe `Arc<QualityCache>` for concurrent GUI access
- Disk persistence across sessions (instant startup on second run)
- Automatic invalidation when parameters change
- Natural three-level caching without GUI thinking about it

### API Usage (Server with Transparent Caching)

```rust
// REST API server - cache handles everything

pub struct QualityApiServer {
    /// Transparent cache (shared across all requests)
    cache: Arc<QualityCache>,
}

impl QualityApiServer {
    pub fn new() -> Result<Self> {
        let cache_config = CacheConfig {
            max_raw_cache_mb: 4096,      // 4GB for server
            enable_disk_cache: true,     // Persist across restarts
            ..Default::default()
        };

        Ok(Self {
            cache: Arc::new(QualityCache::new(cache_config)?),
        })
    }

    /// Assess image quality (transparent caching)
    async fn assess_quality(
        &self,
        image_path: impl AsRef<Path>,
        params: Option<QualityParams>,
    ) -> Result<QualityScores> {
        let params = params.unwrap_or_default();

        // Transparent caching - library handles everything
        self.cache.assess_quality(image_path, &params)
    }

    /// Get visualization for image
    async fn get_visualization(
        &self,
        image_path: impl AsRef<Path>,
        params: Option<QualityParams>,
        style: Option<HeatmapStyle>,
    ) -> Result<QualityVisualization> {
        let params = params.unwrap_or_default();
        let style = style.unwrap_or_default();

        self.cache.visualize(image_path, &params, &style)
    }

    /// Health check endpoint - return cache stats
    async fn health(&self) -> CacheStats {
        self.cache.stats()
    }
}
```

**Benefits:**
- Zero cache management code - all automatic
- Thread-safe for concurrent requests
- Disk persistence across server restarts
- Can tune parameters per request
- Built-in cache statistics for monitoring

---

## Cache Observability

### Cache Statistics

```rust
/// Cache performance statistics
#[derive(Clone, Debug)]
pub struct CacheStats {
    // Level 1 (raw data)
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l1_disk_hits: u64,
    pub l1_memory_hits: u64,
    pub l1_total: u64,
    pub l1_memory_size_mb: usize,
    pub l1_disk_size_mb: usize,

    // Level 2 (scores)
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l2_total: u64,
    pub l2_cache_size: usize,

    // Level 3 (visualizations)
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub l3_total: u64,
    pub l3_cache_size_mb: usize,

    // Performance
    pub avg_l1_compute_ms: f64,
    pub avg_l2_compute_ms: f64,
    pub avg_l3_render_ms: f64,
}

impl CacheStats {
    pub fn l1_hit_rate(&self) -> f64 {
        if self.l1_total == 0 { 0.0 }
        else { self.l1_hits as f64 / self.l1_total as f64 }
    }

    pub fn l2_hit_rate(&self) -> f64 {
        if self.l2_total == 0 { 0.0 }
        else { self.l2_hits as f64 / self.l2_total as f64 }
    }

    pub fn l3_hit_rate(&self) -> f64 {
        if self.l3_total == 0 { 0.0 }
        else { self.l3_hits as f64 / self.l3_total as f64 }
    }

    pub fn pretty_print(&self) {
        println!("Cache Statistics:");
        println!("  Level 1 (Raw Data):");
        println!("    Hit rate: {:.1}% ({}/{})",
            self.l1_hit_rate() * 100.0, self.l1_hits, self.l1_total);
        println!("    Memory hits: {}, Disk hits: {}",
            self.l1_memory_hits, self.l1_disk_hits);
        println!("    Memory: {:.1}MB, Disk: {:.1}MB",
            self.l1_memory_size_mb, self.l1_disk_size_mb);
        println!("    Avg compute time: {:.1}ms", self.avg_l1_compute_ms);

        println!("  Level 2 (Scores):");
        println!("    Hit rate: {:.1}% ({}/{})",
            self.l2_hit_rate() * 100.0, self.l2_hits, self.l2_total);
        println!("    Cache size: {} entries", self.l2_cache_size);
        println!("    Avg compute time: {:.3}ms", self.avg_l2_compute_ms);

        println!("  Level 3 (Visualizations):");
        println!("    Hit rate: {:.1}% ({}/{})",
            self.l3_hit_rate() * 100.0, self.l3_hits, self.l3_total);
        println!("    Cache size: {:.1}MB", self.l3_cache_size_mb);
        println!("    Avg render time: {:.1}ms", self.avg_l3_render_ms);
    }
}
```

### Cache Management API

```rust
impl QualityCache {
    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        // Collect stats from all cache levels
        // ...
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.raw_cache.write().unwrap().clear();
        self.score_cache.write().unwrap().clear();
        self.viz_cache.write().unwrap().clear();
    }

    /// Clear only parameter-dependent caches (L2 + L3)
    pub fn clear_param_dependent(&self) {
        self.score_cache.write().unwrap().clear();
        self.viz_cache.write().unwrap().clear();
    }

    /// Prune disk cache (remove old entries)
    pub fn prune_disk_cache(&self, max_age_days: u64) -> Result<()> {
        if let Some(disk_cache) = &self.raw_disk_cache {
            disk_cache.prune(max_age_days)?;
        }
        Ok(())
    }

    /// Pre-warm cache for a set of images
    pub fn prewarm(&self, paths: &[impl AsRef<Path>], params: &QualityParams) -> Result<()> {
        use rayon::prelude::*;
        paths.par_iter().try_for_each(|path| {
            let _ = self.assess_quality(path, params)?;
            Ok::<_, anyhow::Error>(())
        })
    }
}
```

### Usage Example

```rust
// CLI: Print cache stats after processing
let cache = QualityCache::new(CacheConfig::default())?;

for image in images {
    let scores = cache.assess_quality(image, &params)?;
    println!("{}: {:.1}", image, scores.final_score);
}

// Print statistics
cache.stats().pretty_print();
```

**Output:**
```
Cache Statistics:
  Level 1 (Raw Data):
    Hit rate: 85.0% (85/100)
    Memory hits: 60, Disk hits: 25
    Memory: 36.1MB, Disk: 180.6MB
    Avg compute time: 62.3ms
  Level 2 (Scores):
    Hit rate: 0.0% (0/100)
    Cache size: 100 entries
    Avg compute time: 0.08ms
  Level 3 (Visualizations):
    Hit rate: 40.0% (4/10)
    Cache size: 6.0MB
    Avg render time: 7.2ms
```

---

## Implementation Plan

### Phase 1: Data Structure Refactoring & Cache Infrastructure (Week 1)

**Goal:** Introduce new data structures and basic cache infrastructure

1. **Create new data structures**
   - `QualityRawData` with serialization support
   - `QualityParams` with `Hash`, `Eq`, `PartialEq` derives
   - `QualityScores`
   - `HeatmapStyle`
   - `ImageKey` with content-addressed hashing

2. **Add helper functions**
   - `compute_raw_tenengrad()`
   - `apply_tenengrad_params()`
   - `fuse_probabilities()`
   - `compute_weights()`

3. **Basic cache infrastructure**
   - `CacheConfig` with default configuration
   - `DiskCache<K, V>` generic implementation
   - Content hash function for images
   - Add dependencies: `lru`, `bincode`, `serde`

4. **Keep existing API working**
   - Make `blur_weights_from_nchw()` call new functions internally
   - No CLI changes yet

**Files to modify:**
- `beaker/src/blur_detection.rs` - Add new functions
- `beaker/src/quality_processing.rs` - Add new data structures
- Create `beaker/src/cache.rs` - Cache infrastructure
- `beaker/Cargo.toml` - Add cache dependencies

**Testing:**
- Existing tests should pass unchanged
- Add unit tests for new functions
- Test content-addressed hashing
- Test disk cache read/write

### Phase 2: Separate Computation from Visualization (Week 2)

**Goal:** Extract rendering logic from computation

1. **Create visualization module**
   - `beaker/src/quality_visualization.rs`
   - Move heatmap rendering out of `blur_detection.rs`
   - Functions: `render_heatmap_to_buffer()`, `render_overlay()`

2. **Refactor debug output**
   - Make `--debug-dump-images` use new visualization functions
   - Should work exactly as before, but via new path

3. **Add colormap support**
   - Implement different colormaps (Viridis, Plasma, etc.)
   - Make colormap configurable

**Files to modify:**
- `beaker/src/blur_detection.rs` - Extract viz functions
- Create `beaker/src/quality_visualization.rs` - New module
- `beaker/src/quality_processing.rs` - Use new viz functions

**Testing:**
- Debug images should look identical to before
- Benchmark rendering performance

### Phase 3: Transparent Caching Implementation (Week 3)

**Goal:** Implement `QualityCache` with automatic caching and invalidation

1. **Implement `QualityCache`**
   - Three-level cache structure
   - Thread-safe with `Arc<RwLock<_>>`
   - Content-addressed image keys
   - Automatic invalidation logic

2. **Cache statistics tracking**
   - `CacheStats` structure
   - Hit/miss counters for each level
   - Performance timing tracking
   - Cache size tracking

3. **Cache management functions**
   - `assess_quality()` with transparent L1+L2 caching
   - `visualize()` with L3 caching
   - `clear_all()`, `clear_param_dependent()`
   - `prewarm()` for batch processing
   - `stats()` for observability

4. **LRU eviction policies**
   - Memory-bounded L1 and L3 caches
   - Disk persistence for L1
   - Unbounded L2 (cheap to recompute)

**Files to modify:**
- `beaker/src/cache.rs` - Implement `QualityCache`
- `beaker/src/quality_processing.rs` - Integrate cache

**Testing:**
- Cache hit/miss behavior correct
- Invalidation works when params change
- Disk persistence across runs
- Thread-safety under concurrent access
- Memory bounds enforced

### Phase 4: CLI Integration & Parameterization (Week 4)

**Goal:** Integrate transparent caching into CLI and make parameters tunable

1. **Add parameter support to CLI**
   - `--alpha`, `--beta`, `--tau`, etc. flags
   - Or `--params params.toml` to load from file
   - `--no-cache` to disable caching
   - `--verbose` to show cache statistics

2. **Integrate `QualityCache` into CLI**
   - Replace direct computation with cache API
   - Show cache stats with `--verbose`
   - Add `--cache-dir` to customize cache location

3. **Update quality processing**
   - Accept `QualityParams` as argument
   - Use in computation instead of constants
   - Backward compatibility with defaults

4. **Public API functions**
   ```rust
   // Direct access (low-level)
   pub fn compute_quality_raw(...) -> QualityRawData
   pub fn compute_quality_scores(...) -> QualityScores
   pub fn render_quality_visualization(...) -> QualityVisualization

   // High-level with caching (recommended)
   impl QualityCache {
       pub fn assess_quality(...) -> QualityScores
       pub fn visualize(...) -> QualityVisualization
   }
   ```

**Files to modify:**
- `beaker/src/config.rs` - Add parameter and cache flags
- `beaker/src/quality_processing.rs` - Accept parameters, public API
- `beaker/src/blur_detection.rs` - Use parameters
- `beaker/src/commands/quality.rs` - Use `QualityCache`
- `beaker/src/lib.rs` - Export new types

**Testing:**
- Default behavior unchanged
- Custom parameters produce different results
- Cache persistence works
- Regression tests with known images
- Integration tests for full workflow

### Phase 5: GUI Integration (Week 5+)

**Goal:** Build GUI using transparent cache API

1. **GUI state management**
   - Use `Arc<QualityCache>` for shared state
   - Track current parameters and ranking
   - Progressive rendering strategy

2. **Parameter sliders**
   - Real-time parameter adjustment
   - Instant ranking update (10ms)
   - Progressive heatmap rendering (7-30ms)

3. **Benchmark real-world performance**
   - 100 images, measure actual timings
   - Verify <50ms slider response
   - Confirm cache hit rates

4. **Polish**
   - Smooth transitions
   - Loading indicators for heatmaps
   - Cache statistics display
   - Error handling

**Benefits of transparent caching:**
- GUI code is simple - just calls `cache.assess_quality()` and `cache.visualize()`
- No manual cache management in GUI layer
- Automatic invalidation when parameters change
- Thread-safe for parallel rendering

**New files:**
- `beaker-gui/src/quality_state.rs` - State management (uses `QualityCache`)
- `beaker-gui/src/quality_view.rs` - UI components
- `beaker-gui/src/quality_sliders.rs` - Parameter controls

---

## Success Criteria

### For Library

1. **Clean separation:** Three distinct layers with clear responsibilities
2. **No GUI coupling:** Library has no GUI-specific code
3. **Transparent caching:** Built into library, zero boilerplate for frontends
4. **Backward compatible:** Existing CLI works unchanged
5. **Well documented:** Clear API with usage examples
6. **Tested:** Unit tests for each layer, integration tests for workflow

### For Caching

1. **Transparent:** Frontends just call functions, caching automatic
2. **Intelligent invalidation:** Cache invalidates when inputs change
3. **Configurable:** Memory limits, disk persistence, TTL
4. **Thread-safe:** Concurrent access from multiple threads
5. **Observable:** Cache statistics available for debugging
6. **Disk persistence:** Level 1 cache survives restarts

### For Performance

1. **Level 1 computation:** 57-70ms per image (unchanged from current)
2. **Level 2 computation:** <0.1ms per image (100 images in 10ms)
3. **Level 3 rendering:** 7-30ms depending on what's rendered
4. **Memory usage:** ~600KB per image Level 1, ~3KB Level 2, ~600KB Level 3 (for visible)
5. **Cache hit rates:** >80% for Level 1 in typical workflows

### For GUI

1. **Slider response:** 17-38ms total for 100 images
2. **Progressive rendering:** Ranking updates in 10ms, heatmaps follow
3. **Smooth UX:** No perceptible lag when adjusting parameters
4. **Session persistence:** Instant startup on second run (disk cache)

---

## Migration Strategy

### For Existing Code

1. **Keep current API** - Don't break existing CLI usage
2. **Add new API alongside** - New functions with better layering
3. **Gradually deprecate** - Mark old functions as deprecated
4. **Remove in v2.0** - Clean up old code in major version bump

### For Frontends

1. **CLI** - Minimal changes, add optional parameter flags
2. **GUI** - New code using new API from start
3. **API** - Can use either old or new API

---

## Key Takeaways

**Design Principles:**
> **Separate what changes from what doesn't. Cache transparently. Compute lazily.**

1. **Transparent Caching:** Built into the library, not frontend responsibility
2. **Content-Addressed:** Cache invalidates automatically when files change
3. **Multi-Level:** Three cache levels for different cost/invalidation patterns
4. **Breaking Changes OK:** Optimize for best design, not backward compatibility

**Library Structure:**
- **Layer 1:** Expensive, parameter-independent → Compute once, cache forever (disk + memory)
- **Layer 2:** Cheap, parameter-dependent → Recompute freely (memory only)
- **Layer 3:** Moderate, visual → Render on-demand for visible items (memory LRU)

**No GUI Hooks:**
- Library exposes clean, layered API with built-in caching
- GUI uses API naturally for high performance
- Same API works for CLI, server, batch processing
- Zero cache management boilerplate in frontends

**Transparent Caching Benefits:**
- Frontends just call `cache.assess_quality()` - caching happens automatically
- Thread-safe for concurrent access
- Disk persistence across sessions
- Observable with cache statistics
- Configurable memory/disk limits

**Result:**
- Instant-feeling GUI (17-38ms response) with no manual caching
- Clean, maintainable codebase
- Flexible for future frontends
- Second run instant (disk cache hit)
