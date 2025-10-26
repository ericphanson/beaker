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

## Target Architecture: Layered Quality Assessment

### Core Principle

> **Separate what changes from what doesn't**

The library should expose three natural layers that frontends can cache independently:

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

### CLI Usage (Current Behavior)

```rust
// Command: beaker quality image.jpg --metadata

pub fn quality_command(config: QualityConfig) -> Result<()> {
    let session = load_onnx_session(&config)?;

    for image_path in &config.sources {
        let img = image::open(image_path)?;

        // Compute Layer 1 (expensive)
        let raw = QualityRawData::compute_from_image(
            &img,
            &session,
            "quality-model-v1".to_string(),
        )?;

        // Compute Layer 2 with default parameters
        let scores = QualityScores::compute(&raw, &QualityParams::default());

        // Write metadata
        if config.metadata {
            write_metadata(image_path, &raw, &scores)?;
        }

        // Optional: Generate debug visualizations
        if config.debug_dump_images {
            let viz = QualityVisualization::render(
                &raw,
                &scores,
                &HeatmapStyle::default(),
            )?;
            save_debug_images(image_path, &viz)?;
        }

        println!("Quality: {:.1}", scores.final_score);
    }

    Ok(())
}
```

**Benefits:**
- Same user experience as current
- But now cleanly separated into layers
- Can add `--params` flag to tune from CLI

### GUI Usage (High-Performance)

```rust
// GUI quality triage with 100 images

pub struct QualityGuiState {
    // Level 1 cache: Never invalidated
    raw_data: HashMap<PathBuf, QualityRawData>,

    // Level 2 cache: Invalidated when params change
    scores: HashMap<PathBuf, QualityScores>,

    // Level 3 cache: LRU cache for visible images only
    visualizations: LruCache<PathBuf, QualityVisualization>,

    // Current parameters (adjustable via sliders)
    params: QualityParams,
}

impl QualityGuiState {
    /// Initial load: Populate Level 1 cache (parallel)
    pub async fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        let session = load_onnx_session()?;

        // Parallel computation of expensive Layer 1
        let raw_results: Vec<_> = paths.par_iter()
            .map(|path| {
                let img = image::open(path)?;
                let raw = QualityRawData::compute_from_image(
                    &img,
                    &session,
                    "quality-model-v1".to_string(),
                )?;
                Ok((path.clone(), raw))
            })
            .collect::<Result<_>>()?;

        // Populate Level 1 cache
        self.raw_data = raw_results.into_iter().collect();

        // Compute initial Level 2 scores
        self.recompute_all_scores();

        Ok(())
    }

    /// Slider changed: Recompute Level 2 (instant: ~10ms for 100 images)
    pub fn update_params(&mut self, new_params: QualityParams) {
        self.params = new_params;
        self.recompute_all_scores();
        self.visualizations.clear(); // Invalidate Level 3 cache
    }

    /// Recompute all quality scores (cheap)
    fn recompute_all_scores(&mut self) {
        self.scores = self.raw_data.iter()
            .map(|(path, raw)| {
                let scores = QualityScores::compute(raw, &self.params);
                (path.clone(), scores)
            })
            .collect();
    }

    /// Get ranked images (instant after recompute_all_scores)
    pub fn get_ranked_images(&self) -> Vec<(PathBuf, f32)> {
        let mut ranked: Vec<_> = self.scores.iter()
            .map(|(path, scores)| (path.clone(), scores.final_score))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }

    /// Render visualization for specific image (moderate: 7-30ms)
    pub fn get_visualization(&mut self, path: &Path) -> Result<&QualityVisualization> {
        // Check Level 3 cache
        if !self.visualizations.contains(path) {
            // Render on-demand
            let raw = self.raw_data.get(path)
                .ok_or_else(|| anyhow!("Image not in cache"))?;
            let scores = self.scores.get(path)
                .ok_or_else(|| anyhow!("Scores not computed"))?;

            let viz = QualityVisualization::render(
                raw,
                scores,
                &HeatmapStyle::default(),
            )?;

            self.visualizations.put(path.to_path_buf(), viz);
        }

        Ok(self.visualizations.get(path).unwrap())
    }
}

// GUI event loop
async fn on_slider_change(state: &mut QualityGuiState, new_params: QualityParams) {
    // Phase 1: Instant (10ms)
    state.update_params(new_params);
    let ranked = state.get_ranked_images();

    // Update UI immediately - user sees ranking change
    ui.update_image_list(ranked.clone());

    // Phase 2: Fast (7-8ms)
    let top_image = &ranked[0].0;
    let viz = state.get_visualization(top_image).await?;
    ui.update_main_view(top_image, viz);

    // Phase 3: Background (15-20ms)
    tokio::spawn(async move {
        for (path, _) in ranked.iter().skip(1).take(5) {
            let viz = state.get_visualization(path).await?;
            ui.update_thumbnail(path, viz);
        }
    });
}
```

**Benefits:**
- Natural three-level caching
- No GUI-specific code in library
- GUI just uses library API efficiently
- Easy to understand and maintain

### API Usage (Server)

```rust
// REST API with caching

pub struct QualityApiServer {
    // Persistent Level 1 cache across requests
    cache: Arc<RwLock<HashMap<String, QualityRawData>>>,
    session: Arc<Session>,
}

impl QualityApiServer {
    async fn assess_quality(
        &self,
        image_url: String,
        params: Option<QualityParams>,
    ) -> Result<QualityScores> {
        let params = params.unwrap_or_default();

        // Check Level 1 cache
        let cache_key = format!("{}-{}", image_url, hash_of_image);

        let raw = {
            let cache = self.cache.read().await;
            if let Some(raw) = cache.get(&cache_key) {
                raw.clone()
            } else {
                drop(cache);

                // Compute and cache
                let img = download_image(&image_url).await?;
                let raw = QualityRawData::compute_from_image(
                    &img,
                    &self.session,
                    "quality-model-v1".to_string(),
                )?;

                let mut cache = self.cache.write().await;
                cache.insert(cache_key.clone(), raw.clone());
                raw
            }
        };

        // Compute Level 2 (always fresh with current params)
        Ok(QualityScores::compute(&raw, &params))
    }
}
```

**Benefits:**
- Server caches Level 1 across requests
- Can tune parameters per request
- No need to recompute expensive operations

---

## Implementation Plan

### Phase 1: Data Structure Refactoring (Week 1)

**Goal:** Introduce new data structures without breaking existing code

1. **Create new data structures**
   - `QualityRawData`
   - `QualityParams`
   - `QualityScores`
   - `HeatmapStyle`

2. **Add helper functions**
   - `compute_raw_tenengrad()`
   - `apply_tenengrad_params()`
   - `fuse_probabilities()`
   - `compute_weights()`

3. **Keep existing API working**
   - Make `blur_weights_from_nchw()` call new functions internally
   - No CLI changes yet

**Files to modify:**
- `beaker/src/blur_detection.rs` - Add new functions
- `beaker/src/quality_processing.rs` - Add new data structures

**Testing:**
- Existing tests should pass unchanged
- Add unit tests for new functions

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

### Phase 3: Parameterize Quality Assessment (Week 3)

**Goal:** Make parameters runtime-configurable

1. **Add parameter support to CLI**
   - `--alpha`, `--beta`, `--tau`, etc. flags
   - Or `--params params.toml` to load from file

2. **Update quality processing**
   - Accept `QualityParams` as argument
   - Use in computation instead of constants

3. **Backward compatibility**
   - Default parameters match current constants
   - No behavior change if flags not used

**Files to modify:**
- `beaker/src/config.rs` - Add parameter flags
- `beaker/src/quality_processing.rs` - Accept parameters
- `beaker/src/blur_detection.rs` - Use parameters

**Testing:**
- Default behavior unchanged
- Custom parameters produce different results
- Regression tests with known images

### Phase 4: Layer-Based API (Week 4)

**Goal:** Expose clean layer-based API

1. **Add public API functions**
   ```rust
   // In quality_processing.rs
   pub fn compute_quality_raw(...) -> QualityRawData
   pub fn compute_quality_scores(...) -> QualityScores
   pub fn render_quality_visualization(...) -> QualityVisualization
   ```

2. **Add caching helpers**
   ```rust
   impl QualityRawData {
       pub fn save_to_cache(&self, path: &Path) -> Result<()>
       pub fn load_from_cache(path: &Path) -> Result<Self>
   }
   ```

3. **Documentation**
   - API documentation with examples
   - Usage patterns for different frontends
   - Performance characteristics

**Files to modify:**
- `beaker/src/quality_processing.rs` - Public API
- `beaker/src/lib.rs` - Export new types
- `beaker/README.md` - Usage examples

**Testing:**
- Integration tests for full workflow
- Example CLI using new API
- Benchmark caching performance

### Phase 5: GUI Integration (Week 5+)

**Goal:** Build GUI using new API

1. **GUI state management**
   - Implement three-level caching
   - Progressive rendering
   - Parameter sliders

2. **Benchmark real-world performance**
   - 100 images, measure actual timings
   - Verify <50ms slider response

3. **Polish**
   - Smooth transitions
   - Loading indicators
   - Error handling

**New files:**
- `beaker-gui/src/quality_state.rs` - Caching logic
- `beaker-gui/src/quality_view.rs` - UI components

---

## Success Criteria

### For Library

1. **Clean separation:** Three distinct layers with clear responsibilities
2. **No GUI coupling:** Library has no GUI-specific code
3. **Backward compatible:** Existing CLI works unchanged
4. **Well documented:** Clear API with usage examples
5. **Tested:** Unit tests for each layer, integration tests for workflow

### For Performance

1. **Level 1 computation:** 57-70ms per image (unchanged from current)
2. **Level 2 computation:** <0.1ms per image (100 images in 10ms)
3. **Level 3 rendering:** 7-30ms depending on what's rendered
4. **Memory usage:** ~600KB per image Level 1, ~3KB Level 2, ~600KB Level 3 (for visible)

### For GUI

1. **Slider response:** 17-38ms total for 100 images
2. **Progressive rendering:** Ranking updates in 10ms, heatmaps follow
3. **Smooth UX:** No perceptible lag when adjusting parameters

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

**Design Principle:**
> Separate what changes from what doesn't. Cache aggressively. Compute lazily.

**Library Structure:**
- **Layer 1:** Expensive, parameter-independent → Compute once, cache forever
- **Layer 2:** Cheap, parameter-dependent → Recompute freely
- **Layer 3:** Moderate, visual → Render on-demand for visible items

**No GUI Hooks:**
- Library exposes clean, layered API
- GUI uses API naturally for high performance
- Same API works for CLI, server, batch processing

**Result:**
- Instant-feeling GUI (17-38ms response)
- Clean, maintainable codebase
- Flexible for future frontends
