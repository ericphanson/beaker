# Quality Assessment API Guide

## Architecture

Quality assessment is split into three layers:

1. **Layer 1: Raw Computation** (expensive, parameter-independent)
   - Image preprocessing and ONNX inference (~30ms)
   - Raw Tenengrad gradient computation (~2ms)
   - Total: ~60ms per image
   - Cached automatically with `#[cached]` annotation

2. **Layer 2: Scoring** (cheap, parameter-dependent)
   - Apply parameters to raw data
   - Compute blur probabilities and weights
   - Calculate final quality score
   - Total: <0.1ms per image

3. **Layer 3: Visualization** (moderate, on-demand)
   - Render heatmaps for GUI display
   - Total: 7-30ms depending on size

## Data Structures

### QualityRawData

Parameter-independent computation results. Expensive to compute, but can be cached forever.

```rust
pub struct QualityRawData {
    pub input_width: u32,
    pub input_height: u32,
    pub paq2piq_global: f32,           // 0-100 quality score from model
    pub paq2piq_local: [[u8; 20]; 20], // 20x20 local quality grid
    pub tenengrad_224: [[f32; 20]; 20], // Raw gradient at 224x224
    pub tenengrad_112: [[f32; 20]; 20], // Raw gradient at 112x112
    pub median_tenengrad_224: f32,      // For adaptive thresholding
    pub scale_ratio: f32,               // Scale factor
    pub model_version: String,
    pub computed_at: SystemTime,
}
```

### QualityParams

Tunable parameters for blur detection heuristics.

```rust
pub struct QualityParams {
    pub beta: f32,          // Probability curve exponent (default: 1.2)
    pub tau_ten_224: f32,   // Tenengrad threshold (default: 0.02)
    pub p_floor: f32,       // Probability floor (default: 0.0)
    pub alpha: f32,         // Weight coefficient (default: 0.7)
    pub min_weight: f32,    // Minimum weight (default: 0.2)
    // ... additional parameters
}
```

### QualityScores

Parameter-dependent results computed from raw data.

```rust
pub struct QualityScores {
    pub final_score: f32,                   // Combined quality score
    pub paq2piq_score: f32,                 // ML model score
    pub blur_score: f32,                    // Global blur probability
    pub blur_probability: [[f32; 20]; 20],  // Local blur probabilities
    pub blur_weights: [[f32; 20]; 20],      // Quality weights
    pub params: QualityParams,              // Parameters used
}
```

## Usage Patterns

### CLI Application

```rust
pub fn quality_command(config: QualityConfig) -> Result<()> {
    let session = load_onnx_session_default()?;
    let params = config.params.unwrap_or_default();

    for image_path in &config.sources {
        let raw = compute_quality_raw(image_path, &session)?;
        let scores = QualityScores::compute(&raw, &params);
        println!("{}: {:.1}", image_path.display(), scores.final_score);
    }
    Ok(())
}
```

### GUI Application

```rust
pub struct QualityGuiState {
    session: Arc<Session>,
    params: QualityParams,
    raw_data: HashMap<PathBuf, QualityRawData>,
}

impl QualityGuiState {
    pub fn load_folder(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        // Parallel computation
        use rayon::prelude::*;
        let results: Vec<_> = paths.par_iter()
            .filter_map(|path| {
                let raw = compute_quality_raw(path, &self.session).ok()?;
                Some((path.clone(), raw))
            })
            .collect();

        self.raw_data = results.into_iter().collect();
        Ok(())
    }

    pub fn update_params(&mut self, new_params: QualityParams) {
        self.params = new_params;
        // Recompute scores instantly (<10ms for 100 images)
    }
}
```

### API Server

```rust
pub struct QualityApiServer {
    session: Arc<Session>,
}

impl QualityApiServer {
    pub async fn assess_quality(
        &self,
        path: &Path,
        params: Option<QualityParams>,
    ) -> Result<QualityScores> {
        let params = params.unwrap_or_default();

        // Cached automatically across requests
        let raw = compute_quality_raw(path, &self.session)?;
        Ok(QualityScores::compute(&raw, &params))
    }
}
```

## Testing

### Unit Tests

Test each layer independently:

```rust
#[test]
fn test_compute_raw_tenengrad() {
    let img = Array4::<f32>::from_elem((1, 3, 224, 224), 0.5);
    let result = compute_raw_tenengrad(&img).unwrap();
    assert_eq!(result.t224.shape(), &[20, 20]);
}

#[test]
fn test_apply_params() {
    let t224 = Array2::from_elem((20, 20), 0.05);
    let params = QualityParams::default();
    let (p224, p112) = apply_tenengrad_params(&t224, ...);
    assert!(p224.iter().all(|&p| p >= 0.0 && p <= 1.0));
}
```

### Integration Tests

Test complete workflow:

```rust
#[test]
fn test_end_to_end() {
    let session = load_onnx_session_default().unwrap();
    let raw = compute_quality_raw("test.jpg", &session).unwrap();
    let scores = QualityScores::compute(&raw, &QualityParams::default());
    assert!(scores.final_score > 0.0);
}
```

## Performance Tips

1. **Share ONNX session** across all images (expensive to create)
2. **Use parallel processing** for batch operations (rayon)
3. **Cache raw data** for parameter tuning workflows
4. **Don't cache scores** - they're cheap to recompute
5. **Render visualizations on-demand** only for visible images

## Migration from Old API

Old code using `blur_weights_from_nchw()`:

```rust
// Old
let (w20, p20, t20, blur) = blur_weights_from_nchw(&tensor, None);
```

New layered API:

```rust
// New
let raw = compute_raw_tenengrad(&tensor)?;
let params = QualityParams::default();
let (p224, p112) = apply_tenengrad_params(&raw.t224, &raw.t112, ...);
let blur_prob = fuse_probabilities(&p224, &p112);
let weights = compute_weights(&blur_prob, &params);
```

The old function still works and now uses the new layered functions internally.
