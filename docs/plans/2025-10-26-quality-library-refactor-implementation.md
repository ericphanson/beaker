# Quality Library Refactoring Implementation Plan

**For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.**

## Goal

Refactor quality assessment to separate parameter-independent computation from parameter-dependent scoring, enabling GUI slider tuning without rerunning expensive ONNX inference.

## Architecture

Split quality assessment into three layers: (1) expensive parameter-independent computation (preprocessing, ONNX inference, raw Tenengrad - cache with `#[cached]`), (2) cheap parameter-dependent scoring (blur probability/weight calculation - recompute on slider change), (3) visualization (heatmap rendering - on-demand). Use `cached` crate for automatic memoization of Layer 1.

## Tech Stack

- Rust with ndarray for array operations
- `cached` crate for proc macro memoization
- ONNX Runtime for inference
- Existing blur detection in `beaker/src/blur_detection.rs`
- Existing quality processing in `beaker/src/quality_processing.rs`

## Prerequisites

Before starting implementation:
- Measured timing data from `beaker/src/quality_processing.rs` shows preprocessing (24-31ms) + ONNX (29-31ms) + blur (2ms) = ~60ms total
- Current constants in `beaker/src/blur_detection.rs:6-24` need to become runtime parameters
- Current `blur_weights_from_nchw()` mixes raw computation with parameter application

---

## Task 1: Create QualityParams Structure

**Files:**
- Create: `beaker/src/quality_types.rs`
- Modify: `beaker/src/lib.rs`
- Test: `beaker/tests/quality_params_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/quality_params_test.rs`:

```rust
use beaker::quality_types::QualityParams;

#[test]
fn test_quality_params_default_values() {
    let params = QualityParams::default();

    assert_eq!(params.beta, 1.2);
    assert_eq!(params.tau_ten_224, 0.02);
    assert_eq!(params.p_floor, 0.0);
    assert_eq!(params.alpha, 0.7);
    assert_eq!(params.min_weight, 0.2);
    assert_eq!(params.s_ref, 96.0);
    assert_eq!(params.cov_ref, 4.0);
    assert_eq!(params.core_ratio, 0.60);
}

#[test]
fn test_quality_params_custom_values() {
    let params = QualityParams {
        beta: 1.5,
        alpha: 0.8,
        ..Default::default()
    };

    assert_eq!(params.beta, 1.5);
    assert_eq!(params.alpha, 0.8);
    assert_eq!(params.tau_ten_224, 0.02); // Still default
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_params_test
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `quality_types` in `beaker`
```

### Step 3: Write minimal implementation

Create `beaker/src/quality_types.rs`:

```rust
//! Data structures for quality assessment

/// Tunable parameters for quality heuristics
#[derive(Clone, Debug, PartialEq)]
pub struct QualityParams {
    /// Blur probability mapping exponent (default: 1.2, range: [0.5, 2.0])
    pub beta: f32,

    /// Tenengrad threshold at 224x224 (default: 0.02, range: [0.001, 0.1])
    pub tau_ten_224: f32,

    /// Blur probability floor (default: 0.0, range: [0.0, 0.1])
    pub p_floor: f32,

    /// Weight coefficient (default: 0.7, range: [0.0, 1.0])
    pub alpha: f32,

    /// Minimum blur weight clamp (default: 0.2, range: [0.0, 0.5])
    pub min_weight: f32,

    /// Reference detection size (default: 96.0)
    pub s_ref: f32,

    /// Reference grid coverage (default: 4.0)
    pub cov_ref: f32,

    /// Core vs ring ratio (default: 0.60)
    pub core_ratio: f32,
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
```

Modify `beaker/src/lib.rs` to add module:

```rust
// After existing module declarations
pub mod quality_types;
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_params_test
```

**Expected output:**
```
running 2 tests
test test_quality_params_custom_values ... ok
test test_quality_params_default_values ... ok

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/quality_types.rs beaker/src/lib.rs beaker/tests/quality_params_test.rs
git commit -m "feat: add QualityParams with default values matching current constants"
```

---

## Task 2: Create QualityRawData Structure

**Files:**
- Modify: `beaker/src/quality_types.rs`
- Modify: `beaker/tests/quality_params_test.rs`

### Step 1: Write the failing test

Add to `beaker/tests/quality_params_test.rs`:

```rust
use beaker::quality_types::{QualityParams, QualityRawData};
use std::time::SystemTime;

#[test]
fn test_quality_raw_data_creation() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 75.5,
        paq2piq_local: [[50u8; 20]; 20],
        tenengrad_224: [[0.1f32; 20]; 20],
        tenengrad_112: [[0.05f32; 20]; 20],
        median_tenengrad_224: 0.08,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
        computed_at: SystemTime::now(),
    };

    assert_eq!(raw.input_width, 640);
    assert_eq!(raw.input_height, 480);
    assert_eq!(raw.paq2piq_global, 75.5);
    assert_eq!(raw.model_version, "quality-model-v1");
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_params_test::test_quality_raw_data_creation
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `QualityRawData` in `quality_types`
```

### Step 3: Write minimal implementation

Add to `beaker/src/quality_types.rs`:

```rust
use std::time::SystemTime;

/// Parameter-independent computation results (expensive to compute, ~60ms)
#[derive(Clone, Debug)]
pub struct QualityRawData {
    /// Original image dimensions
    pub input_width: u32,
    pub input_height: u32,

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
    pub computed_at: SystemTime,
}
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_params_test::test_quality_raw_data_creation
```

**Expected output:**
```
test test_quality_raw_data_creation ... ok
```

### Step 5: Commit

```bash
git add beaker/src/quality_types.rs beaker/tests/quality_params_test.rs
git commit -m "feat: add QualityRawData for parameter-independent results"
```

---

## Task 3: Create QualityScores Structure

**Files:**
- Modify: `beaker/src/quality_types.rs`
- Modify: `beaker/tests/quality_params_test.rs`

### Step 1: Write the failing test

Add to `beaker/tests/quality_params_test.rs`:

```rust
use beaker::quality_types::{QualityParams, QualityRawData, QualityScores};

#[test]
fn test_quality_scores_creation() {
    let params = QualityParams::default();
    let scores = QualityScores {
        final_score: 65.0,
        paq2piq_score: 75.0,
        blur_score: 0.3,
        blur_probability: [[0.3f32; 20]; 20],
        blur_weights: [[0.79f32; 20]; 20],
        params: params.clone(),
    };

    assert_eq!(scores.final_score, 65.0);
    assert_eq!(scores.paq2piq_score, 75.0);
    assert_eq!(scores.blur_score, 0.3);
    assert_eq!(scores.params.alpha, 0.7);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_params_test::test_quality_scores_creation
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `QualityScores` in `quality_types`
```

### Step 3: Write minimal implementation

Add to `beaker/src/quality_types.rs`:

```rust
/// Parameter-dependent quality scores (cheap to compute, <0.1ms)
#[derive(Clone, Debug)]
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
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_params_test::test_quality_scores_creation
```

**Expected output:**
```
test test_quality_scores_creation ... ok
```

### Step 5: Commit

```bash
git add beaker/src/quality_types.rs beaker/tests/quality_params_test.rs
git commit -m "feat: add QualityScores for parameter-dependent results"
```

---

## Task 4: Extract Raw Tenengrad Computation

**Files:**
- Modify: `beaker/src/blur_detection.rs`
- Test: `beaker/tests/blur_detection_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/blur_detection_test.rs`:

```rust
use beaker::blur_detection::compute_raw_tenengrad;
use ndarray::Array4;

#[test]
fn test_compute_raw_tenengrad() {
    // Create a simple 224x224 test image in NCHW format
    let img_nchw = Array4::<f32>::from_elem((1, 3, 224, 224), 0.5);

    let result = compute_raw_tenengrad(&img_nchw).unwrap();

    // Verify structure
    assert_eq!(result.t224.shape(), &[20, 20]);
    assert_eq!(result.t112.shape(), &[20, 20]);
    assert!(result.median_224 >= 0.0);
    assert!(result.scale_ratio >= 0.05 && result.scale_ratio <= 0.80);
}

#[test]
fn test_compute_raw_tenengrad_high_contrast() {
    // High contrast image should have higher Tenengrad
    let mut img_nchw = Array4::<f32>::zeros((1, 3, 224, 224));

    // Create checkerboard pattern (high gradient)
    for i in 0..224 {
        for j in 0..224 {
            let val = if (i / 10 + j / 10) % 2 == 0 { 1.0 } else { 0.0 };
            img_nchw[[0, 0, i, j]] = val;
            img_nchw[[0, 1, i, j]] = val;
            img_nchw[[0, 2, i, j]] = val;
        }
    }

    let result = compute_raw_tenengrad(&img_nchw).unwrap();

    // High contrast should produce non-zero Tenengrad
    assert!(result.median_224 > 0.0);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test blur_detection_test
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `compute_raw_tenengrad` in `blur_detection`
```

### Step 3: Write minimal implementation

Add to `beaker/src/blur_detection.rs` (at top, after existing constants):

```rust
use anyhow::Result;

/// Raw Tenengrad computation results (parameter-independent)
#[derive(Clone, Debug)]
pub struct RawTenengradData {
    pub t224: Array2<f32>,      // 20x20 Tenengrad scores at 224x224
    pub t112: Array2<f32>,      // 20x20 Tenengrad scores at 112x112
    pub median_224: f32,        // Median for adaptive thresholding
    pub scale_ratio: f32,       // Scale ratio (112/224)
}

/// Compute raw Tenengrad scores without applying parameters (expensive: ~2ms)
/// This is parameter-independent - compute once, cache forever
pub fn compute_raw_tenengrad(x: &Array4<f32>) -> Result<RawTenengradData> {
    // Convert to grayscale and compute Tenengrad at both scales
    let gray224 = nchw_to_gray_224(x);
    let t224 = tenengrad_mean_grid_20(&gray224);

    let gray112 = downsample_2x_gray_f32(&gray224);
    let t112 = tenengrad_mean_grid_20(&gray112);

    // Compute median and scale ratio (same logic as current code)
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
```

### Step 4: Run test to verify it passes

```bash
cargo test --test blur_detection_test
```

**Expected output:**
```
running 2 tests
test test_compute_raw_tenengrad ... ok
test test_compute_raw_tenengrad_high_contrast ... ok

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/blur_detection.rs beaker/tests/blur_detection_test.rs
git commit -m "feat: extract parameter-independent Tenengrad computation"
```

---

## Task 5: Extract Parameter Application Functions

**Files:**
- Modify: `beaker/src/blur_detection.rs`
- Modify: `beaker/tests/blur_detection_test.rs`

### Step 1: Write the failing test

Add to `beaker/tests/blur_detection_test.rs`:

```rust
use beaker::blur_detection::{apply_tenengrad_params, fuse_probabilities, compute_weights};
use beaker::quality_types::QualityParams;
use ndarray::Array2;

#[test]
fn test_apply_tenengrad_params() {
    let t224 = Array2::<f32>::from_elem((20, 20), 0.05);
    let t112 = Array2::<f32>::from_elem((20, 20), 0.025);
    let median_224 = 0.04;
    let scale_ratio = 0.5;
    let params = QualityParams::default();

    let (p224, p112) = apply_tenengrad_params(&t224, &t112, median_224, scale_ratio, &params);

    assert_eq!(p224.shape(), &[20, 20]);
    assert_eq!(p112.shape(), &[20, 20]);
    // All probabilities should be in [0, 1]
    assert!(p224.iter().all(|&p| p >= 0.0 && p <= 1.0));
    assert!(p112.iter().all(|&p| p >= 0.0 && p <= 1.0));
}

#[test]
fn test_fuse_probabilities() {
    let p224 = Array2::<f32>::from_elem((20, 20), 0.3);
    let p112 = Array2::<f32>::from_elem((20, 20), 0.2);

    let fused = fuse_probabilities(&p224, &p112);

    assert_eq!(fused.shape(), &[20, 20]);
    // Fused probability should be >= max of inputs (probabilistic OR)
    assert!(fused[[0, 0]] >= 0.3 && fused[[0, 0]] >= 0.2);
    assert!(fused[[0, 0]] <= 1.0);
}

#[test]
fn test_compute_weights() {
    let blur_prob = Array2::<f32>::from_elem((20, 20), 0.5);
    let params = QualityParams::default();

    let weights = compute_weights(&blur_prob, &params);

    assert_eq!(weights.shape(), &[20, 20]);
    // weight = (1 - alpha * p).clamp(min_weight, 1.0)
    // With alpha=0.7, p=0.5: weight = 1 - 0.7*0.5 = 0.65
    let expected = (1.0 - params.alpha * 0.5).clamp(params.min_weight, 1.0);
    assert!((weights[[0, 0]] - expected).abs() < 1e-5);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test blur_detection_test -- apply_tenengrad_params fuse_probabilities compute_weights
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `apply_tenengrad_params` in `blur_detection`
error[E0433]: failed to resolve: could not find `fuse_probabilities` in `blur_detection`
error[E0433]: failed to resolve: could not find `compute_weights` in `blur_detection`
```

### Step 3: Write minimal implementation

Add to `beaker/src/blur_detection.rs`:

```rust
use crate::quality_types::QualityParams;

const EPS_T: f32 = 1e-12;
const BIAS112: f32 = 1.25;

/// Apply parameters to raw Tenengrad to get blur probabilities (cheap: <0.1ms)
pub fn apply_tenengrad_params(
    t224: &Array2<f32>,
    t112: &Array2<f32>,
    median_224: f32,
    scale_ratio: f32,
    params: &QualityParams,
) -> (Array2<f32>, Array2<f32>) {
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
```

### Step 4: Run test to verify it passes

```bash
cargo test --test blur_detection_test -- apply_tenengrad_params fuse_probabilities compute_weights
```

**Expected output:**
```
running 3 tests
test test_apply_tenengrad_params ... ok
test test_compute_weights ... ok
test test_fuse_probabilities ... ok

test result: ok. 3 passed
```

### Step 5: Commit

```bash
git add beaker/src/blur_detection.rs beaker/tests/blur_detection_test.rs
git commit -m "feat: extract parameter-dependent blur computation functions"
```

---

## Task 6: Implement QualityScores::compute() Method

**Files:**
- Modify: `beaker/src/quality_types.rs`
- Modify: `beaker/tests/quality_params_test.rs`

### Step 1: Write the failing test

Add to `beaker/tests/quality_params_test.rs`:

```rust
use beaker::quality_types::{QualityParams, QualityRawData, QualityScores};
use std::time::SystemTime;

#[test]
fn test_quality_scores_compute_from_raw() {
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 80.0,
        paq2piq_local: [[60u8; 20]; 20],
        tenengrad_224: [[0.05f32; 20]; 20],
        tenengrad_112: [[0.025f32; 20]; 20],
        median_tenengrad_224: 0.04,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
        computed_at: SystemTime::now(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    // Verify final score is computed
    assert!(scores.final_score > 0.0);
    assert!(scores.final_score <= 100.0);

    // Verify component scores
    assert_eq!(scores.paq2piq_score, 80.0);
    assert!(scores.blur_score >= 0.0 && scores.blur_score <= 1.0);

    // Verify grids have correct shape
    assert_eq!(scores.blur_probability.len(), 20);
    assert_eq!(scores.blur_weights.len(), 20);

    // Verify params are stored
    assert_eq!(scores.params.alpha, 0.7);
}

#[test]
fn test_quality_scores_compute_no_blur() {
    // High Tenengrad = sharp = low blur probability
    let raw = QualityRawData {
        input_width: 640,
        input_height: 480,
        paq2piq_global: 90.0,
        paq2piq_local: [[80u8; 20]; 20],
        tenengrad_224: [[1.0f32; 20]; 20],  // Very high gradient
        tenengrad_112: [[0.5f32; 20]; 20],
        median_tenengrad_224: 0.8,
        scale_ratio: 0.5,
        model_version: "quality-model-v1".to_string(),
        computed_at: SystemTime::now(),
    };

    let params = QualityParams::default();
    let scores = QualityScores::compute(&raw, &params);

    // Low blur probability should mean blur_score is low
    assert!(scores.blur_score < 0.5, "Sharp image should have low blur score");

    // Low blur means high weight, so final_score should be close to paq2piq
    assert!(scores.final_score > 85.0, "Sharp image should preserve quality score");
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_params_test -- compute_from_raw
```

**Expected output:**
```
error[E0599]: no method named `compute` found for struct `QualityScores`
```

### Step 3: Write minimal implementation

Add to `beaker/src/quality_types.rs`:

```rust
use crate::blur_detection::{apply_tenengrad_params, fuse_probabilities, compute_weights};

impl QualityScores {
    /// Compute scores from raw data and parameters (cheap: <0.1ms)
    pub fn compute(raw: &QualityRawData, params: &QualityParams) -> Self {
        // Convert arrays to ndarray for computation
        use ndarray::Array2;
        let t224 = Array2::from_shape_vec((20, 20), raw.tenengrad_224.iter().flatten().copied().collect()).unwrap();
        let t112 = Array2::from_shape_vec((20, 20), raw.tenengrad_112.iter().flatten().copied().collect()).unwrap();

        // Apply parameters to raw Tenengrad to get probabilities
        let (p224, p112) = apply_tenengrad_params(
            &t224,
            &t112,
            raw.median_tenengrad_224,
            raw.scale_ratio,
            params,
        );

        // Fuse probabilities
        let blur_probability_array = fuse_probabilities(&p224, &p112);

        // Compute weights
        let blur_weights_array = compute_weights(&blur_probability_array, params);

        // Convert back to fixed arrays
        let mut blur_probability = [[0.0f32; 20]; 20];
        let mut blur_weights = [[0.0f32; 20]; 20];
        for i in 0..20 {
            for j in 0..20 {
                blur_probability[i][j] = blur_probability_array[[i, j]];
                blur_weights[i][j] = blur_weights_array[[i, j]];
            }
        }

        // Global blur score (mean probability)
        let blur_score: f32 = blur_probability.iter()
            .flat_map(|row| row.iter())
            .sum::<f32>() / 400.0;

        // Final combined score
        let w_mean = (1.0 - params.alpha * blur_score).clamp(params.min_weight, 1.0);
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
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_params_test -- compute_from_raw
```

**Expected output:**
```
running 2 tests
test test_quality_scores_compute_from_raw ... ok
test test_quality_scores_compute_no_blur ... ok

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/quality_types.rs beaker/tests/quality_params_test.rs
git commit -m "feat: implement QualityScores::compute() to calculate scores from raw data"
```

---

## Task 7: Add Cached Dependency

**Files:**
- Modify: `beaker/Cargo.toml`

### Step 1: Add dependency

Modify `beaker/Cargo.toml` in the `[dependencies]` section:

```toml
cached = { version = "0.53", features = ["proc_macro"] }
```

### Step 2: Build to verify dependency resolves

```bash
cd beaker && cargo build
```

**Expected output:**
```
   Compiling cached v0.53.x
   ...
   Compiling beaker v0.x.x
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Step 3: Commit

```bash
git add beaker/Cargo.toml
git commit -m "deps: add cached crate for proc macro memoization"
```

---

## Task 8: Create Cached compute_quality_raw() Function

**Files:**
- Modify: `beaker/src/quality_processing.rs`
- Test: `beaker/tests/quality_integration_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/quality_integration_test.rs`:

```rust
use beaker::quality_processing::compute_quality_raw;
use beaker::quality_types::{QualityParams, QualityScores};
use std::path::Path;

#[test]
fn test_compute_quality_raw_example_image() {
    let session = beaker::quality_processing::load_onnx_session_default().unwrap();

    // Use test image if exists
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let raw = compute_quality_raw(test_image, &session).unwrap();

    // Verify structure
    assert!(raw.input_width > 0);
    assert!(raw.input_height > 0);
    assert!(raw.paq2piq_global >= 0.0 && raw.paq2piq_global <= 100.0);
    assert_eq!(raw.model_version, "quality-model-v1");
}

#[test]
fn test_end_to_end_with_custom_params() {
    let session = beaker::quality_processing::load_onnx_session_default().unwrap();

    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    // Compute raw data (expensive, cached)
    let raw = compute_quality_raw(test_image, &session).unwrap();

    // Compute scores with default params
    let params_default = QualityParams::default();
    let scores_default = QualityScores::compute(&raw, &params_default);

    // Compute scores with stricter blur detection (lower tau)
    let params_strict = QualityParams {
        tau_ten_224: 0.01,  // Lower threshold = more sensitive to blur
        ..Default::default()
    };
    let scores_strict = QualityScores::compute(&raw, &params_strict);

    // Both should have valid scores
    assert!(scores_default.final_score > 0.0);
    assert!(scores_strict.final_score > 0.0);

    // Stricter params should detect more blur (higher blur_score)
    // Note: This may not always be true depending on image, but generally holds
    println!("Default blur: {}, Strict blur: {}",
             scores_default.blur_score, scores_strict.blur_score);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_integration_test
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `compute_quality_raw` in `quality_processing`
```

### Step 3: Write minimal implementation

Add to `beaker/src/quality_processing.rs`:

```rust
use cached::proc_macro::cached;
use crate::quality_types::QualityRawData;
use crate::blur_detection::compute_raw_tenengrad;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use anyhow::{Context, Result};

/// Load ONNX session with default model path
pub fn load_onnx_session_default() -> Result<ort::Session> {
    let model_dir = std::env::var("ONNX_MODEL_CACHE_DIR")
        .unwrap_or_else(|_| "models".to_string());
    let model_path = Path::new(&model_dir).join("quality_model.onnx");

    ort::Session::builder()?
        .with_model_from_file(&model_path)
        .context("Failed to load ONNX model")
}

/// Compute parameter-independent quality data (expensive: ~60ms, cached)
#[cached(
    size = 1000,
    key = "String",
    convert = r#"{ format!("{}", path.as_ref().display()) }"#
)]
pub fn compute_quality_raw(
    path: impl AsRef<Path>,
    session: &ort::Session,
) -> Result<QualityRawData> {
    // Load and preprocess image
    let img = image::open(path.as_ref())
        .context("Failed to open image")?;

    let input_width = img.width();
    let input_height = img.height();

    // Preprocess for ONNX
    let input_array = preprocess_image_for_quality(&img)?;

    // Run ONNX inference
    let input_name = session.inputs[0].name.clone();
    let input_value = ort::Value::from_array(input_array.view())?;
    let outputs = session.run(ort::inputs![input_name.as_str() => &input_value])?;

    // Extract outputs
    let output = outputs[0].try_extract_tensor::<f32>()?;
    let output_view = output.view();

    // Parse ONNX outputs (same as current code)
    let global_idx = 400;
    let paq2piq_global = output_view[global_idx].clamp(0.0, 100.0);

    let mut paq2piq_local = [[0u8; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            let idx = i * 20 + j;
            let val = output_view[idx].clamp(0.0, 100.0);
            paq2piq_local[i][j] = val as u8;
        }
    }

    // Compute raw Tenengrad
    let raw_tenengrad = compute_raw_tenengrad(&input_array)?;

    // Convert ndarray to fixed array
    let mut tenengrad_224 = [[0.0f32; 20]; 20];
    let mut tenengrad_112 = [[0.0f32; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            tenengrad_224[i][j] = raw_tenengrad.t224[[i, j]];
            tenengrad_112[i][j] = raw_tenengrad.t112[[i, j]];
        }
    }

    Ok(QualityRawData {
        input_width,
        input_height,
        paq2piq_global,
        paq2piq_local,
        tenengrad_224,
        tenengrad_112,
        median_tenengrad_224: raw_tenengrad.median_224,
        scale_ratio: raw_tenengrad.scale_ratio,
        model_version: "quality-model-v1".to_string(),
        computed_at: SystemTime::now(),
    })
}
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_integration_test
```

**Expected output:**
```
running 2 tests
test test_compute_quality_raw_example_image ... ok (or skipped)
test test_end_to_end_with_custom_params ... ok (or skipped)

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/quality_processing.rs beaker/tests/quality_integration_test.rs
git commit -m "feat: add cached compute_quality_raw() function"
```

---

## Remaining Tasks Summary

The following tasks continue the implementation. Each follows the same TDD pattern:

### Task 9: Update CLI to Accept Parameter Flags
- Add `--alpha`, `--beta`, etc. to `beaker/src/config.rs`
- Parse into `QualityParams`
- Update `quality_command()` to use new API

### Task 10: Refactor blur_weights_from_nchw() to Use New Functions
- Make existing function call new layered functions
- Maintain backward compatibility
- Ensure existing tests still pass

### Task 11: Add Visualization Separation
- Create `HeatmapStyle` type
- Extract `render_heatmap_to_buffer()` function
- Keep visualization separate from computation

### Task 12: Integration and Documentation
- Update main README with new parameter flags
- Add examples of using layered API
- Performance documentation

## Testing Strategy

**Unit Tests:**
- Each data structure has creation tests
- Each function has behavior tests
- Edge cases (empty, zero, extreme values)

**Integration Tests:**
- End-to-end with real images
- Cache hit verification
- Parameter variation tests

**Regression Tests:**
- Compare outputs with current implementation
- Ensure default params match current constants
- Performance benchmarks (should be same or better)

## Common Issues and Solutions

**Issue: ONNX model not found**
```bash
export ONNX_MODEL_CACHE_DIR=/path/to/models
```

**Issue: Test images not found**
Tests skip gracefully if example images missing

**Issue: Cache not clearing between test runs**
```bash
cargo clean
```

## Performance Expectations

- `compute_quality_raw()`: ~60ms first call, <1ms cache hit
- `QualityScores::compute()`: <0.1ms (always recomputed)
- Total for GUI slider change (100 images): ~10ms

## Reference

- Current constants: `beaker/src/blur_detection.rs:6-24`
- Current implementation: `beaker/src/blur_detection.rs:398-492`
- Timing measurements: `beaker/src/quality_processing.rs` (debug logs)
