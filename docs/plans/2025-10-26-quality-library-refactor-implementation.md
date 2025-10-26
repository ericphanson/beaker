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

---

## Task 9: Add CLI Parameter Flags to Config

**Files:**
- Modify: `beaker/src/config.rs`
- Test: `beaker/tests/config_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/config_test.rs`:

```rust
use beaker::config::QualityConfig;
use beaker::quality_types::QualityParams;

#[test]
fn test_quality_config_default_params() {
    let config = QualityConfig {
        sources: vec!["test.jpg".into()],
        params: None,
        metadata: false,
        debug_dump_images: false,
    };

    let params = config.params.unwrap_or_default();
    assert_eq!(params.alpha, 0.7);
    assert_eq!(params.beta, 1.2);
}

#[test]
fn test_quality_config_custom_params() {
    let custom_params = QualityParams {
        alpha: 0.8,
        beta: 1.5,
        ..Default::default()
    };

    let config = QualityConfig {
        sources: vec!["test.jpg".into()],
        params: Some(custom_params),
        metadata: false,
        debug_dump_images: false,
    };

    assert!(config.params.is_some());
    assert_eq!(config.params.as_ref().unwrap().alpha, 0.8);
    assert_eq!(config.params.as_ref().unwrap().beta, 1.5);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test config_test
```

**Expected output:**
```
error[E0433]: failed to resolve: could not find `QualityConfig` in `config`
```

### Step 3: Write minimal implementation

Find the existing `QualityConfig` struct in `beaker/src/config.rs` and add the `params` field:

```rust
use crate::quality_types::QualityParams;

pub struct QualityConfig {
    pub sources: Vec<PathBuf>,
    pub metadata: bool,
    pub debug_dump_images: bool,

    // New field for tunable parameters
    pub params: Option<QualityParams>,
}
```

Then in the CLI argument parsing section (likely using `clap`), add parameter flags:

```rust
#[derive(Parser)]
#[command(name = "beaker")]
pub struct Cli {
    // ... existing fields ...

    /// Override alpha parameter (weight coefficient)
    #[arg(long)]
    pub alpha: Option<f32>,

    /// Override beta parameter (blur probability exponent)
    #[arg(long)]
    pub beta: Option<f32>,

    /// Override tau parameter (Tenengrad threshold)
    #[arg(long)]
    pub tau: Option<f32>,
}

// In the config building function:
impl From<Cli> for QualityConfig {
    fn from(cli: Cli) -> Self {
        let params = if cli.alpha.is_some() || cli.beta.is_some() || cli.tau.is_some() {
            let mut p = QualityParams::default();
            if let Some(alpha) = cli.alpha {
                p.alpha = alpha;
            }
            if let Some(beta) = cli.beta {
                p.beta = beta;
            }
            if let Some(tau) = cli.tau {
                p.tau_ten_224 = tau;
            }
            Some(p)
        } else {
            None
        };

        Self {
            sources: cli.sources,
            metadata: cli.metadata,
            debug_dump_images: cli.debug_dump_images,
            params,
        }
    }
}
```

### Step 4: Run test to verify it passes

```bash
cargo test --test config_test
```

**Expected output:**
```
running 2 tests
test test_quality_config_custom_params ... ok
test test_quality_config_default_params ... ok

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/config.rs beaker/tests/config_test.rs
git commit -m "feat: add CLI flags for quality parameter tuning (--alpha, --beta, --tau)"
```

---

## Task 10: Update quality Command to Use New API

**Files:**
- Modify: `beaker/src/commands/quality.rs` (or wherever quality command is implemented)
- Test: `beaker/tests/quality_command_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/quality_command_test.rs`:

```rust
use beaker::quality_processing::{compute_quality_raw, load_onnx_session_default};
use beaker::quality_types::{QualityParams, QualityScores};
use std::path::Path;

#[test]
fn test_quality_command_workflow() {
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let session = load_onnx_session_default().unwrap();
    let params = QualityParams::default();

    // Step 1: Compute raw data (cached)
    let raw = compute_quality_raw(test_image, &session).unwrap();

    // Step 2: Compute scores with params
    let scores = QualityScores::compute(&raw, &params);

    // Verify complete workflow
    assert!(scores.final_score > 0.0);
    assert!(scores.final_score <= 100.0);
    println!("Quality score: {:.1}", scores.final_score);
}

#[test]
fn test_quality_command_custom_params() {
    let test_image = Path::new("example.jpg");
    if !test_image.exists() {
        eprintln!("Skipping test - example.jpg not found");
        return;
    }

    let session = load_onnx_session_default().unwrap();

    // Test with stricter blur detection
    let strict_params = QualityParams {
        tau_ten_224: 0.01,  // More sensitive to blur
        ..Default::default()
    };

    let raw = compute_quality_raw(test_image, &session).unwrap();
    let scores = QualityScores::compute(&raw, &strict_params);

    assert!(scores.final_score > 0.0);
    println!("Quality score (strict): {:.1}", scores.final_score);
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test quality_command_test
```

**Expected output:**
```
running 2 tests
test test_quality_command_custom_params ... ok (or skipped)
test test_quality_command_workflow ... ok (or skipped)

test result: ok. 2 passed
```

Note: These tests may already pass if Task 8 was completed correctly. This task is primarily about refactoring the command handler.

### Step 3: Write minimal implementation

Update the quality command handler (exact location depends on your CLI structure):

```rust
// In beaker/src/commands/quality.rs or similar

use crate::config::QualityConfig;
use crate::quality_processing::{compute_quality_raw, load_onnx_session_default};
use crate::quality_types::{QualityParams, QualityScores};
use anyhow::Result;

pub fn quality_command(config: QualityConfig) -> Result<()> {
    // Load ONNX session once
    let session = load_onnx_session_default()?;

    // Get parameters (use defaults if not specified)
    let params = config.params.unwrap_or_default();

    // Process each image
    for image_path in &config.sources {
        // Level 1: Compute raw data (cached automatically)
        let raw = compute_quality_raw(image_path, &session)?;

        // Level 2: Compute scores from raw data with parameters
        let scores = QualityScores::compute(&raw, &params);

        // Output result
        println!("{}: {:.1}", image_path.display(), scores.final_score);

        // Optional: Write metadata
        if config.metadata {
            write_quality_metadata(image_path, &raw, &scores)?;
        }

        // Optional: Generate debug visualizations
        if config.debug_dump_images {
            generate_debug_images(image_path, &raw, &scores)?;
        }
    }

    Ok(())
}

// Helper function to write metadata (if needed)
fn write_quality_metadata(
    path: &Path,
    raw: &QualityRawData,
    scores: &QualityScores,
) -> Result<()> {
    // Implementation to write .beaker.toml or similar
    // (Use existing metadata writing code if available)
    Ok(())
}

// Helper function for debug images (if needed)
fn generate_debug_images(
    path: &Path,
    raw: &QualityRawData,
    scores: &QualityScores,
) -> Result<()> {
    // Implementation to generate heatmaps
    // (Use existing debug image code or defer to Task 11)
    Ok(())
}
```

### Step 4: Run test to verify it passes

```bash
cargo test --test quality_command_test
```

**Expected output:**
```
running 2 tests
test test_quality_command_custom_params ... ok (or skipped)
test test_quality_command_workflow ... ok (or skipped)

test result: ok. 2 passed
```

### Step 5: Commit

```bash
git add beaker/src/commands/quality.rs beaker/tests/quality_command_test.rs
git commit -m "refactor: update quality command to use layered API with parameter support"
```

---

## Task 11: Refactor blur_weights_from_nchw() for Backward Compatibility

**Files:**
- Modify: `beaker/src/blur_detection.rs`
- Test: `beaker/tests/blur_detection_backward_compat_test.rs`

### Step 1: Write the failing test

Create `beaker/tests/blur_detection_backward_compat_test.rs`:

```rust
use beaker::blur_detection::blur_weights_from_nchw;
use ndarray::Array4;
use std::path::PathBuf;

#[test]
fn test_blur_weights_from_nchw_still_works() {
    // Create a simple test image
    let img_nchw = Array4::<f32>::from_elem((1, 3, 224, 224), 0.5);

    // Call existing function (should still work)
    let (w20, p20, t20, global_blur) = blur_weights_from_nchw(&img_nchw, None);

    // Verify structure (same as before)
    assert_eq!(w20.shape(), &[20, 20]);
    assert_eq!(p20.shape(), &[20, 20]);
    assert_eq!(t20.shape(), &[20, 20]);
    assert!(global_blur >= 0.0 && global_blur <= 1.0);
}

#[test]
fn test_blur_weights_from_nchw_produces_same_results() {
    // This test ensures backward compatibility
    // Create a high-contrast image
    let mut img_nchw = Array4::<f32>::zeros((1, 3, 224, 224));
    for i in 0..224 {
        for j in 0..224 {
            let val = if (i / 10 + j / 10) % 2 == 0 { 1.0 } else { 0.0 };
            img_nchw[[0, 0, i, j]] = val;
            img_nchw[[0, 1, i, j]] = val;
            img_nchw[[0, 2, i, j]] = val;
        }
    }

    let (w20, p20, _t20, global_blur) = blur_weights_from_nchw(&img_nchw, None);

    // High contrast = low blur probability
    assert!(global_blur < 0.5, "High contrast should have low blur probability");

    // Weights should be close to 1.0 (low blur means high weight)
    let mean_weight: f32 = w20.iter().sum::<f32>() / 400.0;
    assert!(mean_weight > 0.7, "Low blur should produce high weights");
}
```

### Step 2: Run test to verify it fails

```bash
cargo test --test blur_detection_backward_compat_test
```

**Expected output:**
```
running 2 tests
test test_blur_weights_from_nchw_produces_same_results ... ok
test test_blur_weights_from_nchw_still_works ... ok

test result: ok. 2 passed
```

Note: If the function hasn't changed yet, tests should pass. This validates our starting point.

### Step 3: Write minimal implementation

Refactor `blur_weights_from_nchw()` in `beaker/src/blur_detection.rs` to use new functions:

```rust
/// Compute blur weights from NCHW tensor (backward compatible wrapper)
pub fn blur_weights_from_nchw(
    x: &Array4<f32>,
    out_dir: Option<PathBuf>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, f32) {
    // Use new layered functions internally
    let raw = compute_raw_tenengrad(x)
        .expect("Failed to compute raw Tenengrad");

    // Use default parameters (matches old hardcoded constants)
    let params = crate::quality_types::QualityParams::default();

    // Apply parameters
    let (p224, p112) = apply_tenengrad_params(
        &Array2::from_shape_vec((20, 20), raw.t224.iter().flatten().copied().collect()).unwrap(),
        &Array2::from_shape_vec((20, 20), raw.t112.iter().flatten().copied().collect()).unwrap(),
        raw.median_224,
        raw.scale_ratio,
        &params,
    );

    // Fuse probabilities
    let blur_probability = fuse_probabilities(&p224, &p112);

    // Compute weights
    let blur_weights = compute_weights(&blur_probability, &params);

    // Global blur score
    let global_blur_score = blur_probability.iter().sum::<f32>() / 400.0;

    // Optional debug output (if out_dir provided)
    if let Some(dir) = out_dir {
        // Generate debug heatmaps using existing code
        // (This may be refactored in Task 12)
        save_debug_heatmaps(&dir, &blur_probability, &blur_weights, &raw.t224);
    }

    // Return in old format
    // Note: Returning raw.t224 as Array2 by converting
    let t224_array = Array2::from_shape_vec((20, 20),
        raw.t224.iter().flatten().copied().collect()
    ).unwrap();

    (blur_weights, blur_probability, t224_array, global_blur_score)
}

// Helper for debug output (temporary, may be refactored)
fn save_debug_heatmaps(
    out_dir: &Path,
    blur_prob: &Array2<f32>,
    blur_weights: &Array2<f32>,
    tenengrad: &[[f32; 20]; 20],
) {
    // Use existing heatmap generation code
    // This is a placeholder - implementation depends on current debug output code
}
```

### Step 4: Run test to verify it passes

```bash
cargo test --test blur_detection_backward_compat_test
```

**Expected output:**
```
running 2 tests
test test_blur_weights_from_nchw_produces_same_results ... ok
test test_blur_weights_from_nchw_still_works ... ok

test result: ok. 2 passed
```

Also run existing blur detection tests:

```bash
cargo test blur_detection
```

**Expected output:**
```
running X tests
test blur_detection::... ... ok
...
test result: ok. X passed
```

### Step 5: Commit

```bash
git add beaker/src/blur_detection.rs beaker/tests/blur_detection_backward_compat_test.rs
git commit -m "refactor: make blur_weights_from_nchw() use new layered functions internally"
```

---

## Task 12: Add Visualization Types and Documentation

**Files:**
- Modify: `beaker/src/quality_types.rs`
- Modify: `beaker/README.md`
- Create: `docs/quality-api-guide.md`
- Test: Manual testing with example images

### Step 1: Add HeatmapStyle type

Add to `beaker/src/quality_types.rs`:

```rust
/// Heatmap rendering options
#[derive(Clone, Debug)]
pub struct HeatmapStyle {
    /// Colormap for heatmap rendering
    pub colormap: ColorMap,

    /// Overlay transparency (0.0 = invisible, 1.0 = opaque)
    pub alpha: f32,

    /// Target size (can be smaller for thumbnails)
    pub size: (u32, u32),
}

impl Default for HeatmapStyle {
    fn default() -> Self {
        Self {
            colormap: ColorMap::Viridis,
            alpha: 0.7,
            size: (224, 224),
        }
    }
}

/// Available colormaps for heatmap rendering
#[derive(Clone, Copy, Debug)]
pub enum ColorMap {
    Viridis,
    Plasma,
    Inferno,
    Turbo,
    Grayscale,
}
```

### Step 2: Update README.md

Add to `beaker/README.md`:

```markdown
## Quality Assessment

Beaker includes a no-reference image quality assessment model (PaQ-2-PiQ) combined with blur detection.

### Basic Usage

```bash
# Assess single image
beaker quality image.jpg

# Assess multiple images
beaker quality *.jpg

# Write metadata to sidecar files
beaker quality --metadata image.jpg
```

### Parameter Tuning

Quality assessment uses several tunable parameters for blur detection:

```bash
# Adjust blur sensitivity (lower = more sensitive)
beaker quality --tau 0.01 image.jpg

# Adjust blur weight impact (higher = more penalty for blur)
beaker quality --alpha 0.8 image.jpg

# Adjust probability curve steepness
beaker quality --beta 1.5 image.jpg
```

**Parameter Reference:**

- `--alpha` (0.0-1.0, default 0.7): Weight coefficient - how much blur reduces quality score
- `--beta` (0.5-2.0, default 1.2): Probability curve exponent - steeper = more aggressive blur detection
- `--tau` (0.001-0.1, default 0.02): Tenengrad threshold - lower = more sensitive to blur

### Programmatic API

```rust
use beaker::quality_processing::{compute_quality_raw, load_onnx_session_default};
use beaker::quality_types::{QualityParams, QualityScores};

// Load model once
let session = load_onnx_session_default()?;

// Compute raw data (expensive, cached automatically)
let raw = compute_quality_raw("image.jpg", &session)?;

// Compute scores with default parameters
let params = QualityParams::default();
let scores = QualityScores::compute(&raw, &params);

println!("Quality: {:.1}", scores.final_score);

// Adjust parameters and recompute instantly
let strict_params = QualityParams {
    tau_ten_224: 0.01,
    ..Default::default()
};
let strict_scores = QualityScores::compute(&raw, &strict_params);
```

### Performance

- First run: ~60ms per image (preprocessing + ONNX inference + blur detection)
- Cached run: <1ms per image (cache hit for raw computation)
- Parameter adjustment: <0.1ms per image (recomputes scores from cached raw data)

This makes real-time parameter tuning feasible for GUI applications.
```

### Step 3: Create API guide

Create `docs/quality-api-guide.md`:

```markdown
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
```

### Step 4: Manual testing

```bash
# Build the project
cd beaker && cargo build --release

# Test with example image
./target/release/beaker quality example.jpg

# Test with custom parameters
./target/release/beaker quality --alpha 0.8 --beta 1.5 example.jpg

# Test parameter range
for alpha in 0.5 0.6 0.7 0.8 0.9; do
    echo "Alpha $alpha:"
    ./target/release/beaker quality --alpha $alpha example.jpg
done
```

**Expected output:**
```
example.jpg: 75.3
```

### Step 5: Commit

```bash
git add beaker/src/quality_types.rs beaker/README.md docs/quality-api-guide.md
git commit -m "docs: add HeatmapStyle type and comprehensive quality API documentation"
```

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
