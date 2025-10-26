//! Data structures for quality assessment

use std::time::SystemTime;

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
