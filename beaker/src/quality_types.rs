//! Data structures for quality assessment

use image::RgbaImage;
use serde::Serialize;

/// Tunable parameters for quality heuristics
#[derive(Clone, Debug, PartialEq, Serialize)]
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

/// Tunable parameters for triage decision heuristics
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TriageParams {
    /// Base threshold for core/ring sharpness ratio (default: 1.19)
    /// Values below this suggest overall softness/defocus
    pub core_ring_sharpness_ratio_bad: f32,

    /// Base threshold for grid cells covered (default: 6.15)
    /// Values below this with good sharpness suggest small coverage
    pub grid_cells_covered_bad: f32,

    /// Delta to tighten BAD region for core/ring sharpness ratio (default: 0.4)
    /// BAD if ratio <= (base - delta). Shrinks bad region to reduce false negatives
    pub delta_bad_core_ring_sharpness_ratio: f32,

    /// Delta to tighten BAD region for grid cells covered (default: 1.5)
    /// BAD if cells <= (base - delta). Shrinks bad region to reduce false negatives
    pub delta_bad_grid_cells_covered: f32,

    /// Delta to loosen GOOD region for core/ring sharpness ratio (default: 0.0)
    /// GOOD if ratio > (base + delta). Expands good region to admit more goods
    pub delta_good_core_ring_sharpness_ratio: f32,

    /// Delta to loosen GOOD region for grid cells covered (default: 0.0)
    /// GOOD if cells > (base + delta). Expands good region to admit more goods
    pub delta_good_grid_cells_covered: f32,
}

impl Default for TriageParams {
    fn default() -> Self {
        Self {
            core_ring_sharpness_ratio_bad: 1.19,
            grid_cells_covered_bad: 6.15,
            delta_bad_core_ring_sharpness_ratio: 0.4,
            delta_bad_grid_cells_covered: 1.5,
            delta_good_core_ring_sharpness_ratio: 0.0,
            delta_good_grid_cells_covered: 0.0,
        }
    }
}

/// Quality maps for per-detection quality assessment
pub struct QualityMaps<'a> {
    /// PaQ-2-PiQ 20x20 local quality map (0..100)
    pub q20: &'a ndarray::Array2<f32>,
    /// Blur weights 20x20 (1 - ALPHA * P)
    pub w20: &'a ndarray::Array2<f32>,
    /// Fused blur probability 20x20 (0..1)
    pub p20: &'a ndarray::Array2<f32>,
}

/// Parameter-independent computation results (expensive to compute, ~60ms)
#[derive(Clone, Debug)]
pub struct QualityRawData {
    /// Original image dimensions
    pub input_width: u32,
    pub input_height: u32,

    /// ONNX model outputs (parameter-independent)
    pub paq2piq_global: f32, // Global quality score (0-100)
    pub paq2piq_local: [[u8; 20]; 20], // 20x20 local quality grid

    /// Raw blur detection (parameter-independent)
    pub tenengrad_224: [[f32; 20]; 20], // Raw Tenengrad at 224x224
    pub tenengrad_112: [[f32; 20]; 20], // Raw Tenengrad at 112x112
    pub median_tenengrad_224: f32,      // Median for adaptive thresholding
    pub scale_ratio: f32,               // 112/224 scale ratio

    /// Provenance
    pub model_version: String,
}

/// Parameter-dependent quality scores (cheap to compute, <0.1ms)
#[derive(Clone, Debug)]
pub struct QualityScores {
    /// Final quality score (combining paq2piq and blur)
    pub final_score: f32,

    /// Component scores (for analysis)
    pub paq2piq_score: f32,
    pub blur_score: f32, // Global blur probability (0-1)

    /// Intermediate results (20x20 grids)
    pub blur_probability: [[f32; 20]; 20], // Fused blur probability
    pub blur_weights: [[f32; 20]; 20], // Weights (1 - alpha*P)
}

use crate::blur_detection::{apply_tenengrad_params, compute_weights, fuse_probabilities};

impl QualityScores {
    /// Compute scores from raw data and parameters (cheap: <0.1ms)
    pub fn compute(raw: &QualityRawData, params: &QualityParams) -> Self {
        // Convert arrays to ndarray for computation
        use ndarray::Array2;
        let t224 = Array2::from_shape_vec(
            (20, 20),
            raw.tenengrad_224.iter().flatten().copied().collect(),
        )
        .unwrap();
        let t112 = Array2::from_shape_vec(
            (20, 20),
            raw.tenengrad_112.iter().flatten().copied().collect(),
        )
        .unwrap();

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
        let blur_score: f32 = blur_probability
            .iter()
            .flat_map(|row| row.iter())
            .sum::<f32>()
            / 400.0;

        // Final combined score
        let w_mean = (1.0 - params.alpha * blur_score).clamp(params.min_weight, 1.0);
        let final_score = raw.paq2piq_global * w_mean;

        Self {
            final_score,
            paq2piq_score: raw.paq2piq_global,
            blur_score,
            blur_probability,
            blur_weights,
        }
    }
}

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

/// Visualization layer (parameter-dependent, rendered on-demand)
#[derive(Clone)]
#[allow(dead_code)]
pub struct QualityVisualization {
    /// Rendered heatmap images (in-memory buffers)
    pub blur_probability_heatmap: Option<RgbaImage>,
    pub blur_weights_heatmap: Option<RgbaImage>,
    pub tenengrad_heatmap: Option<RgbaImage>,

    /// Overlay on original image
    pub blur_overlay: Option<RgbaImage>,
}
