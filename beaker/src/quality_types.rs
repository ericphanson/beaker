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
