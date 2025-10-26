# Implementation Plan Gaps Analysis

**Date:** 2025-10-26
**Comparing:**
- `2025-10-26-library-architecture.md` (Architecture design)
- `2025-10-26-quality-library-refactor-implementation.md` (TDD implementation plan)

## Summary

The implementation plan covers the core data structures and layered computation well (Tasks 1-11), but is **missing the entire visualization layer implementation** (Task 12 only adds types and documentation, not the actual rendering code).

---

## Missing Pieces

### 1. Serialization Support (Minor)

**Architecture shows:**
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityRawData { ... }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityParams { ... }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityScores { ... }
```

**Implementation plan has:**
```rust
#[derive(Clone, Debug)]
pub struct QualityRawData { ... }

#[derive(Clone, Debug, PartialEq)]
pub struct QualityParams { ... }

#[derive(Clone, Debug)]
pub struct QualityScores { ... }
```

**Gap:** Missing `Serialize`, `Deserialize` derives on all three main types.

**Impact:** Low priority - only needed if implementing disk caching or serialization to JSON/TOML. Can add later with minimal changes.

**Fix:** Add to Tasks 1, 2, 3:
```rust
use serde::{Serialize, Deserialize};
```

---

### 2. preprocessed_tensor Field (Minor)

**Architecture shows:**
```rust
pub struct QualityRawData {
    // ...

    /// Preprocessed image tensor (224x224x3, normalized)
    /// Store this to avoid re-preprocessing
    #[serde(skip)]
    pub preprocessed_tensor: Array4<f32>,

    // ...
}
```

**Implementation plan:** Missing this field entirely.

**Gap:** The `preprocessed_tensor` field in QualityRawData for potential re-use.

**Impact:** Low priority - mainly useful if you want to avoid re-preprocessing when rendering visualizations. Current plan recomputes on demand which is acceptable.

**Decision:** Can omit for initial implementation, add later if profiling shows benefit.

---

### 3. QualityRawData Helper Methods (Minor)

**Architecture shows:**
```rust
impl QualityRawData {
    pub fn compute_from_image(
        img: &DynamicImage,
        session: &Session,
        model_version: String,
    ) -> Result<Self> { ... }

    pub fn to_cache_bytes(&self) -> Result<Vec<u8>> { ... }

    pub fn from_cache_bytes(bytes: &[u8]) -> Result<Self> { ... }
}
```

**Implementation plan:**
- Task 8 implements `compute_quality_raw()` as a standalone function, not as a method
- No cache serialization methods

**Gap:** Methods approach vs functions approach. Both work, but architecture favors methods for organization.

**Impact:** Low - both approaches are valid. Standalone functions may be simpler for #[cached] annotation.

**Decision:** Keep standalone functions for simplicity. If we later want methods, can add them as wrappers.

---

### 4. QualityVisualization Structure (MAJOR - Missing entire task)

**Architecture shows:**
```rust
pub struct QualityVisualization {
    pub blur_probability_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    pub blur_weights_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    pub tenengrad_heatmap: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    pub blur_overlay: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl QualityVisualization {
    pub fn render(
        raw: &QualityRawData,
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<Self> { ... }

    pub fn render_blur_only(
        scores: &QualityScores,
        style: &HeatmapStyle,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> { ... }
}
```

**Implementation plan:**
- Task 12 defines `HeatmapStyle` and `ColorMap` types
- Task 12 does NOT implement `QualityVisualization` struct
- Task 12 does NOT implement any rendering methods

**Gap:** Entire visualization layer is missing from implementation tasks!

**Impact:** HIGH - visualization is needed for:
- `--debug-dump-images` CLI flag
- GUI heatmap display
- Overlay rendering

**Fix Required:** Need new Task 13 (see below).

---

### 5. Visualization Helper Functions (MAJOR - Missing)

**Architecture shows these functions:**

```rust
/// Render heatmap to in-memory buffer (moderate: ~3-4ms)
pub fn render_heatmap_to_buffer(
    data: &Array2<f32>,
    style: &HeatmapStyle,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    // Upscale 20x20 to target size with bilinear interpolation
    // Apply colormap
    // ...
}

/// Render overlay (composite heatmap over original image)
pub fn render_overlay(
    original: &DynamicImage,
    heatmap_data: &Array2<f32>,
    style: &HeatmapStyle,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    // Resize heatmap to match original
    // Composite with alpha blending
    // ...
}

// Helper functions needed:
fn bilinear_sample(data: &Array2<f32>, u: f32, v: f32) -> f32 { ... }
fn apply_colormap(value: f32, colormap: ColorMap) -> Rgba<u8> { ... }
fn composite_with_alpha(...) -> Result<...> { ... }
```

**Implementation plan:** None of these are implemented.

**Gap:** All visualization rendering functions are missing.

**Impact:** HIGH - these are essential for:
- Debug heatmap output
- GUI visualization
- Overlay rendering

**Fix Required:** Need implementation tasks for these (see Task 13 below).

---

### 6. Task 11 Implementation Gap (Minor)

**Architecture shows:** `blur_weights_from_nchw()` should call new layered functions.

**Implementation plan Task 11:** Correctly refactors the function but has a placeholder:
```rust
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

**Gap:** `save_debug_heatmaps()` implementation is incomplete.

**Impact:** Medium - debug output won't work until visualization layer is implemented.

**Fix:** Will be resolved by Task 13 (visualization implementation).

---

## Recommended New Task: Task 13

Add a complete Task 13 to implement the visualization layer:

### Task 13: Implement Visualization Layer

**Files:**
- Create: `beaker/src/quality_visualization.rs`
- Modify: `beaker/src/lib.rs`
- Modify: `beaker/src/quality_types.rs`
- Test: `beaker/tests/quality_visualization_test.rs`

**Subtasks:**

#### 13.1: Create QualityVisualization struct
- Define struct with Optional heatmap fields
- Implement QualityVisualization::render()
- Implement QualityVisualization::render_blur_only()

#### 13.2: Implement bilinear_sample()
- Bilinear interpolation for upscaling 20x20 to arbitrary size
- Tests for interpolation accuracy

#### 13.3: Implement apply_colormap()
- Support all ColorMap variants (Viridis, Plasma, Inferno, Turbo, Grayscale)
- Map f32 values [0.0, 1.0] to Rgba colors
- Tests for each colormap

#### 13.4: Implement render_heatmap_to_buffer()
- Upscale 20x20 data to target size
- Apply colormap
- Return in-memory ImageBuffer
- Tests with different sizes

#### 13.5: Implement render_overlay()
- Resize heatmap to match original image
- Composite with alpha blending
- Tests with different alpha values

#### 13.6: Implement composite_with_alpha()
- Alpha blend two RGBA images
- Tests for proper blending

Each subtask should follow same 5-step TDD pattern as Tasks 1-12.

---

## Priority Assessment

### Must Have (Blocking)
- **Task 13: Visualization Layer** - Required for debug output and GUI

### Nice to Have (Non-blocking)
- Serialization derives - Only needed for advanced caching
- preprocessed_tensor field - Only if profiling shows benefit
- Method-based API on QualityRawData - Organization preference

### Can Skip
- to_cache_bytes/from_cache_bytes - Only needed for custom disk caching (io_cached already handles this)

---

## Recommendation

**Add Task 13 to the implementation plan** with full TDD detail for the visualization layer. This is the main gap preventing a complete implementation.

All other gaps are minor and can be addressed later based on actual needs.

---

## Validation Checklist

After adding Task 13, verify the implementation plan covers:

- [x] Task 1-3: Core data structures (QualityParams, QualityRawData, QualityScores)
- [x] Task 4-5: Layer separation (raw computation, parameter application)
- [x] Task 6: QualityScores::compute() logic
- [x] Task 7: Caching dependency
- [x] Task 8: Cached compute_quality_raw()
- [x] Task 9-10: CLI integration
- [x] Task 11: Backward compatibility
- [x] Task 12: Visualization types and docs
- [ ] **Task 13: Visualization implementation** ← MISSING

Once Task 13 is added, the implementation plan will be complete and ready for execution.
