# Quality Triage Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable rapid quality assessment and triage of detections across directory with filtering, statistics, heatmap visualization, and keyboard-driven workflow.

**Architecture:** Extend existing DetectionView with quality data structures, add FilterState for quality-based filtering, implement TriageMode for reviewing unknowns, integrate heatmap visualization from beaker lib, and add keyboard shortcuts for efficient triage.

**Tech Stack:** Rust, egui, beaker lib (with quality API), serde for triage overrides

---

## Context

**Already implemented:**
- Proposal 0: File Navigation & Opening (welcome screen, file dialogs, drag & drop)
- Proposal A: Bulk/Directory Mode (in progress - directory processing with progress UI)
- Library architecture: QualityRawData, QualityScores, render_quality_visualization API

**Current state:**
- `DetectionView` shows single image with basic detection info
- Detection struct has: class_name, confidence, x1/y1/x2/y2, blur_score
- TOML parsing reads detection data from beaker output

**What we're building:**
- Extend Detection with full quality data from DetectionQuality struct
- Filter detections by triage_decision (good/bad/unknown)
- Triage mode to review unknowns one-by-one with keyboard shortcuts
- Quality statistics panel showing aggregate metrics
- Sort detections by various quality metrics
- Heatmap visualization overlay (blur probability, tenengrad, weights)
- Export triage overrides for ML training

---

## Task 1: Extend Detection struct with quality data

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:13-26`
- Test: Add unit test in `beaker-gui/src/views/detection.rs`

**Step 1: Write the failing test**

Add test at end of `beaker-gui/src/views/detection.rs`:

```rust
#[test]
fn test_detection_quality_data_parsing() {
    // Test parsing full quality data from TOML
    let toml_data = r#"
[[detect.detections]]
class_name = "head"
confidence = 0.95
x1 = 100.0
y1 = 150.0
x2 = 200.0
y2 = 250.0

[detect.detections.quality]
triage_decision = "surely_good"
triage_rationale = "Sharp and well-covered"
roi_quality_mean = 72.5
roi_blur_probability_mean = 0.15
roi_blur_weight_mean = 0.895
roi_detail_probability = 0.88
size_prior_factor = 0.92
grid_coverage_prior = 0.85
grid_cells_covered = 8.3
core_ring_sharpness_ratio = 2.41
tenengrad_core_mean = 0.045
tenengrad_ring_mean = 0.018
"#;

    let toml_value: toml::Value = toml::from_str(toml_data).unwrap();
    let detections = parse_detections_from_toml(&toml_value);

    assert_eq!(detections.len(), 1);
    let det = &detections[0];
    assert_eq!(det.class_name, "head");

    let quality = det.quality.as_ref().unwrap();
    assert_eq!(quality.triage_decision, "surely_good");
    assert_eq!(quality.triage_rationale, "Sharp and well-covered");
    assert_eq!(quality.roi_quality_mean, 72.5);
    assert_eq!(quality.roi_blur_probability_mean, 0.15);
}
```

**Step 2: Run test to verify it fails**

Run: `just test`
Expected: FAIL - function `parse_detections_from_toml` doesn't exist, Detection doesn't have `quality` field

**Step 3: Write minimal implementation**

Modify `Detection` struct (beaker-gui/src/views/detection.rs:13-26):

```rust
// Simplified detection structure for GUI
#[derive(Clone, Debug)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub quality: Option<DetectionQuality>,
}

/// Quality metrics for a detection (from beaker lib)
#[derive(Clone, Debug)]
pub struct DetectionQuality {
    pub triage_decision: String,  // "surely_good" | "surely_bad" | "unknown"
    pub triage_rationale: String,
    pub roi_quality_mean: f32,
    pub roi_blur_probability_mean: f32,
    pub roi_blur_weight_mean: f32,
    pub roi_detail_probability: f32,
    pub size_prior_factor: f32,
    pub grid_coverage_prior: f32,
    pub grid_cells_covered: f32,
    pub core_ring_sharpness_ratio: f32,
    pub tenengrad_core_mean: f32,
    pub tenengrad_ring_mean: f32,
}
```

Extract parsing logic to helper function:

```rust
/// Parse detections from TOML value
fn parse_detections_from_toml(toml_value: &toml::Value) -> Vec<Detection> {
    let mut detections = Vec::new();

    if let Some(dets) = toml_value
        .get("detect")
        .and_then(|v| v.get("detections"))
        .and_then(|v| v.as_array())
    {
        for det_toml in dets {
            if let Some(det_table) = det_toml.as_table() {
                let class_name = det_table
                    .get("class_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let confidence = det_table
                    .get("confidence")
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.0) as f32;

                let x1 = det_table.get("x1").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                let y1 = det_table.get("y1").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                let x2 = det_table.get("x2").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                let y2 = det_table.get("y2").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;

                // Parse quality data if present
                let quality = det_table.get("quality").and_then(|q| q.as_table()).map(|q| {
                    DetectionQuality {
                        triage_decision: q.get("triage_decision")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        triage_rationale: q.get("triage_rationale")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        roi_quality_mean: q.get("roi_quality_mean")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        roi_blur_probability_mean: q.get("roi_blur_probability_mean")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        roi_blur_weight_mean: q.get("roi_blur_weight_mean")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        roi_detail_probability: q.get("roi_detail_probability")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        size_prior_factor: q.get("size_prior_factor")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        grid_coverage_prior: q.get("grid_coverage_prior")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        grid_cells_covered: q.get("grid_cells_covered")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        core_ring_sharpness_ratio: q.get("core_ring_sharpness_ratio")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        tenengrad_core_mean: q.get("tenengrad_core_mean")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                        tenengrad_ring_mean: q.get("tenengrad_ring_mean")
                            .and_then(|v| v.as_float()).unwrap_or(0.0) as f32,
                    }
                });

                detections.push(Detection {
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    quality,
                });
            }
        }
    }

    detections
}
```

Update `run_detection` method to use helper:

```rust
// In run_detection(), replace the parsing block (lines 118-174) with:
let detections = parse_detections_from_toml(&toml_value);
```

Update existing test to use helper:

```rust
// In test_toml_parsing_detections, replace the parsing block with:
let detections = parse_detections_from_toml(&toml_value);
// Update assertions - quality is now None for this test data
assert!(detections[0].quality.is_none());
```

**Step 4: Run test to verify it passes**

Run: `just test`
Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): extend Detection with full quality data

- Add DetectionQuality struct with all triage metrics
- Extract parse_detections_from_toml() helper
- Update tests to verify quality data parsing
- Maintain backward compatibility (quality is Option<T>)"
```

---

## Task 2: Add quality filter state and UI

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:5-10`
- Modify: `beaker-gui/src/views/detection.rs:189-246`

**Step 1: Write the failing test**

Add test in `beaker-gui/src/views/detection.rs`:

```rust
#[test]
fn test_quality_filter_logic() {
    let detections = vec![
        Detection {
            class_name: "head".to_string(),
            confidence: 0.95,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "surely_good".to_string(),
                triage_rationale: "Sharp".to_string(),
                roi_quality_mean: 75.0,
                roi_blur_probability_mean: 0.1,
                roi_blur_weight_mean: 0.9,
                roi_detail_probability: 0.85,
                size_prior_factor: 0.9,
                grid_coverage_prior: 0.85,
                grid_cells_covered: 8.0,
                core_ring_sharpness_ratio: 2.5,
                tenengrad_core_mean: 0.05,
                tenengrad_ring_mean: 0.02,
            }),
        },
        Detection {
            class_name: "head".to_string(),
            confidence: 0.65,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "unknown".to_string(),
                triage_rationale: "Borderline".to_string(),
                roi_quality_mean: 60.0,
                roi_blur_probability_mean: 0.45,
                roi_blur_weight_mean: 0.7,
                roi_detail_probability: 0.5,
                size_prior_factor: 0.7,
                grid_coverage_prior: 0.6,
                grid_cells_covered: 5.0,
                core_ring_sharpness_ratio: 1.2,
                tenengrad_core_mean: 0.03,
                tenengrad_ring_mean: 0.025,
            }),
        },
        Detection {
            class_name: "head".to_string(),
            confidence: 0.55,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "surely_bad".to_string(),
                triage_rationale: "Too blurry".to_string(),
                roi_quality_mean: 45.0,
                roi_blur_probability_mean: 0.8,
                roi_blur_weight_mean: 0.4,
                roi_detail_probability: 0.2,
                size_prior_factor: 0.5,
                grid_coverage_prior: 0.4,
                grid_cells_covered: 3.0,
                core_ring_sharpness_ratio: 0.8,
                tenengrad_core_mean: 0.01,
                tenengrad_ring_mean: 0.012,
            }),
        },
    ];

    let filter = QualityFilter {
        show_good: true,
        show_unknown: false,
        show_bad: false,
    };

    let filtered = filter.apply(&detections);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].quality.as_ref().unwrap().triage_decision, "surely_good");

    let filter_all = QualityFilter {
        show_good: true,
        show_unknown: true,
        show_bad: true,
    };
    let filtered_all = filter_all.apply(&detections);
    assert_eq!(filtered_all.len(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `just test`
Expected: FAIL - QualityFilter doesn't exist

**Step 3: Write minimal implementation**

Add to `beaker-gui/src/views/detection.rs` after Detection struct:

```rust
/// Quality filtering state
#[derive(Clone, Debug)]
pub struct QualityFilter {
    pub show_good: bool,
    pub show_unknown: bool,
    pub show_bad: bool,
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self {
            show_good: true,
            show_unknown: true,
            show_bad: true,
        }
    }
}

impl QualityFilter {
    /// Apply filter to detections, returning filtered list
    pub fn apply<'a>(&self, detections: &'a [Detection]) -> Vec<&'a Detection> {
        detections
            .iter()
            .filter(|det| {
                if let Some(quality) = &det.quality {
                    match quality.triage_decision.as_str() {
                        "surely_good" => self.show_good,
                        "surely_bad" => self.show_bad,
                        "unknown" => self.show_unknown,
                        _ => true, // Show unknown triage values
                    }
                } else {
                    // Show detections without quality data
                    true
                }
            })
            .collect()
    }

    /// Count detections by triage decision
    pub fn count_by_triage(detections: &[Detection]) -> (usize, usize, usize) {
        let mut good = 0;
        let mut unknown = 0;
        let mut bad = 0;

        for det in detections {
            if let Some(quality) = &det.quality {
                match quality.triage_decision.as_str() {
                    "surely_good" => good += 1,
                    "surely_bad" => bad += 1,
                    "unknown" => unknown += 1,
                    _ => unknown += 1,
                }
            }
        }

        (good, unknown, bad)
    }
}
```

Add filter to DetectionView struct:

```rust
pub struct DetectionView {
    image: Option<DynamicImage>,
    detections: Vec<Detection>,
    texture: Option<egui::TextureHandle>,
    selected_detection: Option<usize>,
    quality_filter: QualityFilter,  // NEW
}
```

Update `new()` method:

```rust
Ok(Self {
    image: Some(image),
    detections,
    texture: None,
    selected_detection: None,
    quality_filter: QualityFilter::default(),  // NEW
})
```

**Step 4: Run test to verify it passes**

Run: `just test`
Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add quality filtering logic

- Add QualityFilter struct with show_good/unknown/bad flags
- Implement apply() to filter detections by triage decision
- Add count_by_triage() for statistics
- Add filter to DetectionView state"
```

---

## Task 3: Add quality filter UI with checkboxes

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:show_detections_list()`

**Step 1: Add filter UI to sidebar**

No test for UI - manual verification.

Update `show_detections_list()` method:

```rust
fn show_detections_list(&mut self, ui: &mut egui::Ui) {
    // RUNTIME ASSERT: If we have a selected detection, it must be in bounds
    if let Some(selected) = self.selected_detection {
        assert!(
            selected < self.detections.len(),
            "DetectionView invariant violated: selected_detection {} >= detection count {}",
            selected,
            self.detections.len()
        );
    }

    ui.heading("Detections");

    // Quality filter UI
    let (good_count, unknown_count, bad_count) = QualityFilter::count_by_triage(&self.detections);

    ui.horizontal(|ui| {
        ui.label("Filter:");
    });

    ui.horizontal(|ui| {
        ui.checkbox(&mut self.quality_filter.show_good, format!("✓ Good ({})", good_count));
        ui.checkbox(&mut self.quality_filter.show_unknown, format!("? Unknown ({})", unknown_count));
        ui.checkbox(&mut self.quality_filter.show_bad, format!("✗ Bad ({})", bad_count));
    });

    ui.separator();

    // Apply filter
    let filtered_detections = self.quality_filter.apply(&self.detections);

    ui.label(format!(
        "Showing {} of {} detections",
        filtered_detections.len(),
        self.detections.len()
    ));
    ui.separator();

    for (original_idx, det) in self.detections.iter().enumerate() {
        // Skip if filtered out
        if !filtered_detections.iter().any(|d| std::ptr::eq(*d, det)) {
            continue;
        }

        let is_selected = self.selected_detection == Some(original_idx);

        // RUNTIME ASSERT: Confidence must be valid
        assert!(
            det.confidence >= 0.0 && det.confidence <= 1.0,
            "Detection confidence out of range: {}",
            det.confidence
        );

        // Styled card for each detection
        let bg_color = if is_selected {
            egui::Color32::from_rgb(230, 240, 255)
        } else {
            // Color-code by quality
            if let Some(quality) = &det.quality {
                match quality.triage_decision.as_str() {
                    "surely_good" => egui::Color32::from_rgb(240, 255, 240), // Light green
                    "surely_bad" => egui::Color32::from_rgb(255, 240, 240),  // Light red
                    "unknown" => egui::Color32::from_rgb(255, 255, 230),     // Light yellow
                    _ => egui::Color32::WHITE,
                }
            } else {
                egui::Color32::WHITE
            }
        };

        egui::Frame::none()
            .fill(bg_color)
            .rounding(6.0)
            .inner_margin(12.0)
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(200)))
            .show(ui, |ui| {
                if ui.selectable_label(false, &det.class_name).clicked() {
                    self.selected_detection = Some(original_idx);
                }
                ui.label(format!("Confidence: {:.2}", det.confidence));

                // Show triage decision
                if let Some(quality) = &det.quality {
                    let triage_text = match quality.triage_decision.as_str() {
                        "surely_good" => "✓ Good",
                        "surely_bad" => "✗ Bad",
                        "unknown" => "? Unknown",
                        _ => "? Unknown",
                    };
                    ui.label(triage_text);
                }
            });

        ui.add_space(8.0);
    }
}
```

**Step 2: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: See filter checkboxes with counts, color-coded detection cards

**Step 3: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add quality filter UI to sidebar

- Show checkboxes for Good/Unknown/Bad with counts
- Color-code detection cards by triage decision
- Display triage decision text on each card
- Filter detection list based on checkbox state"
```

---

## Task 4: Add quality statistics panel

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:show_detections_list()`

**Step 1: Add statistics helper**

Add to `beaker-gui/src/views/detection.rs`:

```rust
/// Quality statistics for a set of detections
#[derive(Default)]
pub struct QualityStats {
    pub good_count: usize,
    pub unknown_count: usize,
    pub bad_count: usize,
    pub total_count: usize,
    pub avg_confidence: f32,
    pub avg_blur_prob: f32,
    pub avg_quality_score: f32,
}

impl QualityStats {
    pub fn compute(detections: &[Detection]) -> Self {
        let (good_count, unknown_count, bad_count) = QualityFilter::count_by_triage(detections);
        let total_count = detections.len();

        let mut sum_confidence = 0.0;
        let mut sum_blur_prob = 0.0;
        let mut sum_quality_score = 0.0;
        let mut count_with_quality = 0;

        for det in detections {
            sum_confidence += det.confidence;
            if let Some(quality) = &det.quality {
                sum_blur_prob += quality.roi_blur_probability_mean;
                sum_quality_score += quality.roi_quality_mean;
                count_with_quality += 1;
            }
        }

        let avg_confidence = if total_count > 0 {
            sum_confidence / total_count as f32
        } else {
            0.0
        };

        let avg_blur_prob = if count_with_quality > 0 {
            sum_blur_prob / count_with_quality as f32
        } else {
            0.0
        };

        let avg_quality_score = if count_with_quality > 0 {
            sum_quality_score / count_with_quality as f32
        } else {
            0.0
        };

        Self {
            good_count,
            unknown_count,
            bad_count,
            total_count,
            avg_confidence,
            avg_blur_prob,
            avg_quality_score,
        }
    }
}
```

**Step 2: Add stats display to UI**

Update `show_detections_list()` to add stats panel after heading:

```rust
fn show_detections_list(&mut self, ui: &mut egui::Ui) {
    // ... existing assertions ...

    ui.heading("Detections");

    // Quality statistics panel
    let stats = QualityStats::compute(&self.detections);

    egui::CollapsingHeader::new("📊 Statistics")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("stats_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Total:");
                    ui.label(format!("{}", stats.total_count));
                    ui.end_row();

                    ui.label("✓ Good:");
                    ui.label(format!("{} ({:.1}%)",
                        stats.good_count,
                        stats.good_count as f32 / stats.total_count as f32 * 100.0
                    ));
                    ui.end_row();

                    ui.label("? Unknown:");
                    ui.label(format!("{} ({:.1}%)",
                        stats.unknown_count,
                        stats.unknown_count as f32 / stats.total_count as f32 * 100.0
                    ));
                    ui.end_row();

                    ui.label("✗ Bad:");
                    ui.label(format!("{} ({:.1}%)",
                        stats.bad_count,
                        stats.bad_count as f32 / stats.total_count as f32 * 100.0
                    ));
                    ui.end_row();

                    ui.separator();
                    ui.separator();
                    ui.end_row();

                    ui.label("Avg Confidence:");
                    ui.label(format!("{:.2}", stats.avg_confidence));
                    ui.end_row();

                    ui.label("Avg Blur Prob:");
                    ui.label(format!("{:.2}", stats.avg_blur_prob));
                    ui.end_row();

                    ui.label("Avg Quality:");
                    ui.label(format!("{:.1}", stats.avg_quality_score));
                    ui.end_row();
                });
        });

    ui.separator();

    // ... rest of existing code (filter UI, detection list) ...
}
```

**Step 3: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: See collapsible statistics panel with counts and percentages

**Step 4: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add quality statistics panel

- Add QualityStats struct with aggregate metrics
- Show collapsible statistics panel with counts and percentages
- Display average confidence, blur probability, quality score
- Format statistics in grid layout"
```

---

## Task 5: Add quality metrics detail view

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:show_detections_list()`

**Step 1: Add detailed metrics display**

Update the detection card rendering in `show_detections_list()`:

```rust
// In the Frame::show() block for each detection card:
egui::Frame::none()
    .fill(bg_color)
    .rounding(6.0)
    .inner_margin(12.0)
    .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(200)))
    .show(ui, |ui| {
        if ui.selectable_label(false, &det.class_name).clicked() {
            self.selected_detection = Some(original_idx);
        }
        ui.label(format!("Confidence: {:.2}", det.confidence));

        // Show triage decision
        if let Some(quality) = &det.quality {
            let triage_text = match quality.triage_decision.as_str() {
                "surely_good" => "✓ Good",
                "surely_bad" => "✗ Bad",
                "unknown" => "? Unknown",
                _ => "? Unknown",
            };
            ui.label(triage_text);

            // Show quality metrics if selected
            if is_selected {
                ui.separator();

                egui::CollapsingHeader::new("Quality Metrics")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(format!("Rationale: {}", quality.triage_rationale));
                        ui.separator();

                        egui::Grid::new(format!("quality_grid_{}", original_idx))
                            .num_columns(2)
                            .spacing([10.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Blur Prob:");
                                ui.label(format!("{:.2}", quality.roi_blur_probability_mean));
                                ui.end_row();

                                ui.label("Quality Score:");
                                ui.label(format!("{:.1}", quality.roi_quality_mean));
                                ui.end_row();

                                ui.label("Detail Prob:");
                                ui.label(format!("{:.2}", quality.roi_detail_probability));
                                ui.end_row();

                                ui.label("Core/Ring:");
                                ui.label(format!("{:.2}", quality.core_ring_sharpness_ratio));
                                ui.end_row();

                                ui.label("Coverage:");
                                ui.label(format!("{:.1} cells", quality.grid_cells_covered));
                                ui.end_row();

                                ui.label("Size Prior:");
                                ui.label(format!("{:.2}", quality.size_prior_factor));
                                ui.end_row();
                            });
                    });
            }
        }
    });
```

**Step 2: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: Click detection to see expanded quality metrics

**Step 3: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add detailed quality metrics view

- Show triage rationale when detection is selected
- Display key quality metrics in collapsible section
- Show blur prob, quality score, detail prob, core/ring ratio
- Show coverage and size prior factors"
```

---

## Task 6: Add triage mode state machine

**Files:**
- Modify: `beaker-gui/src/views/detection.rs`

**Step 1: Write the failing test**

Add test:

```rust
#[test]
fn test_triage_mode_navigation() {
    let detections = vec![
        Detection {
            class_name: "head".to_string(),
            confidence: 0.95,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "surely_good".to_string(),
                triage_rationale: "Good".to_string(),
                roi_quality_mean: 75.0,
                roi_blur_probability_mean: 0.1,
                roi_blur_weight_mean: 0.9,
                roi_detail_probability: 0.85,
                size_prior_factor: 0.9,
                grid_coverage_prior: 0.85,
                grid_cells_covered: 8.0,
                core_ring_sharpness_ratio: 2.5,
                tenengrad_core_mean: 0.05,
                tenengrad_ring_mean: 0.02,
            }),
        },
        Detection {
            class_name: "head".to_string(),
            confidence: 0.65,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "unknown".to_string(),
                triage_rationale: "Borderline".to_string(),
                roi_quality_mean: 60.0,
                roi_blur_probability_mean: 0.45,
                roi_blur_weight_mean: 0.7,
                roi_detail_probability: 0.5,
                size_prior_factor: 0.7,
                grid_coverage_prior: 0.6,
                grid_cells_covered: 5.0,
                core_ring_sharpness_ratio: 1.2,
                tenengrad_core_mean: 0.03,
                tenengrad_ring_mean: 0.025,
            }),
        },
        Detection {
            class_name: "head".to_string(),
            confidence: 0.75,
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            quality: Some(DetectionQuality {
                triage_decision: "unknown".to_string(),
                triage_rationale: "Uncertain".to_string(),
                roi_quality_mean: 65.0,
                roi_blur_probability_mean: 0.35,
                roi_blur_weight_mean: 0.75,
                roi_detail_probability: 0.6,
                size_prior_factor: 0.8,
                grid_coverage_prior: 0.7,
                grid_cells_covered: 6.0,
                core_ring_sharpness_ratio: 1.5,
                tenengrad_core_mean: 0.04,
                tenengrad_ring_mean: 0.022,
            }),
        },
    ];

    let mut triage = TriageMode::new(&detections);

    assert_eq!(triage.total_unknowns(), 2);
    assert_eq!(triage.current_index(), 0);
    assert!(triage.current_detection_index().is_some());

    // Navigate to next
    triage.next();
    assert_eq!(triage.current_index(), 1);

    // Try to go past end
    triage.next();
    assert_eq!(triage.current_index(), 1); // Should stay at last

    // Navigate back
    triage.previous();
    assert_eq!(triage.current_index(), 0);
}
```

**Step 2: Run test to verify it fails**

Run: `just test`
Expected: FAIL - TriageMode doesn't exist

**Step 3: Write minimal implementation**

Add to `beaker-gui/src/views/detection.rs`:

```rust
/// Triage mode for reviewing unknown detections
#[derive(Clone)]
pub struct TriageMode {
    /// Indices of unknown detections in the original detection list
    unknown_indices: Vec<usize>,
    /// Current position in unknown_indices
    current_position: usize,
    /// User overrides: original_index -> new_decision
    overrides: std::collections::HashMap<usize, String>,
}

impl TriageMode {
    /// Create new triage mode for unknown detections
    pub fn new(detections: &[Detection]) -> Self {
        let unknown_indices: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(_, det)| {
                det.quality
                    .as_ref()
                    .map(|q| q.triage_decision == "unknown")
                    .unwrap_or(false)
            })
            .map(|(idx, _)| idx)
            .collect();

        Self {
            unknown_indices,
            current_position: 0,
            overrides: std::collections::HashMap::new(),
        }
    }

    pub fn total_unknowns(&self) -> usize {
        self.unknown_indices.len()
    }

    pub fn current_index(&self) -> usize {
        self.current_position
    }

    pub fn current_detection_index(&self) -> Option<usize> {
        self.unknown_indices.get(self.current_position).copied()
    }

    pub fn next(&mut self) {
        if self.current_position + 1 < self.unknown_indices.len() {
            self.current_position += 1;
        }
    }

    pub fn previous(&mut self) {
        if self.current_position > 0 {
            self.current_position -= 1;
        }
    }

    pub fn mark_as(&mut self, detection_idx: usize, decision: String) {
        self.overrides.insert(detection_idx, decision);
    }

    pub fn get_override(&self, detection_idx: usize) -> Option<&String> {
        self.overrides.get(&detection_idx)
    }

    pub fn reviewed_count(&self) -> usize {
        self.overrides.len()
    }
}
```

**Step 4: Run test to verify it passes**

Run: `just test`
Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add triage mode state machine

- Add TriageMode struct for reviewing unknowns
- Track unknown detection indices
- Support navigation (next/previous)
- Track user overrides (mark as good/bad)
- Add tests for triage navigation logic"
```

---

## Task 7: Add triage mode UI

**Files:**
- Modify: `beaker-gui/src/views/detection.rs`

**Step 1: Add triage mode to DetectionView**

Update DetectionView struct:

```rust
pub struct DetectionView {
    image: Option<DynamicImage>,
    detections: Vec<Detection>,
    texture: Option<egui::TextureHandle>,
    selected_detection: Option<usize>,
    quality_filter: QualityFilter,
    triage_mode: Option<TriageMode>,  // NEW
}
```

Update `new()` method:

```rust
Ok(Self {
    image: Some(image),
    detections,
    texture: None,
    selected_detection: None,
    quality_filter: QualityFilter::default(),
    triage_mode: None,  // NEW
})
```

**Step 2: Add triage mode UI methods**

Add methods to DetectionView:

```rust
impl DetectionView {
    // ... existing methods ...

    fn show_triage_mode_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(triage) = &mut self.triage_mode {
            ui.heading("🔍 Triage Mode");

            ui.label(format!(
                "Reviewing unknown detections: {}/{} reviewed",
                triage.reviewed_count(),
                triage.total_unknowns()
            ));

            if let Some(det_idx) = triage.current_detection_index() {
                if let Some(det) = self.detections.get(det_idx) {
                    ui.separator();

                    // Show current detection info
                    ui.label(format!("Detection {}/{}",
                        triage.current_index() + 1,
                        triage.total_unknowns()
                    ));
                    ui.label(format!("Class: {}", det.class_name));
                    ui.label(format!("Confidence: {:.2}", det.confidence));

                    if let Some(quality) = &det.quality {
                        ui.separator();
                        ui.label(format!("Rationale: {}", quality.triage_rationale));

                        egui::Grid::new("triage_metrics")
                            .num_columns(2)
                            .spacing([10.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Blur Prob:");
                                ui.label(format!("{:.2}", quality.roi_blur_probability_mean));
                                ui.end_row();

                                ui.label("Core/Ring:");
                                ui.label(format!("{:.2}", quality.core_ring_sharpness_ratio));
                                ui.end_row();

                                ui.label("Coverage:");
                                ui.label(format!("{:.1} cells", quality.grid_cells_covered));
                                ui.end_row();
                            });
                    }

                    ui.separator();

                    // Show current override if any
                    if let Some(override_decision) = triage.get_override(det_idx) {
                        ui.label(format!("Marked as: {}", override_decision));
                    }

                    ui.separator();

                    // Action buttons
                    ui.horizontal(|ui| {
                        if ui.button("✓ Good (G)").clicked() {
                            triage.mark_as(det_idx, "surely_good".to_string());
                            triage.next();
                        }
                        if ui.button("✗ Bad (B)").clicked() {
                            triage.mark_as(det_idx, "surely_bad".to_string());
                            triage.next();
                        }
                        if ui.button("? Keep Unknown (U)").clicked() {
                            triage.next();
                        }
                    });

                    ui.separator();

                    // Navigation buttons
                    ui.horizontal(|ui| {
                        if ui.button("← Previous").clicked() {
                            triage.previous();
                        }
                        if ui.button("Next →").clicked() {
                            triage.next();
                        }
                    });
                }
            } else {
                ui.label("No unknown detections to review");
            }

            ui.separator();

            if ui.button("Exit Triage Mode").clicked() {
                self.triage_mode = None;
            }
        }
    }
}
```

**Step 3: Update show_detections_list to support triage mode**

Update `show_detections_list()`:

```rust
fn show_detections_list(&mut self, ui: &mut egui::Ui) {
    // ... existing assertions ...

    ui.heading("Detections");

    // Triage mode UI
    if self.triage_mode.is_some() {
        self.show_triage_mode_ui(ui);
        return; // Don't show normal detection list in triage mode
    }

    // Statistics panel (existing code)
    // ...

    // Filter UI (existing code)
    // ...

    // Add button to enter triage mode
    let (_, unknown_count, _) = QualityFilter::count_by_triage(&self.detections);
    if unknown_count > 0 {
        if ui.button(format!("🔍 Review {} Unknown Detections", unknown_count)).clicked() {
            self.triage_mode = Some(TriageMode::new(&self.detections));
            if let Some(triage) = &self.triage_mode {
                // Select first unknown detection
                if let Some(det_idx) = triage.current_detection_index() {
                    self.selected_detection = Some(det_idx);
                }
            }
        }
        ui.separator();
    }

    // Rest of existing detection list code...
}
```

**Step 4: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: See "Review X Unknown Detections" button, click to enter triage mode

**Step 5: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add triage mode UI

- Add button to enter triage mode when unknowns exist
- Show triage progress and current detection
- Add action buttons: Mark as Good, Bad, or Keep Unknown
- Add navigation buttons: Previous/Next
- Show exit button to return to normal view
- Auto-select current detection in triage mode"
```

---

## Task 8: Add keyboard shortcuts for triage

**Files:**
- Modify: `beaker-gui/src/views/detection.rs:show()`

**Step 1: Add keyboard handling**

Update the `show()` method to handle keyboard input:

```rust
pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
    // RUNTIME ASSERT: View must have an image
    assert!(
        self.image.is_some(),
        "DetectionView invariant violated: no image loaded"
    );

    // Handle keyboard shortcuts
    self.handle_keyboard_shortcuts(ctx);

    egui::SidePanel::right("detections_panel")
        .default_width(crate::style::DETECTION_PANEL_WIDTH)
        .show_inside(ui, |ui| {
            self.show_detections_list(ui);
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        self.show_image_with_bboxes(ui, ctx);
    });
}
```

Add keyboard handler method:

```rust
impl DetectionView {
    // ... existing methods ...

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        // Only handle shortcuts if we're in triage mode
        if let Some(triage) = &mut self.triage_mode {
            if let Some(det_idx) = triage.current_detection_index() {
                // G key: Mark as Good
                if ctx.input(|i| i.key_pressed(egui::Key::G)) {
                    triage.mark_as(det_idx, "surely_good".to_string());
                    triage.next();
                    if let Some(next_idx) = triage.current_detection_index() {
                        self.selected_detection = Some(next_idx);
                    }
                }

                // B key: Mark as Bad
                if ctx.input(|i| i.key_pressed(egui::Key::B)) {
                    triage.mark_as(det_idx, "surely_bad".to_string());
                    triage.next();
                    if let Some(next_idx) = triage.current_detection_index() {
                        self.selected_detection = Some(next_idx);
                    }
                }

                // U key: Keep Unknown (skip)
                if ctx.input(|i| i.key_pressed(egui::Key::U)) {
                    triage.next();
                    if let Some(next_idx) = triage.current_detection_index() {
                        self.selected_detection = Some(next_idx);
                    }
                }

                // Arrow Right: Next
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                    triage.next();
                    if let Some(next_idx) = triage.current_detection_index() {
                        self.selected_detection = Some(next_idx);
                    }
                }

                // Arrow Left: Previous
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    triage.previous();
                    if let Some(prev_idx) = triage.current_detection_index() {
                        self.selected_detection = Some(prev_idx);
                    }
                }

                // Escape: Exit triage mode
                if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.triage_mode = None;
                }
            }
        }
    }
}
```

**Step 2: Add keyboard hint to triage UI**

Update `show_triage_mode_ui()` to show keyboard hints:

```rust
fn show_triage_mode_ui(&mut self, ui: &mut egui::Ui) {
    if let Some(triage) = &mut self.triage_mode {
        ui.heading("🔍 Triage Mode");

        // Show keyboard hints
        ui.horizontal(|ui| {
            ui.label("⌨️ Shortcuts:");
            ui.label("G=Good, B=Bad, U=Skip, ←/→=Navigate, Esc=Exit");
        });
        ui.separator();

        // ... rest of existing triage UI code ...
    }
}
```

**Step 3: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: In triage mode, press G/B/U/arrows to navigate, Esc to exit

**Step 4: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add keyboard shortcuts for triage

- G key: mark as good and advance
- B key: mark as bad and advance
- U key: keep unknown and advance
- Arrow keys: navigate previous/next
- Escape: exit triage mode
- Show keyboard hints in triage UI
- Auto-select detection on navigation"
```

---

## Task 9: Add heatmap visualization support

**Files:**
- Create: `beaker-gui/src/heatmap.rs`
- Modify: `beaker-gui/src/lib.rs`
- Modify: `beaker-gui/src/views/detection.rs`

**Step 1: Create heatmap module**

Create `beaker-gui/src/heatmap.rs`:

```rust
use anyhow::Result;
use std::path::Path;

/// Heatmap layer type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeatmapLayer {
    None,
    BlurProbability,
    Tenengrad,
    Weights,
}

impl HeatmapLayer {
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "None",
            Self::BlurProbability => "Blur Probability",
            Self::Tenengrad => "Tenengrad",
            Self::Weights => "Weights",
        }
    }

    pub fn all() -> &'static [HeatmapLayer] {
        &[
            Self::None,
            Self::BlurProbability,
            Self::Tenengrad,
            Self::Weights,
        ]
    }
}

/// Heatmap state for a detection
pub struct HeatmapState {
    pub current_layer: HeatmapLayer,
    pub opacity: f32,
    pub texture: Option<egui::TextureHandle>,
}

impl Default for HeatmapState {
    fn default() -> Self {
        Self {
            current_layer: HeatmapLayer::None,
            opacity: 0.7,
            texture: None,
        }
    }
}

impl HeatmapState {
    /// Load heatmap image from debug output directory
    pub fn load_heatmap(
        &mut self,
        image_path: &Path,
        layer: HeatmapLayer,
        ctx: &egui::Context,
    ) -> Result<()> {
        if layer == HeatmapLayer::None {
            self.texture = None;
            return Ok(());
        }

        // Find debug images directory
        let stem = image_path.file_stem().unwrap().to_string_lossy();
        let parent = image_path.parent().unwrap();
        let debug_dir = parent.join(format!("quality_debug_images_{}", stem));

        if !debug_dir.exists() {
            anyhow::bail!("Debug images not found. Run detection with --debug-dump-images flag.");
        }

        // Determine which file to load based on layer
        let heatmap_file = match layer {
            HeatmapLayer::BlurProbability => debug_dir.join("p_fused_overlay.jpg"),
            HeatmapLayer::Tenengrad => debug_dir.join("t224_overlay.jpg"),
            HeatmapLayer::Weights => debug_dir.join("weights_overlay.jpg"),
            HeatmapLayer::None => unreachable!(),
        };

        if !heatmap_file.exists() {
            anyhow::bail!("Heatmap file not found: {}", heatmap_file.display());
        }

        // Load image
        let img = image::open(&heatmap_file)?;
        let img_rgba = img.to_rgba8();
        let size = [img_rgba.width() as usize, img_rgba.height() as usize];
        let pixels = img_rgba.into_raw();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);

        // Create texture
        let texture = ctx.load_texture(
            format!("heatmap_{}_{:?}", stem, layer),
            color_image,
            Default::default(),
        );

        self.texture = Some(texture);
        self.current_layer = layer;

        Ok(())
    }
}
```

**Step 2: Add module to lib**

Update `beaker-gui/src/lib.rs`:

```rust
pub mod app;
pub mod recent_files;
pub mod style;
pub mod views;
pub mod heatmap;  // NEW
```

**Step 3: Add heatmap state to DetectionView**

Update DetectionView struct:

```rust
use crate::heatmap::{HeatmapLayer, HeatmapState};

pub struct DetectionView {
    image: Option<DynamicImage>,
    detections: Vec<Detection>,
    texture: Option<egui::TextureHandle>,
    selected_detection: Option<usize>,
    quality_filter: QualityFilter,
    triage_mode: Option<TriageMode>,
    heatmap_state: HeatmapState,  // NEW
    image_path: String,  // NEW - store for heatmap loading
}
```

Update `new()` method:

```rust
pub fn new(image_path: &str) -> Result<Self> {
    // ... existing code ...

    Ok(Self {
        image: Some(image),
        detections,
        texture: None,
        selected_detection: None,
        quality_filter: QualityFilter::default(),
        triage_mode: None,
        heatmap_state: HeatmapState::default(),  // NEW
        image_path: image_path.to_string(),  // NEW
    })
}
```

**Step 4: Commit**

```bash
git add beaker-gui/src/heatmap.rs beaker-gui/src/lib.rs beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add heatmap visualization infrastructure

- Add HeatmapLayer enum with None/BlurProbability/Tenengrad/Weights
- Add HeatmapState for managing heatmap textures
- Add load_heatmap() to load overlay images from debug directory
- Store image_path in DetectionView for heatmap loading
- Prepare for heatmap UI in next commit"
```

---

## Task 10: Add heatmap UI controls

**Files:**
- Modify: `beaker-gui/src/views/detection.rs`

**Step 1: Add heatmap controls to sidebar**

Update `show_detections_list()` to add heatmap controls before statistics:

```rust
fn show_detections_list(&mut self, ui: &mut egui::Ui) {
    // ... existing assertions ...

    ui.heading("Detections");

    // Triage mode UI (existing)
    if self.triage_mode.is_some() {
        self.show_triage_mode_ui(ui);
        return;
    }

    // Heatmap controls
    ui.horizontal(|ui| {
        ui.label("Heatmap:");
        egui::ComboBox::from_id_source("heatmap_layer")
            .selected_text(self.heatmap_state.current_layer.as_str())
            .show_ui(ui, |ui| {
                for layer in HeatmapLayer::all() {
                    let response = ui.selectable_value(
                        &mut self.heatmap_state.current_layer,
                        *layer,
                        layer.as_str(),
                    );

                    if response.clicked() {
                        // Load heatmap when layer changes
                        let image_path = std::path::Path::new(&self.image_path);
                        let _ = self.heatmap_state.load_heatmap(
                            image_path,
                            *layer,
                            ui.ctx(),
                        );
                    }
                }
            });
    });

    if self.heatmap_state.current_layer != HeatmapLayer::None {
        ui.horizontal(|ui| {
            ui.label("Opacity:");
            ui.add(egui::Slider::new(&mut self.heatmap_state.opacity, 0.0..=1.0)
                .show_value(false));
            ui.label(format!("{:.0}%", self.heatmap_state.opacity * 100.0));
        });
    }

    ui.separator();

    // Statistics panel (existing)
    // ... rest of existing code ...
}
```

**Step 2: Update image display to show heatmap overlay**

Update `show_image_with_bboxes()`:

```rust
fn show_image_with_bboxes(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
    // Check if we should show heatmap instead of bounding box image
    if self.heatmap_state.current_layer != HeatmapLayer::None {
        if let Some(heatmap_texture) = &self.heatmap_state.texture {
            // Scale heatmap to fit panel
            let available_size = ui.available_size();
            let image_aspect = heatmap_texture.size()[0] as f32 / heatmap_texture.size()[1] as f32;
            let panel_aspect = available_size.x / available_size.y;

            let size = if image_aspect > panel_aspect {
                egui::vec2(available_size.x, available_size.x / image_aspect)
            } else {
                egui::vec2(available_size.y * image_aspect, available_size.y)
            };

            ui.centered_and_justified(|ui| {
                ui.image((heatmap_texture.id(), size));
            });
            return;
        } else {
            // Show error message if heatmap failed to load
            ui.centered_and_justified(|ui| {
                ui.label("⚠️ Heatmap not available");
                ui.label("Run detection with --debug-dump-images flag");
            });
            return;
        }
    }

    // Show normal bounding box image (existing code)
    if let Some(img) = &self.image {
        // ... existing texture loading and display code ...
    }
}
```

**Step 3: Manual test**

1. First, generate debug images:
   ```bash
   just build-release
   ./target/release/beaker quality path/to/image.jpg --debug-dump-images
   ./target/release/beaker detect path/to/image.jpg
   ```

2. Then test GUI:
   ```bash
   ./target/release/beaker-gui path/to/image.jpg
   ```

Expected: See heatmap dropdown, select different layers to see overlays

**Step 4: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add heatmap UI controls

- Add heatmap layer dropdown (None/Blur/Tenengrad/Weights)
- Add opacity slider for heatmap overlay
- Load heatmap image when layer changes
- Display heatmap instead of bounding box when active
- Show error message if heatmaps not available"
```

---

## Task 11: Export triage overrides to JSON

**Files:**
- Modify: `beaker-gui/src/views/detection.rs`
- Add dependency: `serde_json` to `beaker-gui/Cargo.toml`

**Step 1: Add serde dependency**

Update `beaker-gui/Cargo.toml`:

```toml
[dependencies]
# ... existing dependencies ...
serde_json = "1.0"
```

**Step 2: Add export functionality**

Add to TriageMode impl:

```rust
impl TriageMode {
    // ... existing methods ...

    /// Export triage overrides to JSON file
    pub fn export_to_file(&self, image_path: &Path) -> Result<String> {
        use serde_json::json;

        let output_path = image_path.with_extension("beaker_triage.json");

        let overrides_json: Vec<_> = self.overrides
            .iter()
            .map(|(idx, decision)| {
                json!({
                    "detection_index": idx,
                    "user_decision": decision,
                })
            })
            .collect();

        let export_data = json!({
            "image_path": image_path.to_string_lossy(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "total_unknowns": self.total_unknowns(),
            "reviewed_count": self.reviewed_count(),
            "overrides": overrides_json,
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&export_data)?)?;

        Ok(output_path.to_string_lossy().to_string())
    }
}
```

**Step 3: Add export button to triage UI**

Update `show_triage_mode_ui()`:

```rust
fn show_triage_mode_ui(&mut self, ui: &mut egui::Ui) {
    if let Some(triage) = &mut self.triage_mode {
        // ... existing triage UI code ...

        ui.separator();

        // Export button
        if triage.reviewed_count() > 0 {
            if ui.button("💾 Export Triage Overrides").clicked() {
                let image_path = std::path::Path::new(&self.image_path);
                match triage.export_to_file(image_path) {
                    Ok(path) => {
                        eprintln!("Exported triage overrides to: {}", path);
                        // TODO: Show success notification in UI
                    }
                    Err(e) => {
                        eprintln!("Failed to export triage overrides: {}", e);
                        // TODO: Show error notification in UI
                    }
                }
            }
        }

        if ui.button("Exit Triage Mode").clicked() {
            self.triage_mode = None;
        }
    }
}
```

**Step 4: Add chrono dependency**

Update `beaker-gui/Cargo.toml`:

```toml
[dependencies]
# ... existing dependencies ...
serde_json = "1.0"
chrono = "0.4"
```

**Step 5: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: In triage mode, mark some detections, click "Export Triage Overrides", verify JSON file created

**Step 6: Commit**

```bash
git add beaker-gui/Cargo.toml beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add triage override export

- Add export_to_file() to export overrides as JSON
- Save to {image_stem}.beaker_triage.json
- Include timestamp, counts, and override decisions
- Add export button to triage mode UI
- Add serde_json and chrono dependencies"
```

---

## Task 12: Add sort by quality metrics

**Files:**
- Modify: `beaker-gui/src/views/detection.rs`

**Step 1: Add sort mode enum**

Add to `beaker-gui/src/views/detection.rs`:

```rust
/// Sort mode for detection list
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SortMode {
    Original,           // Original detection order
    ConfidenceDesc,     // Highest confidence first
    BlurProbAsc,        // Lowest blur first (sharpest)
    BlurProbDesc,       // Highest blur first (most blurry)
    QualityScoreDesc,   // Highest quality first
    CoverageDesc,       // Best coverage first
}

impl SortMode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Original => "Original Order",
            Self::ConfidenceDesc => "Confidence ↓",
            Self::BlurProbAsc => "Sharpest First",
            Self::BlurProbDesc => "Most Blurry First",
            Self::QualityScoreDesc => "Quality ↓",
            Self::CoverageDesc => "Coverage ↓",
        }
    }

    pub fn all() -> &'static [SortMode] {
        &[
            Self::Original,
            Self::ConfidenceDesc,
            Self::BlurProbAsc,
            Self::BlurProbDesc,
            Self::QualityScoreDesc,
            Self::CoverageDesc,
        ]
    }

    /// Sort detections by this mode
    pub fn sort(&self, detections: &[Detection]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..detections.len()).collect();

        match self {
            Self::Original => {
                // No sorting needed
            }
            Self::ConfidenceDesc => {
                indices.sort_by(|&a, &b| {
                    detections[b].confidence
                        .partial_cmp(&detections[a].confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Self::BlurProbAsc => {
                indices.sort_by(|&a, &b| {
                    let blur_a = detections[a].quality.as_ref()
                        .map(|q| q.roi_blur_probability_mean)
                        .unwrap_or(1.0);
                    let blur_b = detections[b].quality.as_ref()
                        .map(|q| q.roi_blur_probability_mean)
                        .unwrap_or(1.0);
                    blur_a.partial_cmp(&blur_b).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Self::BlurProbDesc => {
                indices.sort_by(|&a, &b| {
                    let blur_a = detections[a].quality.as_ref()
                        .map(|q| q.roi_blur_probability_mean)
                        .unwrap_or(0.0);
                    let blur_b = detections[b].quality.as_ref()
                        .map(|q| q.roi_blur_probability_mean)
                        .unwrap_or(0.0);
                    blur_b.partial_cmp(&blur_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Self::QualityScoreDesc => {
                indices.sort_by(|&a, &b| {
                    let qual_a = detections[a].quality.as_ref()
                        .map(|q| q.roi_quality_mean)
                        .unwrap_or(0.0);
                    let qual_b = detections[b].quality.as_ref()
                        .map(|q| q.roi_quality_mean)
                        .unwrap_or(0.0);
                    qual_b.partial_cmp(&qual_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Self::CoverageDesc => {
                indices.sort_by(|&a, &b| {
                    let cov_a = detections[a].quality.as_ref()
                        .map(|q| q.grid_cells_covered)
                        .unwrap_or(0.0);
                    let cov_b = detections[b].quality.as_ref()
                        .map(|q| q.grid_cells_covered)
                        .unwrap_or(0.0);
                    cov_b.partial_cmp(&cov_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        indices
    }
}
```

**Step 2: Add sort mode to DetectionView**

Update struct:

```rust
pub struct DetectionView {
    image: Option<DynamicImage>,
    detections: Vec<Detection>,
    texture: Option<egui::TextureHandle>,
    selected_detection: Option<usize>,
    quality_filter: QualityFilter,
    triage_mode: Option<TriageMode>,
    heatmap_state: HeatmapState,
    image_path: String,
    sort_mode: SortMode,  // NEW
}
```

Update `new()`:

```rust
Ok(Self {
    image: Some(image),
    detections,
    texture: None,
    selected_detection: None,
    quality_filter: QualityFilter::default(),
    triage_mode: None,
    heatmap_state: HeatmapState::default(),
    image_path: image_path.to_string(),
    sort_mode: SortMode::Original,  // NEW
})
```

**Step 3: Add sort UI and update detection list**

Update `show_detections_list()`:

```rust
fn show_detections_list(&mut self, ui: &mut egui::Ui) {
    // ... existing code (assertions, heading, triage mode, heatmap controls) ...

    // Sort controls
    ui.horizontal(|ui| {
        ui.label("Sort by:");
        egui::ComboBox::from_id_source("sort_mode")
            .selected_text(self.sort_mode.as_str())
            .show_ui(ui, |ui| {
                for mode in SortMode::all() {
                    ui.selectable_value(&mut self.sort_mode, *mode, mode.as_str());
                }
            });
    });
    ui.separator();

    // Statistics panel (existing)
    // ...

    // Filter UI (existing)
    // ...

    // Apply filter and sort
    let filtered_detections = self.quality_filter.apply(&self.detections);
    let sorted_indices = self.sort_mode.sort(&self.detections);

    ui.label(format!(
        "Showing {} of {} detections",
        filtered_detections.len(),
        self.detections.len()
    ));
    ui.separator();

    // Iterate through sorted indices
    for &original_idx in &sorted_indices {
        let det = &self.detections[original_idx];

        // Skip if filtered out
        if !filtered_detections.iter().any(|d| std::ptr::eq(*d, det)) {
            continue;
        }

        // ... rest of existing detection card rendering code ...
        // (use original_idx consistently)
    }
}
```

**Step 4: Manual test**

Run: `just build-release && ./target/release/beaker-gui path/to/image.jpg`
Expected: See sort dropdown, select different modes to reorder detections

**Step 5: Commit**

```bash
git add beaker-gui/src/views/detection.rs
git commit -m "feat(gui): add sort by quality metrics

- Add SortMode enum with 6 sort options
- Sort by: confidence, blur probability, quality score, coverage
- Add sort dropdown to UI
- Apply sorting after filtering
- Maintain original indices for selection"
```

---

## Task 13: Run full build and tests

**Files:**
- All GUI files

**Step 1: Build release**

Run: `just build-release`
Expected: Successful build

**Step 2: Run all tests**

Run: `just test`
Expected: All tests pass

**Step 3: Run CI locally**

Run: `just ci`
Expected: All checks pass (format, lint, tests)

**Step 4: Commit if any fixes needed**

```bash
# If any fixes were needed:
git add .
git commit -m "fix: address CI issues for quality triage workflow"
```

---

## Task 14: Final integration test and documentation

**Files:**
- Test with real images
- Update README or docs if needed

**Step 1: Manual integration test**

Test complete workflow:

1. Generate test image with quality data:
   ```bash
   just build-release
   ./target/release/beaker quality examples/example.jpg --debug-dump-images
   ./target/release/beaker detect examples/example.jpg
   ```

2. Test GUI workflow:
   ```bash
   ./target/release/beaker-gui examples/example.jpg
   ```

3. Verify features:
   - ✓ Quality metrics shown in detection cards
   - ✓ Filter by good/unknown/bad works
   - ✓ Statistics panel shows correct counts
   - ✓ Triage mode enters correctly
   - ✓ Keyboard shortcuts work (G/B/U/arrows/Esc)
   - ✓ Heatmap layers load and display
   - ✓ Opacity slider works
   - ✓ Sort by different metrics works
   - ✓ Export triage overrides creates JSON file

**Step 2: Test with directory (if Proposal A is complete)**

If directory mode is implemented:

```bash
./target/release/beaker detect examples/
./target/release/beaker-gui examples/
```

Verify:
- ✓ Can filter across all images
- ✓ Triage mode works across directory
- ✓ Statistics show aggregate across all detections

**Step 3: Commit final changes**

```bash
git add .
git commit -m "feat(gui): complete quality triage workflow implementation

Complete implementation of Proposal B: Quality Triage Workflow

Features:
- Extended Detection struct with full quality data
- Quality filtering by triage decision (good/bad/unknown)
- Quality statistics panel with aggregate metrics
- Triage mode for reviewing unknowns one-by-one
- Keyboard shortcuts (G/B/U/arrows/Esc)
- Heatmap visualization with layer selection
- Opacity control for heatmap overlays
- Sort by quality metrics (6 modes)
- Export triage overrides to JSON

This enables rapid quality assessment and triage of detections
across directories, with visual feedback via heatmaps and
efficient keyboard-driven workflow."
```

---

## Task 15: Push changes

**Files:**
- All committed changes

**Step 1: Verify branch**

Run: `git branch`
Expected: On `claude/quality-triage-workflow-plan-011CUWX3aQY8xczSTBeq6Xre`

**Step 2: Push to remote**

Run: `git push -u origin claude/quality-triage-workflow-plan-011CUWX3aQY8xczSTBeq6Xre`
Expected: Successful push

**Step 3: Verify CI passes**

Check GitHub Actions for CI status.
Expected: All checks pass

---

## Summary

This plan implements **Proposal B: Quality Triage Workflow** with the following features:

**Core Functionality:**
- Extended Detection struct with 12 quality metrics
- Quality filtering (Good/Unknown/Bad) with counts
- Quality statistics panel (percentages, averages)
- Triage mode for reviewing unknowns sequentially
- Keyboard shortcuts for efficient triage (G/B/U/arrows/Esc)
- Sort by 6 quality metrics
- Export triage overrides to JSON

**Visualization:**
- Heatmap layer selection (Blur/Tenengrad/Weights)
- Opacity control for overlays
- Color-coded detection cards by triage decision
- Detailed quality metrics display per detection

**User Experience:**
- Keyboard-driven workflow for power users
- Clear visual feedback (color coding, progress)
- Export overrides for ML training
- Graceful handling of missing quality data

**Implementation Stats:**
- ~15 tasks over 2-3 days
- ~800-1000 LOC added to beaker-gui
- TDD approach with tests for core logic
- Backward compatible (quality data is optional)

This enables users to rapidly triage 10-100+ detections across directories with visual quality feedback and efficient keyboard navigation.
