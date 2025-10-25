# Detection Features Plan - Beaker GUI

**Date:** 2025-10-25
**Status:** Proposal Phase
**Context:** MVP detection view exists with basic bounding box display

---

## Current State

**Working MVP features:**
- Detection view with bounding boxes rendered by beaker lib
- Sidebar listing detections with class name, confidence, blur score
- Image display with aspect-ratio-preserving scaling
- Basic selection (click to highlight)

**Available detection data** (from beaker lib):
- **Basic**: class_name, confidence, x1/y1/x2/y2, angle_radians, class_id
- **Quality metrics** (DetectionQuality struct):
  - `triage_decision`: "good" | "bad" | "unknown"
  - `triage_rationale`: human-readable explanation
  - `roi_quality_mean`: PaQ quality score in ROI
  - `roi_blur_probability_mean`: blur probability (0-1)
  - `roi_blur_weight_mean`: blur weight
  - `roi_detail_probability`: native-res detail (0-1)
  - `size_prior_factor`: detection size factor (0-1)
  - `grid_coverage_prior`: coverage factor (0-1)
  - `grid_cells_covered`: cells covered (raw count)
  - `core_ring_sharpness_ratio`: subject vs background sharpness
  - `tenengrad_core_mean`: core sharpness metric
  - `tenengrad_ring_mean`: ring sharpness metric

---

## Feature Proposals

### Proposal A: Filter & Sort Foundation (Essential)

**Goal:** Let users find detections of interest quickly through filtering and sorting.

**Features:**
1. **Filter by quality triage**
   - Checkboxes: ☑ Good  ☑ Unknown  ☑ Bad
   - Update display in real-time
   - Show count per category (e.g., "Good (12) Bad (3) Unknown (5)")

2. **Filter by confidence threshold**
   - Slider: 0.0 to 1.0 with live preview
   - Show filtered count vs total

3. **Filter by class**
   - Multi-select for "head", "bird", etc.
   - Show count per class

4. **Sort detections**
   - Dropdown: "Confidence ↓", "Confidence ↑", "Blur score ↓", "Blur score ↑", "Size ↓", "Coverage ↓"
   - Apply to sidebar list

5. **Search/filter bar**
   - Quick text filter by class name or index

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ Detections (12/20 shown)         [Clear]   │
│ ─────────────────────────────────────────   │
│ Quality: ☑Good ☑Unknown ☐Bad               │
│ Confidence: [====|=====] 0.65              │
│ Class: ☑head ☑bird                         │
│ Sort: [Confidence ↓         ▾]             │
│ ─────────────────────────────────────────   │
│ □ head #1  Conf: 0.95  ✓Good               │
│ □ head #2  Conf: 0.87  ?Unknown            │
│ ...                                         │
└─────────────────────────────────────────────┘
```

**Why essential:**
- With many detections (10-50+), filtering is critical for usability
- Quality triage is the most important filter (good/bad/unknown)
- Low implementation complexity, high user value

**Implementation notes:**
- Store filter state in `DetectionView`
- Apply filters when rendering sidebar
- Use egui's built-in widgets (Checkbox, Slider, ComboBox)
- ~200 LOC, 1-2 days work

---

### Proposal B: Rich Quality Metrics Display (High Value)

**Goal:** Surface all quality metrics in an intuitive, visual way.

**Features:**
1. **Expanded detection cards**
   - Collapsible sections showing all quality metrics
   - Color-coded triage badges (green=good, yellow=unknown, red=bad)
   - Show triage rationale as tooltip or expandable text

2. **Quality metrics panel** (when detection selected)
   - Grid layout with all 12 quality metrics
   - Visual indicators (progress bars, gauges)
   - Contextual explanations (what does "core_ring_sharpness_ratio" mean?)

3. **Metrics visualization**
   - Mini sparklines showing metric distribution across all detections
   - Highlight where current detection falls

4. **Quality overview** (top of sidebar)
   - Summary stats: "12 good, 5 unknown, 3 bad"
   - Quality distribution histogram
   - Average confidence, blur scores

**UI mockup (expanded card):**
```
┌─────────────────────────────────────────────┐
│ □ head #1  Conf: 0.95  [✓ Good]            │
│   ├─ Blur: 0.12  Detail: 0.89              │
│   ├─ Size: 0.92  Coverage: 8.3 cells       │
│   └─ Core/Ring: 2.41 (sharp subject)       │
│   [Show all metrics ▾]                     │
│                                             │
│   ┌─ All Quality Metrics ─────────────┐    │
│   │ Triage: Good ✓                    │    │
│   │ "Sharp enough and well covered"   │    │
│   │                                    │    │
│   │ ROI Quality Mean:      ▓▓▓▓░  67  │    │
│   │ Blur Probability:      ░░░▓▓  0.12│    │
│   │ Detail Probability:    ▓▓▓▓▓  0.89│    │
│   │ Core/Ring Sharpness:   ▓▓▓▓░  2.41│    │
│   │ Grid Coverage:         ▓▓▓▓░  8.3 │    │
│   │ Size Prior:            ▓▓▓▓░  0.92│    │
│   │ ...                                │    │
│   └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Why high value:**
- Beaker has rich quality data - surfacing it makes the tool powerful
- Helps users understand *why* a detection is good/bad
- Educational: users learn what makes a quality detection
- Enables manual quality review workflows

**Implementation notes:**
- Extend `Detection` struct to include all quality fields
- Parse quality metrics from TOML metadata
- Use egui's Grid, ProgressBar widgets
- Add tooltips with metric explanations
- ~400 LOC, 2-3 days work

---

### Proposal C: Interactive Visualization Modes (Medium Priority)

**Goal:** Let users visualize detections in different ways for different analysis needs.

**Features:**
1. **Visualization mode toggle**
   - "Boxes + Angles" (current)
   - "Quality heatmap" (color-code boxes by quality)
   - "Confidence heatmap" (color-code boxes by confidence)
   - "Minimal" (no boxes, just image)

2. **Quality-based coloring**
   - Good = green, Unknown = yellow, Bad = red
   - Override default box colors

3. **Transparency control**
   - Slider to adjust box/overlay opacity
   - Useful when boxes obscure image details

4. **Angle visualization options**
   - Toggle angle arrows on/off
   - Show angle values as text overlay
   - Color-code by angle range (group orientations)

5. **Hover preview**
   - Hover over detection card → highlight box on image
   - Hover over box → show mini tooltip with key metrics

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ View Mode: [Boxes+Angles ▾]                │
│ Color by:  [Quality      ▾]                │
│ Opacity:   [▓▓▓▓░] 80%                     │
│ Show angles: ☑  Show labels: ☐             │
│ ─────────────────────────────────────────   │
│   [Image with color-coded boxes]           │
└─────────────────────────────────────────────┘
```

**Why medium priority:**
- Enhances exploration and analysis workflows
- Quality heatmap is particularly valuable for quick triage
- Adds visual polish

**Implementation notes:**
- Re-render bounding box image with custom colors based on mode
- Use egui::painter for real-time overlays (vs re-running beaker lib)
- Store visualization preferences in DetectionView state
- ~300 LOC, 2-3 days work

---

### Proposal D: Angle/Orientation Analysis (Specialized)

**Goal:** Leverage angle data for orientation-specific analysis.

**Features:**
1. **Angle distribution histogram**
   - Circular histogram showing angle distribution
   - Click to filter detections by angle range

2. **Group by orientation**
   - Auto-group into bins: "Facing left", "Facing right", "Facing up", etc.
   - Collapsible groups in sidebar

3. **Angle statistics**
   - Mean, median, std dev of angles
   - Detect clustering (e.g., "most birds facing 45°")

4. **Angle-based sorting**
   - Sort by angle ascending/descending
   - Group similar angles together

5. **Compass visualization**
   - Small compass showing each detection's orientation
   - Click to jump to detection

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ Angle Analysis                              │
│ ─────────────────────────────────────────   │
│  ┌─ Angle Distribution ─┐                   │
│  │      N                │                   │
│  │   W  +  E  ◄ circular │                   │
│  │      S     histogram  │                   │
│  └───────────────────────┘                   │
│                                              │
│ Mean angle: 42° (NE)                         │
│ Most common: 30-60° (8 detections)          │
│ ─────────────────────────────────────────   │
│ ▼ Facing Right (30-150°)  [8]               │
│   □ head #1  95° Conf: 0.95                 │
│   □ head #2  102° Conf: 0.87                │
│ ▶ Facing Left (210-330°)  [3]               │
│ ▶ Facing Up (330-30°)     [1]               │
└─────────────────────────────────────────────┘
```

**Why specialized:**
- Orientation is unique to bird detection use case
- Valuable for behavioral analysis
- Requires moderate implementation effort

**Implementation notes:**
- Use egui custom painting for circular histogram
- Angle binning logic for grouping
- Parse angle_radians from detection data
- ~350 LOC, 3-4 days work

---

### Proposal E: Comparison & Export Tools (Power User)

**Goal:** Enable advanced workflows: comparing detections, exporting results.

**Features:**
1. **Multi-selection**
   - Checkbox per detection
   - Bulk actions: "Export selected", "Delete selected", "Compare selected"

2. **Side-by-side comparison**
   - Select 2-4 detections → open comparison view
   - Show crops side-by-side with all metrics
   - Highlight differences (which has better quality?)

3. **Export filtered results**
   - Export current filtered set as JSON/CSV
   - Include selected quality metrics
   - Export cropped images for selected detections

4. **Statistics panel**
   - Show aggregates: avg confidence, quality distribution
   - Export summary report

5. **Copy metrics to clipboard**
   - Click to copy detection data for external analysis

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ Detections (3 selected)    [Compare] [Export]│
│ ─────────────────────────────────────────   │
│ ☑ head #1  Conf: 0.95  ✓Good               │
│ ☐ head #2  Conf: 0.87  ?Unknown            │
│ ☑ head #3  Conf: 0.92  ✓Good               │
│ ☑ head #4  Conf: 0.45  ✗Bad                │
│ ─────────────────────────────────────────   │
│ [Comparison View: 3 detections]             │
│ ┌───────┬───────┬───────┐                   │
│ │ #1    │ #3    │ #4    │                   │
│ │ [crop]│ [crop]│ [crop]│                   │
│ │ Good  │ Good  │ Bad   │                   │
│ │ 0.95  │ 0.92  │ 0.45  │                   │
│ │ Blur: │ Blur: │ Blur: │                   │
│ │ 0.12  │ 0.15  │ 0.89  │ ← Outlier!        │
│ └───────┴───────┴───────┘                   │
└─────────────────────────────────────────────┘
```

**Why power user:**
- Valuable for research/analysis workflows
- Enables integration with external tools
- Comparison helps calibrate quality intuition

**Implementation notes:**
- Add selection state to DetectionView
- Implement export using serde_json, csv crate
- Comparison view as separate panel/window
- Use beaker lib to re-generate crops for comparison
- ~500 LOC, 4-5 days work

---

### Proposal F: Zoom & Pan (Essential for Large Images)

**Goal:** Navigate high-resolution images effectively.

**Features:**
1. **Zoom controls**
   - Mouse wheel to zoom in/out
   - Zoom slider in toolbar
   - Fit-to-window, 100%, 200% buttons

2. **Pan**
   - Click-and-drag to pan zoomed image
   - Mini-map showing current viewport

3. **Zoom to detection**
   - Click detection in sidebar → zoom and center on that detection
   - Highlight with animated border

4. **Overview + detail**
   - Split view: overview (left) + zoomed detail (right)
   - Click overview to set detail focus

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ Zoom: [- ◼ +] [Fit] [100%] [200%]          │
│ ─────────────────────────────────────────   │
│ ┌───────────────┬─────────────────┐         │
│ │ [overview]    │ [zoomed detail] │         │
│ │   ┌───┐       │                 │         │
│ │   │ ░ │<focus │  ▓▓▓▓▓          │         │
│ │   └───┘       │  ▓head▓ 0.95    │         │
│ │               │  ▓▓▓▓▓          │         │
│ └───────────────┴─────────────────┘         │
└─────────────────────────────────────────────┘
```

**Why essential:**
- High-res images (e.g., 4K) need zoom to see details
- "Zoom to detection" is a killer workflow improvement
- Expected feature in any image viewer

**Implementation notes:**
- Use egui's Image widget with custom pan/zoom logic
- Track zoom level and pan offset in DetectionView state
- Transform mouse coords for detection hit-testing
- ~400 LOC, 3-4 days work
- **Challenge**: May need to re-render bounding boxes at different zoom levels for quality

---

### Proposal G: Keyboard Shortcuts & Accessibility (Polish)

**Goal:** Make the app fast and accessible for power users.

**Features:**
1. **Keyboard navigation**
   - Arrow keys: navigate between detections
   - Space: toggle selection
   - Enter: zoom to selected detection
   - 1-5: filter by quality (1=Good, 2=Unknown, 3=Bad, 4=All)
   - G/B/U: jump to next Good/Bad/Unknown

2. **Shortcuts palette**
   - Ctrl+K or Cmd+K: open command palette
   - Fuzzy search for actions: "filter by good", "export selected", etc.

3. **Status bar**
   - Show current selection, filter status
   - Keyboard hint (e.g., "Press ↑↓ to navigate")

4. **Accessibility**
   - High contrast mode
   - Configurable font sizes
   - Screen reader support (egui has some support)

**UI mockup:**
```
┌─────────────────────────────────────────────┐
│ [Command Palette: Ctrl+K]                   │
│ ┌─────────────────────────────────────────┐ │
│ │ > filter_________________               │ │
│ │   Filter by Good Quality                │ │
│ │   Filter by Confidence > 0.8            │ │
│ │   Clear All Filters                     │ │
│ └─────────────────────────────────────────┘ │
│                                              │
│ Status: 12 detections shown | 3 selected    │
│ Tip: Press G to jump to next Good detection │
└─────────────────────────────────────────────┘
```

**Why polish:**
- Dramatically improves productivity for repeat users
- Command palette is modern UX pattern (VSCode, Obsidian, etc.)
- Low cost, high perceived quality

**Implementation notes:**
- Use egui's input handling for key events
- Store keyboard shortcuts in a registry
- Command palette as modal overlay
- ~250 LOC, 2 days work

---

## Recommended Implementation Roadmap

### Phase 1: Essential Foundations (1-2 weeks)
**Goal:** Make the app usable for real workflows with 10+ detections

1. **Proposal A: Filter & Sort** (essential)
   - Quality triage filter (good/bad/unknown)
   - Confidence threshold slider
   - Sort dropdown

2. **Proposal F: Zoom & Pan** (essential for hi-res images)
   - Basic zoom in/out
   - Click-and-drag pan
   - Zoom to detection

**Deliverable:** Users can filter 50 detections down to 10 good ones, then zoom in to inspect each.

---

### Phase 2: Rich Data Surfacing (2-3 weeks)
**Goal:** Leverage beaker's rich quality data, make app educational

3. **Proposal B: Rich Quality Metrics** (high value)
   - Expanded detection cards with all metrics
   - Quality overview stats
   - Metric explanations/tooltips

4. **Proposal C: Visualization Modes** (partial)
   - Quality heatmap coloring
   - Toggle angle arrows
   - Hover preview

**Deliverable:** Users understand *why* detections are good/bad, can quickly triage by color.

---

### Phase 3: Specialized & Power Features (2-3 weeks)
**Goal:** Enable advanced analysis workflows

5. **Proposal D: Angle Analysis** (specialized)
   - Angle histogram
   - Group by orientation
   - Angle-based filtering

6. **Proposal E: Comparison & Export** (power user)
   - Multi-selection
   - Side-by-side comparison
   - Export JSON/CSV

7. **Proposal G: Keyboard Shortcuts** (polish)
   - Arrow key navigation
   - Command palette
   - Status bar

**Deliverable:** Power users can analyze 100+ images efficiently, export results for further analysis.

---

## Alternative: Phased Rollout by Use Case

Instead of feature-by-feature, implement by user workflow:

### Workflow 1: Quick Quality Triage (Week 1)
- Filter by quality (A)
- Quality heatmap coloring (C)
- Zoom to detection (F)

→ **User can quickly find and inspect good detections**

### Workflow 2: Deep Quality Analysis (Week 2-3)
- Full quality metrics display (B)
- Angle analysis (D)
- Comparison view (E)

→ **User can understand quality in depth, compare detections**

### Workflow 3: Batch Processing (Week 4)
- Export filtered results (E)
- Keyboard shortcuts (G)
- Statistics panel (E)

→ **User can process many images efficiently**

---

## Open Questions

1. **Cutout handling**: User said "skip cutout stuff for now" - does this mean:
   - Don't show crop previews in comparison view?
   - Don't implement export cropped images?
   - Answer: Probably means don't add features *requiring* cutouts, but can use them if already generated

2. **Multi-image mode**: Should the app support multiple images at once?
   - Tab-based image switching?
   - Aggregate statistics across images?
   - This would be a separate major feature

3. **Real-time re-detection**: Should users be able to adjust confidence threshold and re-run detection?
   - Probably not in Phase 1 (too slow, complex state management)
   - Could add in Phase 3 with proper caching

4. **Annotation mode**: Should users be able to manually correct/annotate detections?
   - Add/remove bounding boxes
   - Adjust angles
   - Mark false positives
   - Would require new "annotation view" - significant work

5. **Performance**: With 50+ detections, will sidebar scroll be smooth?
   - Use egui::ScrollArea with virtual scrolling if needed
   - Lazy-load detection data

---

## Testing Strategy

Each proposal should include:

1. **Unit tests** for pure logic (filtering, sorting, angle binning)
2. **Integration tests** exercising the full view (use egui_kittest)
3. **Manual test scenarios** (e.g., "Load image with 50 detections, filter to 10, zoom to detection #3")
4. **Snapshot tests** for 1-2 key states per proposal

**Testing priorities:**
- Phase 1: Test filtering logic, zoom math thoroughly (critical for usability)
- Phase 2: Test quality metric parsing, color mapping
- Phase 3: Test export format correctness, comparison view layout

---

## UI/UX Principles

**Design goals:**
- **Information density**: Show lots of data without overwhelming
- **Progressive disclosure**: Hide complexity behind expand/collapse
- **Visual hierarchy**: Most important info (quality triage, confidence) stands out
- **Contextual help**: Tooltips explain technical metrics
- **Consistency**: Same patterns across proposals (e.g., collapsible cards)

**Egui patterns to use:**
- `egui::CollapsingHeader` for expandable sections
- `egui::Grid` for metric layouts
- `egui::ProgressBar` for 0-1 normalized metrics
- `egui::Slider` for thresholds
- `egui::combo_box_with_label` for dropdowns
- `egui::color_picker` for color scheme customization (future)

---

## Summary

**Recommended MVP+1 scope** (4-6 weeks):
1. **Proposal A** - Filter & Sort (essential)
2. **Proposal F** - Zoom & Pan (essential)
3. **Proposal B** - Rich Quality Metrics (high value)
4. **Proposal C** - Visualization Modes (partial - just quality heatmap)

This delivers a **featureful, production-ready app** that:
- Handles real workflows (10-50 detections per image)
- Surfaces beaker's unique quality data
- Feels polished and professional
- Serves as foundation for power features later

**Want maximum features?** Add Proposal D (Angle Analysis) and Proposal E (Export) for a **6-8 week super-app**.

**Want minimal but solid?** Just A + F (2 weeks) for a **usable, zoomable, filterable viewer**.
