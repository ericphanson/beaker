# Detection Features Plan - Beaker GUI

**Date:** 2025-10-25
**Status:** Proposal Phase
**Context:** MVP detection view exists with basic bounding box display

---

## Current State & Usage Patterns

**Working MVP features:**
- Detection view with bounding boxes rendered by beaker lib
- Sidebar listing detections with class name, confidence, blur score
- Image display with aspect-ratio-preserving scaling
- Basic selection (click to highlight)

**Typical usage:**
- **Primary use case**: Processing directories of images (10-100+ images)
- **Typical detections per image**: 1-2 detections
- **Total detections across directory**: 10-200 detections to triage
- **Secondary use case**: Single-image analysis (occasional users)

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

**Available debug visualizations** (`--debug-dump-images` flag):
- **Heatmaps**: Tenengrad (t224, t112), blur probability (p224, p112), fused blur (pfused), weights
- **Overlays**: All heatmaps overlaid on original image at 55% alpha
- **Colormap**: Turbo-like colormap (blue=low, cyan, yellow, orange, red=high)

---

## Feature Architecture: Single Image vs Bulk Mode

### Single Image Mode (Current MVP)
**Scope:** Analyze 1 image with 1-2 detections
- Simple sidebar with 1-2 detection cards
- Image viewer with bounding boxes
- Click to select/highlight detection

### Bulk/Directory Mode (Primary Focus)
**Scope:** Analyze 10-100+ images, 10-200 total detections
- Gallery view with thumbnails
- Aggregate filtering/sorting across all images
- Quality triage workflow
- Batch operations

**Key insight:** With 1-2 detections per image, filtering/sorting only makes sense when aggregating across multiple images.

---

## Core Workflow: Running Detection from GUI

**IMPORTANT:** The GUI doesn't just load existing .beaker.toml files - it **runs detection itself**.

### Basic Workflow
1. Open app → Welcome screen
2. Click "Open Folder" → Select directory with images
3. **App runs `beaker detect` on directory** (not just loading existing results!)
4. Show progress bar during processing
5. Surface errors gracefully (per-image checkmarks as processed)
6. Note: `detect` runs `quality` first → 2 passes through directory, 1 image at a time
7. Browse results: original image + detections + quality info

### Advanced Workflows (build on basic)
- Quality heatmap overlays
- Filtering by quality/confidence
- Triage unknown detections
- Export results

### Lib→GUI Interop Requirements

**Critical architecture question:** How does GUI get progress updates from beaker lib?

---

## Current Beaker Architecture Analysis

**Existing progress infrastructure** (`beaker/src/model_processing.rs:88-375`):

The good news: **Beaker already has most of what we need!**

1. **Sequential per-image processing** (lines 243-327):
   ```rust
   for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
       match P::process_single_image(&mut session, image_path, &config, &output_manager) {
           Ok(result) => {
               successful_count += 1;
               // ... save metadata ...
           }
           Err(e) => {
               failed_count += 1;
               failed_images.push(image_path.to_str().unwrap_or_default().to_string());
               warn!("Failed to process {}:\n {}", image_path.display(), e);
               // CONTINUES PROCESSING - doesn't crash!
           }
       }
   }
   ```
   **✅ Already has graceful per-image error handling!**

2. **Existing progress bar support** (`beaker/src/color_utils.rs:273-288`):
   ```rust
   pub fn create_batch_progress_bar(total: usize) -> Option<ProgressBar> {
       if total > 1 && stderr().is_terminal() {
           let pb = ProgressBar::new(total as u64);
           add_progress_bar(pb.clone());
           pb.set_prefix(format!("[{}/{}] Processing {}", index + 1, total, filename));
           pb.set_message(format!("ETA: {eta:.1}s"));
           // ...
       }
   }
   ```
   **✅ Already tracks progress with indicatif!**

3. **Global multi-progress infrastructure** (`beaker/src/progress.rs:1-32`):
   ```rust
   static MULTI: Lazy<Arc<MultiProgress>> = Lazy::new(|| Arc::new(MultiProgress::new()));

   pub fn add_progress_bar(pb: indicatif::ProgressBar) {
       global_mp().add(pb);
   }
   ```
   **✅ Already has global progress management!**

4. **Two-pass processing** (`beaker/src/detection.rs:95-112`):
   ```rust
   pub fn run_detection(config: DetectionConfig) -> Result<usize> {
       log::info!("   Analyzing image quality");
       let quality_results = run_model_processing_with_quality_outputs::<QualityProcessor>(quality_config);

       log::info!("   Detecting...");
       run_model_processing::<DetectionProcessor>(config)
   }
   ```
   **✅ Quality → Detection is explicit!**

5. **Strict vs permissive mode** (`beaker/src/model_processing.rs:368-373`):
   ```rust
   // If strict mode is enabled, fail if any images failed
   if config.base().strict && failed_count > 0 {
       return Err(anyhow::anyhow!(
           "{} image(s) failed to process (without `--permissive` flag)", failed_count
       ));
   }
   ```
   **✅ Already has configurable error handling!**

---

## What Actually Needs to Change

The current architecture uses `indicatif::ProgressBar` for CLI output. For GUI, we need to **extract the progress events** before they go to indicatif.

### Minimal Change Approach: Event Channel

Add a **single optional callback channel** to the existing processing loop. This is <100 LOC change to beaker lib.

**New types** (add to `beaker/src/model_processing.rs`):

```rust
use std::sync::mpsc::Sender;

/// Progress events emitted during processing
#[derive(Debug, Clone)]
pub enum ProcessingEvent {
    /// Processing started for an image
    ImageStart {
        path: PathBuf,
        index: usize,
        total: usize,
        stage: ProcessingStage,
    },

    /// Image processing completed (success or failure)
    ImageComplete {
        path: PathBuf,
        index: usize,
        result: ProcessingResult,
    },

    /// Stage transition (quality → detection)
    StageChange {
        stage: ProcessingStage,
        images_total: usize,
    },

    /// Overall progress update
    Progress {
        completed: usize,
        total: usize,
        stage: ProcessingStage,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum ProcessingStage {
    Quality,
    Detection,
}

#[derive(Debug, Clone)]
pub enum ProcessingResult {
    Success {
        detections_count: usize,
        good_count: usize,
        bad_count: usize,
        unknown_count: usize,
        processing_time_ms: f64,
    },
    Error {
        error_message: String,
    },
}
```

**Modified processing function** (changes to `run_model_processing_with_quality_outputs`):

```rust
pub fn run_model_processing_with_quality_outputs<P: ModelProcessor>(
    config: P::Config,
    progress_tx: Option<Sender<ProcessingEvent>>,  // ← NEW PARAMETER
) -> Result<(usize, HashMap<String, QualityResult>)> {
    // ... existing setup code ...

    for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
        // Emit start event
        if let Some(ref tx) = progress_tx {
            let _ = tx.send(ProcessingEvent::ImageStart {
                path: image_path.clone(),
                index,
                total: image_files.len(),
                stage: /* determine current stage */,
            });
        }

        match P::process_single_image(&mut session, image_path, &config, &output_manager) {
            Ok(result) => {
                successful_count += 1;

                // Emit success event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: ProcessingResult::Success {
                            // extract from result...
                        },
                    });
                }
                // ... existing success handling ...
            }
            Err(e) => {
                failed_count += 1;

                // Emit error event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: ProcessingResult::Error {
                            error_message: e.to_string(),
                        },
                    });
                }
                // ... existing error handling ...
            }
        }

        // Progress bar updates (existing code unchanged for CLI)
        if let Some(ref pb) = progress_bar {
            pb.inc(1);
        }
    }

    Ok((successful_count, quality_results))
}
```

**Modified detection entry point** (changes to `run_detection`):

```rust
pub fn run_detection(
    config: DetectionConfig,
    progress_tx: Option<Sender<ProcessingEvent>>,  // ← NEW PARAMETER
) -> Result<usize> {
    // Quality stage
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(ProcessingEvent::StageChange {
            stage: ProcessingStage::Quality,
            images_total: /* count from config */,
        });
    }

    let quality_config = QualityConfig::from_detection_config(&config);
    let quality_results = run_model_processing_with_quality_outputs::<QualityProcessor>(
        quality_config,
        progress_tx.clone(),  // ← Pass through
    );

    let config = match quality_results {
        Ok((_count, results)) => DetectionConfig {
            quality_results: Some(results),
            ..config
        },
        Err(_) => config,
    };

    // Detection stage
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(ProcessingEvent::StageChange {
            stage: ProcessingStage::Detection,
            images_total: /* count from config */,
        });
    }

    run_model_processing::<DetectionProcessor>(config, progress_tx)
}
```

---

## GUI Implementation

**In beaker-gui** (`beaker-gui/src/processing.rs`):

```rust
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use beaker::{DetectionConfig, ProcessingEvent};

pub struct ProcessingState {
    pub images: Vec<ImageState>,
    pub current_stage: ProcessingStage,
    pub overall_progress: f32,
}

pub enum ImageState {
    Waiting,
    Processing,
    Success { detections: Vec<Detection> },
    Error { message: String },
}

impl ProcessingState {
    pub fn start_processing(config: DetectionConfig) -> (Self, Receiver<ProcessingEvent>) {
        let (tx, rx) = channel();

        // Spawn background thread
        thread::spawn(move || {
            if let Err(e) = beaker::run_detection(config, Some(tx.clone())) {
                // Send error event
                let _ = tx.send(ProcessingEvent::Error { /* ... */ });
            }
        });

        let state = ProcessingState {
            images: vec![],
            current_stage: ProcessingStage::Quality,
            overall_progress: 0.0,
        };

        (state, rx)
    }

    pub fn update(&mut self, event: ProcessingEvent) {
        match event {
            ProcessingEvent::ImageStart { path, index, .. } => {
                if index < self.images.len() {
                    self.images[index] = ImageState::Processing;
                }
            }
            ProcessingEvent::ImageComplete { index, result, .. } => {
                if index < self.images.len() {
                    self.images[index] = match result {
                        ProcessingResult::Success { .. } => ImageState::Success { /* ... */ },
                        ProcessingResult::Error { error_message } => ImageState::Error { message: error_message },
                    };
                }
            }
            ProcessingEvent::StageChange { stage, .. } => {
                self.current_stage = stage;
            }
            ProcessingEvent::Progress { completed, total, .. } => {
                self.overall_progress = completed as f32 / total as f32;
            }
        }
    }
}
```

**In beaker-gui update loop** (`beaker-gui/src/views/detection.rs`):

```rust
impl DetectionView {
    pub fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for progress events
        if let Some(ref rx) = self.progress_receiver {
            while let Ok(event) = rx.try_recv() {
                self.processing_state.update(event);
                ctx.request_repaint(); // Update UI immediately
            }
        }

        // Render UI based on processing_state
        // ...
    }
}
```

---

## Cancellation Support

Add cancellation using `Arc<AtomicBool>`:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub fn run_detection(
    config: DetectionConfig,
    progress_tx: Option<Sender<ProcessingEvent>>,
    cancel_flag: Option<Arc<AtomicBool>>,  // ← NEW
) -> Result<usize> {
    // ... processing loop ...

    for (index, image_path) in image_files.iter().enumerate() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Processing cancelled by user");
                return Ok(successful_count); // Return partial results
            }
        }

        // ... process image ...
    }
}
```

GUI cancel button:
```rust
if ui.button("Cancel").clicked() {
    self.cancel_flag.store(true, Ordering::Relaxed);
}
```

---

## Migration Timeline

### Week 1: Lib changes (80-120 LOC)
1. Add `ProcessingEvent` enum and related types (30 LOC)
2. Add optional `Sender<ProcessingEvent>` parameter to:
   - `run_model_processing_with_quality_outputs` (20 LOC changes)
   - `run_detection` (10 LOC changes)
3. Emit events at key points (40 LOC)
4. Add cancellation support with `Arc<AtomicBool>` (20 LOC)
5. Update CLI to pass `None` for new parameters (backward compatible!)

### Week 2: GUI integration (200-300 LOC)
1. Create `ProcessingState` struct (100 LOC)
2. Spawn background thread with channel (50 LOC)
3. Update UI on progress events (50 LOC)
4. Add progress UI components (100 LOC)

---

## Why This Approach Works

1. **Minimal lib changes**: <120 LOC, mostly additive
2. **Backward compatible**: CLI passes `None`, works unchanged
3. **Leverages existing infrastructure**: Uses existing error handling, progress tracking
4. **No async required**: Simple channels + threads (egui-friendly)
5. **Graceful degradation**: If channel send fails, processing continues
6. **Already battle-tested**: Error handling, two-pass flow already working

---

## Summary

**The beaker lib is already 80% ready for GUI integration!** We just need to:

✅ **Keep**: Graceful error handling, two-pass processing, strict/permissive modes
🔧 **Add**: Optional event channel (80 LOC)
🔧 **Add**: Cancellation flag (20 LOC)

Total lib work: **~100 LOC, <1 week** (not 1-2 weeks as initially estimated!)

---

## Feature Proposals

### 📁 Proposal 0: File Navigation & Opening (Critical Foundation)

**Goal:** Enable users to open images/folders from within the GUI using native file dialogs.

**Current limitation:** MVP requires CLI invocation with image path, which is:
- Not user-friendly for GUI app
- Fine for headless testing, but not for interactive use
- Missing standard desktop app UX

**Features:**

1. **Welcome screen** (when app launches without arguments)
   - Large, clear buttons: "Open Image", "Open Folder"
   - Recent files list (last 10 opened images/folders)
   - Drag & drop zone: "Drop image or folder here"
   - Getting started tips

2. **Native file dialogs**
   - File > Open Image: Native file picker for single image
   - File > Open Folder: Native folder picker
   - Filter by supported formats (.jpg, .png, .beaker.toml)
   - Remember last opened directory

3. **Drag & drop support**
   - Drag image file → open in single-image mode
   - Drag folder → open in bulk/directory mode
   - Drag multiple images → open first or show picker?
   - Visual feedback during drag (highlight drop zone)

4. **Recent files**
   - File > Recent menu with last 10 items
   - Show path and timestamp
   - Clear recent list option
   - Persist to disk (e.g., ~/.beaker-gui/recent.json)

5. **Native menu integration** (macOS/Windows/Linux)
   - File > Open Image (Cmd+O)
   - File > Open Folder (Cmd+Shift+O)
   - File > Recent >
   - File > Close (Cmd+W)

6. **Smart mode detection**
   - If folder contains .beaker.toml files → Bulk mode
   - If single image selected → Single-image mode
   - Show loading progress for large folders

**UI mockup (Welcome screen):**
```
┌────────────────────────────────────────────────────┐
│                  Beaker - Bird Analysis            │
│                                                     │
│         ┌───────────────────────────────┐          │
│         │                               │          │
│         │   Drop image or folder here   │          │
│         │                               │          │
│         │         📁 or 🖼️              │          │
│         │                               │          │
│         └───────────────────────────────┘          │
│                                                     │
│     [Open Image]        [Open Folder]              │
│                                                     │
│ ─────── Recent Files ──────────                    │
│  📁 /path/to/birds/           (2 hours ago)        │
│  🖼️  /path/to/bird_042.jpg   (yesterday)          │
│  📁 /path/to/dataset/         (3 days ago)         │
│                                                     │
│ 💡 Tip: Process a folder to triage quality across  │
│    multiple images, or open single image to inspect│
└────────────────────────────────────────────────────┘
```

**UI mockup (After opening folder):**
```
┌────────────────────────────────────────────────────┐
│ File  View  Help                    [macOS menu]   │
├────────────────────────────────────────────────────┤
│ 📁 /path/to/birds/ (47 images)     [Change Folder]│
│ ────────────────────────────────────────────────── │
│ [Gallery view with thumbnails...]                  │
│ ...                                                 │
└────────────────────────────────────────────────────┘
```

**Why critical:**
- **Fundamental UX**: Users expect to open files from GUI, not CLI
- **Discoverability**: New users can explore without knowing CLI flags
- **Workflow integration**: Fits standard desktop app patterns
- **Professionalism**: Apps without file dialogs feel incomplete

**Scope:**
- **Both modes**: Essential for single-image AND bulk modes
- **First-run experience**: Welcome screen is the user's introduction to the app

**Implementation notes:**
- Use `rfd` (Rust File Dialog) crate for native dialogs
- Welcome screen as default view when no args passed
- Store recent files in JSON at `~/.config/beaker-gui/recent.json` (Linux) or equivalent
- Drag & drop via egui's `dropped_files()` API
- ~400-500 LOC, 3-4 days work

**Dependencies:**
```toml
rfd = "0.15"  # Native file dialogs
serde_json = "1.0"  # Recent files persistence
dirs = "5.0"  # Cross-platform config directory
```

---

### 🎯 Proposal A: Bulk/Directory Mode Foundation (Essential)

**Goal:** Run detection on directories and manage results with aggregate views.

**Features:**

1. **Directory processing** (integrates with Proposal 0)
   - User selects folder via "Open Folder" dialog
   - GUI runs `beaker detect` on all images in directory
   - **Progress UI during processing:**
     - Overall progress: "Processing 23/47 images..."
     - Current stage: "Quality analysis..." or "Detecting..."
     - Per-image status list with checkmarks/errors (see UI mockup below)
     - Cancel button to stop gracefully

2. **Progress display** (critical for UX)
   - Live progress bar showing overall % complete
   - Image list showing status per image:
     - ⏳ bird_001.jpg: Processing...
     - ✓ bird_002.jpg: 2 detections (1 good, 1 unknown)
     - ⚠ bird_003.jpg: Error - file corrupt
   - Real-time updates as each image completes
   - Estimated time remaining

3. **Image gallery view** (after processing completes)
   - Thumbnail grid showing all processed images
   - Badge on each thumbnail: "2 detections (1 good, 1 unknown)"
   - Click thumbnail → open image in main view
   - Color-code thumbnails by worst quality (red if any bad detections)

4. **Aggregate detection list**
   - Sidebar shows ALL detections from ALL images
   - Format: "image1.jpg - head #1: Good (0.95)"
   - Click detection → jump to that image + zoom to detection

5. **Navigation**
   - Next/previous image buttons
   - Next/previous detection buttons
   - Keyboard: Arrow keys navigate images, J/K navigate detections

**UI mockup (During processing):**
```
┌─────────────────────────────────────────────────────────┐
│ Processing: /path/to/birds/ (47 images)                │
│ ──────────────────────────────────────────────────────  │
│ Stage: Detecting... (Quality complete)                  │
│ Progress: [▓▓▓▓▓▓▓▓▓▓░░░░░░] 23/47 (49%)              │
│ Estimated time: 2 minutes remaining      [Cancel]      │
│ ──────────────────────────────────────────────────────  │
│                                                          │
│ Images:                                                  │
│  ✓ bird_001.jpg: 2 detections (1 good, 1 unknown)      │
│  ✓ bird_002.jpg: 1 detection (1 good)                  │
│  ⚠ bird_003.jpg: Error - unsupported format            │
│  ✓ bird_004.jpg: 2 detections (2 good)                 │
│  ...                                                     │
│  ⏳ bird_023.jpg: Processing...            ← Current    │
│  ⏸ bird_024.jpg: Waiting...                            │
│  ⏸ bird_025.jpg: Waiting...                            │
│  ...                                                     │
│  ⏸ bird_047.jpg: Waiting...                            │
│                                                          │
│ 💡 Tip: This may take a few minutes for large folders  │
└─────────────────────────────────────────────────────────┘
```

**UI mockup (After processing complete):**
```
┌─────────────────────────────────────────────────────────┐
│ Gallery: /path/to/birds/ (47 images, 89 detections)    │
│ ──────────────────────────────────────────────────────  │
│ ┌──────┬──────┬──────┬──────┬──────┐                   │
│ │img1  │img2  │img3  │img4  │img5  │                   │
│ │ ✓1 ?1│ ✓2   │ ⚠Err │ ✓1 ?1│ ✓2   │  ← Badges        │
│ └──────┴──────┴──────┴──────┴──────┘                   │
│                                                          │
│ Current: img1.jpg           [← Prev | Next →]          │
│ ┌─────────────────────┬──────────────────────┐         │
│ │  [Image with boxes] │ All Detections (89)  │         │
│ │                     │ ──────────────────── │         │
│ │     [img with 1-2   │ img1.jpg - head #1   │         │
│ │      detections]    │   ✓ Good  Conf: 0.95│         │
│ │                     │ img1.jpg - head #2   │         │
│ │                     │   ? Unknown 0.52    │         │
│ │                     │ img2.jpg - head #1   │ ← Click │
│ │                     │   ✓ Good  Conf: 0.88│   jumps │
│ └─────────────────────┴──────────────────────┘   image │
└─────────────────────────────────────────────────────────┘
```

**Why essential:**
- Matches primary use case (directory processing)
- Enables all other bulk features (filtering, comparison, triage)
- Foundation for quality triage workflow
- **Actually runs detection** - not just a viewer

**Implementation notes:**
- New `DirectoryView` struct managing processing state
- Background thread for detection (don't block UI)
- Channel-based progress communication (lib → GUI)
- Store Vec<ImageDetectionState> with status per image
- Progress bar using egui::ProgressBar
- **Requires lib changes**: Add progress callback/channel to `run_detection()`
- **Requires error handling**: Single image failures shouldn't crash whole run
- ~800-1000 LOC, 2 weeks work (including lib changes)

**Dependencies:**
- Lib changes: Add progress API to beaker (1 week)
- GUI implementation: Build progress UI + integration (1 week)

---

### 🔥 Proposal B: Quality Triage Workflow (High Priority)

**Goal:** Rapidly find and review quality detections across entire directory.

**Features:**

1. **Aggregate quality filtering**
   - Filter across ALL images: "Show only Good detections"
   - Checkboxes: ☑ Good  ☑ Unknown  ☐ Bad
   - Show count: "Good (45) Unknown (32) Bad (12)"
   - Filter updates both sidebar and gallery view

2. **Triage mode**
   - Special view mode: "Review all Unknown detections"
   - Show images one-by-one with unknown detections highlighted
   - Quick actions: Mark as "Actually Good" or "Actually Bad"
   - Keyboard shortcuts: G (good), B (bad), U (skip/keep unknown), → (next)

3. **Quality statistics panel**
   - Summary across directory:
     - "45/89 detections are Good (50.5%)"
     - "12/89 are Bad (13.5%)"
     - "32/89 need review (36.0%)"
   - Avg confidence per quality tier
   - Distribution histograms

4. **Sort by quality metrics**
   - Sort dropdown: "Worst blur first", "Best confidence first", "Smallest coverage first"
   - Helps prioritize review (start with worst detections)

5. **Quality trends**
   - Show quality metrics across directory
   - Identify problematic images: "image_042.jpg has 2 bad detections"
   - Chart: quality score vs image index (spot patterns)

**UI mockup (Triage mode):**
```
┌──────────────────────────────────────────────────────┐
│ 🔍 Triage Mode: Reviewing Unknown Detections        │
│ Progress: 8/32 reviewed                [Exit Triage] │
│ ─────────────────────────────────────────────────── │
│                                                       │
│      [Large image: img_023.jpg]                      │
│      [Detection highlighted with bbox]               │
│                                                       │
│ Detection: head #1                                   │
│ Current triage: Unknown (?)                          │
│ Rationale: "borderline sharpness region held out     │
│            core_ring_sharpness_ratio=1.21"           │
│                                                       │
│ Quality Metrics:                                     │
│   Blur probability: 0.42   ▓▓▓░░ (borderline)       │
│   Core/Ring sharp:  1.21   ▓▓░░░ (low)              │
│   Coverage:         5.8    ▓▓▓░░ (moderate)         │
│                                                       │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Mark as:  [G] Good  [B] Bad  [U] Keep Unknown  │ │
│ │           [→] Next  [←] Previous               │ │
│ └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Why high priority:**
- Directly addresses main use case: triaging directories
- Leverages beaker's rich quality data
- Saves massive time vs CLI workflow
- Educational: helps users learn quality patterns

**Implementation notes:**
- Triage mode as separate view state
- Store user overrides: HashMap<detection_id, UserTriageOverride>
- Export overrides to separate JSON file for training data
- ~400 LOC, 1 week work

---

### 🎨 Proposal C: Quality Heatmap Visualization (High Value)

**Goal:** Surface beaker's debug heatmaps in GUI for visual quality understanding.

**Features:**

1. **Heatmap layer selector**
   - Dropdown/tabs: "Blur Probability", "Tenengrad", "Fused Blur", "Weights", "None"
   - Toggle overlay on/off
   - Adjust opacity slider (0-100%)

2. **Auto-generate heatmaps**
   - Run quality analysis with `--debug-dump-images` equivalent
   - Load generated heatmap overlays
   - Cache for performance

3. **Heatmap gallery**
   - Side-by-side view: Original | Blur Prob | Tenengrad | Fused
   - Click to toggle between views
   - Useful for comparing different quality metrics

4. **ROI highlighting**
   - Overlay detection ROI on heatmap
   - Show sampled region used for quality calculation
   - Animate ROI pooling area

5. **Multi-scale visualization**
   - Toggle between 224 and 112 scale heatmaps (t224 vs t112, p224 vs p112)
   - Show scale difference side-by-side

**UI mockup:**
```
┌────────────────────────────────────────────────────┐
│ Heatmap View: img_007.jpg                          │
│ Layer: [Blur Probability ▾]  Opacity: [▓▓▓▓░] 70%│
│ ────────────────────────────────────────────────── │
│  ┌──────────────────────────────────────────┐     │
│  │                                           │     │
│  │   [Image with blur heatmap overlay]      │     │
│  │   Blue = sharp, Red = blurry             │     │
│  │                                           │     │
│  │   ┌──────┐ ← Detection ROI highlighted   │     │
│  │   │░░Red░│   (high blur in this region!) │     │
│  │   └──────┘                                │     │
│  │                                           │     │
│  └──────────────────────────────────────────┘     │
│                                                     │
│ Available Heatmaps:                                │
│  [Original] [Blur Prob] [Tenengrad] [Fused Blur]  │
│                                                     │
│ ROI Stats:                                         │
│  Mean blur prob in ROI: 0.65 (high!)               │
│  This explains the "bad" triage decision           │
└────────────────────────────────────────────────────┘
```

**Why high value:**
- Makes `--debug-dump-images` output accessible
- Visual learning tool: see *why* detection is good/bad
- Debugging: verify quality calculations are correct
- Unique feature leveraging beaker's internals

**Implementation notes:**
- Extend beaker lib to expose heatmap generation API (or re-run quality with debug flag)
- Load overlay images as textures
- Use egui's Image widget with multiple layers
- May need to run quality analysis on-demand (performance consideration)
- ~600 LOC, 1-2 weeks work

---

### 📊 Proposal D: Rich Quality Metrics Display (Single Image)

**Goal:** Deep-dive into quality metrics for individual detections (complements Proposal C).

**Features:**

1. **Expanded detection cards**
   - Collapsible sections showing all 12 quality metrics
   - Color-coded triage badges (green=good, yellow=unknown, red=bad)
   - Show triage rationale prominently

2. **Metric visualizations**
   - Progress bars for normalized metrics (0-1 range)
   - Gauges for ratios (core/ring sharpness)
   - Icons/symbols for intuitive reading

3. **Contextual explanations**
   - Tooltip on each metric explaining what it measures
   - Links to docs for deep dives
   - "What does this mean?" helper text

4. **Metric comparison** (when 2 detections in image)
   - Side-by-side metric comparison
   - Highlight differences (which has better coverage?)
   - Explain which metrics matter most for triage decision

**UI mockup:**
```
┌─────────────────────────────────────────────────────┐
│ Detection Details: head #1                          │
│ ──────────────────────────────────────────────────  │
│                                                      │
│ Triage: [✓ Good] ←───────────────┐                 │
│ Reason: "Sharp enough and well   │                 │
│         covered with margin"     │                 │
│         (click to see why) ──────┘                 │
│                                                      │
│ ┌─ Core Metrics ─────────────────────────────────┐ │
│ │ Confidence:        0.95  ▓▓▓▓▓ Excellent       │ │
│ │ Blur Probability:  0.12  ░░▓▓▓ Low (sharp!)    │ │
│ │ Detail Probability: 0.89  ▓▓▓▓▓ High detail    │ │
│ │ Core/Ring Sharp:   2.41  ▓▓▓▓░ Good focus      │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ ┌─ Advanced Metrics ▼ ────────────────────────────┐ │
│ │ ROI Quality Mean:     67.3  ▓▓▓▓░              │ │
│ │ Size Prior:           0.92  ▓▓▓▓▓              │ │
│ │ Grid Coverage:        8.3 cells                │ │
│ │ Tenengrad Core:       0.045                    │ │
│ │ Tenengrad Ring:       0.018                    │ │
│ │ ...                                            │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ 💡 Why is this "Good"?                              │
│ • Core/ring sharpness (2.41) > threshold (1.59)    │
│ • Coverage (8.3 cells) > threshold (6.15)          │
│ • Both criteria met with safety margin             │
└─────────────────────────────────────────────────────┘
```

**Why valuable:**
- Educational: helps users understand quality system
- Debugging: verify metric calculations
- Trust: explain automated decisions
- Works for both single-image and bulk modes

**Scope:**
- **Single Image Mode**: Deep dive on 1-2 detections
- **Bulk Mode**: Click detection in list → show metrics panel

**Implementation notes:**
- Extend GUI Detection struct to include all quality fields
- Parse from TOML metadata
- Use egui::Grid, ProgressBar, CollapsingHeader widgets
- Write metric explanation strings (maybe load from separate file)
- ~400 LOC, 3-4 days work

---

### 🔍 Proposal E: Zoom & Pan (Essential for Hi-Res Images)

**Goal:** Navigate high-resolution images effectively to inspect detection details.

**Features:**

1. **Zoom controls**
   - Mouse wheel to zoom in/out
   - Zoom slider in toolbar
   - Buttons: Fit-to-window, 100%, 200%
   - Zoom to detection: click detection → zoom and center on it

2. **Pan**
   - Click-and-drag to pan when zoomed
   - Smooth panning with momentum (optional)

3. **Mini-map**
   - Small overview showing full image
   - Highlight current viewport
   - Click mini-map to jump to region

4. **Zoom-synchronized heatmap**
   - When zoomed in, heatmap overlay stays aligned
   - Important for seeing quality at pixel level

**UI mockup:**
```
┌────────────────────────────────────────────────────┐
│ Zoom: [- ◼ +] [Fit] [100%] [200%]                 │
│ ────────────────────────────────────────────────── │
│  ┌─────────────────────────┬──────────┐           │
│  │ [Zoomed image 200%]     │ Mini-map │           │
│  │                         │ ┌──────┐ │           │
│  │  ▓▓▓▓▓▓▓                │ │  ░░  │ │           │
│  │  ▓head ▓  0.95          │ │  ▓▓  │ │← viewport│
│  │  ▓▓▓▓▓▓▓                │ └──────┘ │           │
│  │  (zoomed to detection)  │          │           │
│  │                         │          │           │
│  │  (pan with click-drag)  │          │           │
│  └─────────────────────────┴──────────┘           │
└────────────────────────────────────────────────────┘
```

**Why essential:**
- Hi-res images (4K+) need zoom to see detection details
- "Zoom to detection" dramatically improves workflow
- Standard feature in any image viewer
- Critical for inspecting blur/sharpness at pixel level

**Scope:**
- **Single Image Mode**: Zoom into 1-2 detections
- **Bulk Mode**: Zoom into current image while navigating directory

**Implementation notes:**
- Track zoom level, pan offset in view state
- Transform mouse coords for hit-testing
- Use egui::Image with custom pan/zoom transform
- Mini-map as separate small Image widget
- ~400 LOC, 3-4 days work

---

### ⚡ Proposal F: Keyboard Shortcuts & Power User Features (Polish)

**Goal:** Make directory triage fast and efficient for power users.

**Features:**

1. **Image navigation**
   - `→` / `←`: Next/previous image in directory
   - `J` / `K`: Next/previous detection across all images
   - `Home` / `End`: First/last image

2. **Quality filtering shortcuts**
   - `G`: Show only Good detections
   - `B`: Show only Bad detections
   - `U`: Show only Unknown detections
   - `A`: Show All detections (clear filter)

3. **Triage shortcuts** (in triage mode)
   - `G`: Mark current detection as Good
   - `B`: Mark as Bad
   - `U`: Keep as Unknown
   - `→`: Next detection
   - `Esc`: Exit triage mode

4. **View shortcuts**
   - `Z`: Toggle zoom-to-detection
   - `H`: Toggle heatmap overlay
   - `1-6`: Switch heatmap layer (1=blur, 2=tenengrad, etc.)
   - `Space`: Toggle detection selection

5. **Command palette**
   - `Ctrl+K` / `Cmd+K`: Open command palette
   - Fuzzy search for actions: "show blur heatmap", "triage unknown", etc.

6. **Status bar**
   - Show current image, detection count, filter status
   - Keyboard hint: "Press → for next image"

**UI mockup:**
```
┌────────────────────────────────────────────────────┐
│ [Cmd+K Command Palette]                            │
│ ┌─────────────────────────────────────────────────┐│
│ │ > blur_________________________________        ││
│ │   Show Blur Heatmap (H)                        ││
│ │   Filter High Blur Detections                  ││
│ │   Sort by Blur Score ↓                         ││
│ └─────────────────────────────────────────────────┘│
│                                                     │
│ Status: img_023.jpg (23/47) | 2 detections | 1 Good│
│ Tip: Press G to show only Good detections         │
└────────────────────────────────────────────────────┘
```

**Why polish:**
- 10x productivity for directory triage (process 100 images in minutes)
- Modern UX pattern (VSCode, Obsidian, etc.)
- Low cost, high perceived quality
- Makes bulk workflow feel professional

**Scope:**
- **Bulk Mode**: Essential for efficiency
- **Single Image Mode**: Less critical but nice-to-have

**Implementation notes:**
- Use egui's input handling for key events
- Store shortcuts in registry/hashmap
- Command palette as modal overlay with fuzzy matching
- ~300 LOC, 2-3 days work

---

### 📤 Proposal G: Export & Reporting (Power User)

**Goal:** Extract triage results for further analysis or training data.

**Features:**

1. **Export filtered detections**
   - Export current filtered set as JSON/CSV
   - Include selected quality metrics
   - Option: export only Good, only Bad, or custom filter

2. **Export triage overrides**
   - Save user's manual triage decisions (from Proposal B)
   - Format: `{"detection_id": "img_023_head_1", "user_triage": "good", "original_triage": "unknown"}`
   - Use for model training/refinement

3. **Generate summary report**
   - Markdown or HTML report
   - Statistics: quality distribution, worst images, etc.
   - Include thumbnails of bad detections
   - Export chart images

4. **Batch export crops**
   - Export cropped images for all Good detections
   - Useful for creating datasets

5. **Copy metrics to clipboard**
   - Right-click detection → "Copy metrics as JSON"
   - Quick data extraction

**UI mockup:**
```
┌────────────────────────────────────────────────────┐
│ Export                                [X]          │
│ ────────────────────────────────────────────────── │
│                                                     │
│ What to export?                                    │
│  ● All detections (89)                             │
│  ○ Filtered detections (45 Good)                   │
│  ○ Selected detections (3)                         │
│                                                     │
│ Format:                                            │
│  ● JSON  ○ CSV  ○ Summary Report (HTML)           │
│                                                     │
│ Include:                                           │
│  ☑ Image paths                                     │
│  ☑ Quality metrics (all 12)                        │
│  ☑ Triage decisions                                │
│  ☑ User overrides                                  │
│  ☐ Cropped images                                  │
│                                                     │
│ Output path:                                       │
│  [/path/to/export.json          ] [Browse]        │
│                                                     │
│              [Cancel]  [Export]                    │
└────────────────────────────────────────────────────┘
```

**Why power user:**
- Enables research workflows
- Integration with external tools (ML training, etc.)
- Document quality assessment results
- Share findings with team

**Scope:**
- **Bulk Mode**: Essential for processing many images
- **Single Image Mode**: Less useful

**Implementation notes:**
- Use serde_json, csv crate for export
- Report generation with simple HTML templating
- ~300 LOC, 2-3 days work

---

### 🔄 Proposal H: Comparison View (Specialized)

**Goal:** Compare detections side-by-side to understand quality differences.

**Use cases:**
1. **Within image**: Compare 2 detections in same image
2. **Across images**: Compare same detection across different images (e.g., before/after processing)
3. **Training**: Understand borderline cases (why is one Good and another Bad?)

**Features:**

1. **Side-by-side layout**
   - Select 2-4 detections → open comparison view
   - Show crops (if available) or zoomed regions
   - Align at same zoom level

2. **Metric diff**
   - Highlight differing metrics
   - Color-code: green=better, red=worse
   - Show deltas (e.g., "Blur: 0.12 vs 0.45 (Δ +0.33)")

3. **Heatmap comparison**
   - Show heatmaps side-by-side
   - Same colormap scale for fair comparison

4. **Triage comparison**
   - Show why one is Good and other is Bad
   - Highlight threshold crossings

**UI mockup:**
```
┌────────────────────────────────────────────────────┐
│ Comparison: 2 detections             [Close]      │
│ ────────────────────────────────────────────────── │
│                                                     │
│ ┌──────────────────────┬──────────────────────┐   │
│ │ img_023.jpg - head#1 │ img_041.jpg - head#1 │   │
│ │ ┌──────────────────┐ │ ┌──────────────────┐ │   │
│ │ │  [Crop/zoom]     │ │ │  [Crop/zoom]     │ │   │
│ │ │  Sharp, clear    │ │ │  Blurry, soft    │ │   │
│ │ └──────────────────┘ │ └──────────────────┘ │   │
│ │                      │                      │   │
│ │ Triage: ✓ Good      │ Triage: ✗ Bad       │   │
│ │                      │                      │   │
│ │ Confidence: 0.95    │ Confidence: 0.52 ▼  │   │
│ │ Blur Prob:  0.12    │ Blur Prob:  0.78 ▲  │   │
│ │ Core/Ring:  2.41    │ Core/Ring:  0.95 ▼  │   │
│ │ Coverage:   8.3     │ Coverage:   3.2  ▼  │   │
│ │                      │                      │   │
│ │ [Show Heatmap]      │ [Show Heatmap]      │   │
│ └──────────────────────┴──────────────────────┘   │
│                                                     │
│ 📊 Key Differences:                                │
│ • Blur probability: 0.12 vs 0.78 (Δ +0.66) ← Main!│
│ • Core/ring sharpness: 2.41 vs 0.95 (Δ -1.46)     │
│ → Left detection is sharper and better focused    │
└────────────────────────────────────────────────────┘
```

**Why specialized:**
- Educational: understand quality boundaries
- Debugging: verify triage logic
- Research: analyze failure modes
- Not needed for basic triage workflow

**Scope:**
- Works in both Single Image (compare 2 dets) and Bulk Mode (compare across images)

**Implementation notes:**
- New ComparisonView struct
- Select detections from list, enter comparison mode
- Layout with egui::Grid (2-4 columns)
- Highlight diffs with color coding
- ~400 LOC, 3-4 days work

---

## Recommended Implementation Roadmap

### Phase 0: Foundation & File Navigation (1 week)
**Goal:** Enable basic app launch and file/folder opening

1. **Proposal 0: File Navigation & Opening** (critical)
   - Welcome screen
   - Native file dialogs (Open Image, Open Folder)
   - Drag & drop support
   - Recent files list

**Deliverable:** Launch app, see welcome screen, open folder via dialog → load images.

---

### Phase 1: Bulk Foundation + Lib Changes (3-4 weeks)
**Goal:** Make directory processing functional with progress reporting

**Week 1-2: Lib changes (beaker crate)**
- Add progress callback/channel API to `run_detection()`
- Graceful per-image error handling
- Cancellation support

**Week 3-4: GUI implementation**

2. **Proposal A: Bulk/Directory Mode** (essential)
   - Directory processing with progress UI
   - Channel-based communication with lib
   - Per-image status tracking
   - Error display
   - Cancel functionality
   - Image gallery (after processing)
   - Aggregate detection list
   - Navigation

3. **Proposal E: Zoom & Pan** (essential)
   - Basic zoom in/out
   - Pan when zoomed
   - Zoom to detection

**Deliverable:** Open folder via dialog → watch progress bar as detection runs → see gallery of 50 processed images with checkmarks/errors → navigate between images → zoom to inspect detections.

---

### Phase 2: Quality Triage (3-4 weeks)
**Goal:** Leverage quality data for efficient triage

4. **Proposal B: Quality Triage Workflow** (high priority)
   - Aggregate filtering (good/bad/unknown)
   - Triage mode for reviewing unknowns
   - Quality statistics

5. **Proposal C: Quality Heatmap Visualization** (high value)
   - Load debug heatmaps
   - Layer selector
   - Opacity control

6. **Proposal D: Rich Quality Metrics** (complements C)
   - Expanded metric cards
   - Triage explanations
   - Contextual help

**Deliverable:** Open folder of 100 images via dialog, filter to 30 unknowns, review them with heatmaps, export 80 good detections.

---

### Phase 3: Power Features (2-3 weeks)
**Goal:** Polish and power-user workflows

7. **Proposal F: Keyboard Shortcuts** (polish)
   - Navigation shortcuts
   - Triage shortcuts
   - Command palette

8. **Proposal G: Export & Reporting** (power user)
   - Export JSON/CSV
   - Triage overrides
   - Summary reports

9. **Proposal H: Comparison View** (optional)
   - Side-by-side comparison
   - Metric diffs

**Deliverable:** Triage 500 images in under an hour, export results for training pipeline.

---

## Alternative: Phased by Workflow

### Workflow 1: Basic App & Quick Triage (Week 1-3)
- File navigation (0)
- Directory loading (A)
- Quality filtering (B partial)
- Zoom to detection (E partial)
→ **Launch app, open folder, triage 50 images, find good detections**

### Workflow 2: Deep Quality Understanding (Week 4-6)
- Heatmap visualization (C)
- Rich metrics display (D)
- Triage mode (B complete)
→ **Understand *why* detections are good/bad**

### Workflow 3: Batch Processing (Week 7-8)
- Keyboard shortcuts (F)
- Export (G)
- Comparison (H)
→ **Process 100+ images efficiently**

---

## Feature Scope Summary

| Proposal | Single Image | Bulk Mode | Priority | Effort |
|----------|--------------|-----------|----------|--------|
| 0: File Navigation | ★★★ Essential | ★★★ Essential | **Critical** | 3-4 days |
| A: Bulk/Directory Mode | N/A | ★★★ Essential | **Critical** | 1 week |
| B: Quality Triage | ★ Nice | ★★★ Essential | **High** | 1 week |
| C: Heatmap Viz | ★★ Useful | ★★★ High value | **High** | 1-2 weeks |
| D: Rich Metrics | ★★★ Essential | ★★ Useful | Medium | 3-4 days |
| E: Zoom & Pan | ★★★ Essential | ★★★ Essential | **Critical** | 3-4 days |
| F: Shortcuts | ★ Nice | ★★★ Essential | Medium | 2-3 days |
| G: Export | ★ Low value | ★★★ Essential | Medium | 2-3 days |
| H: Comparison | ★★ Useful | ★ Specialized | Low | 3-4 days |

**Legend:**
- ★★★ = Essential/High value
- ★★ = Useful
- ★ = Nice-to-have/Specialized

---

## Open Questions

1. **Heatmap generation strategy**
   - **Question:** Should debug heatmaps be generated on-demand when user toggles them on, or always generated during initial detection run?
   - **Trade-off:**
     - On-demand: Faster initial processing, but delay when enabling heatmaps
     - Always generate: Slower initial processing, but instant heatmap display
   - **Recommendation:** Generate on-demand with caching. Most users won't use heatmaps, so don't slow down the common case.
   - **Implementation:** Add `--with-debug-heatmaps` checkbox in GUI; when checked, pass flag to beaker lib

2. **Triage override persistence**
   - **Question:** When user manually marks an "unknown" detection as "good" or "bad" in triage mode, where should this override be saved?
   - **Options:**
     - a) Separate `{stem}_triage_overrides.json` file alongside .beaker.toml
     - b) Extend .beaker.toml with new `[triage_overrides]` section
     - c) New `{stem}.beaker_triage.json` file
   - **Recommendation:** Option (c) - keeps triage data separate from automated analysis results, easy to export for ML training

3. **Directory watching (future enhancement)**
   - **Question:** Should GUI auto-reload when new images are added to watched directory?
   - **Use case:** Live processing workflows where images are added during session
   - **Complexity:** Requires file system watcher (e.g., `notify` crate)
   - **Recommendation:** Defer to Phase 4+ (not MVP). Add manual "Refresh" button first.

4. **Multi-image aggregate metrics (future enhancement)**
   - **Question:** What aggregate statistics should be shown across all images in directory?
   - **Ideas:**
     - "50% of images have at least 1 bad detection"
     - Image-level quality score (worst detection quality per image)
     - Quality distribution histogram across entire directory
     - "Images with errors" count
   - **Recommendation:** Include in Proposal B (Quality Triage Workflow) - this is valuable for understanding overall directory quality

---

## Testing Strategy

### Unit Tests
- Directory loading/parsing logic
- Filtering across multiple images
- Quality metric aggregation
- Export format correctness

### Integration Tests
- Load directory with 10 test images
- Filter to Good detections
- Navigate through images
- Export results

### Manual Test Scenarios
1. **Directory triage**: Load 50 images, filter to 10 good detections, zoom to each
2. **Heatmap workflow**: Enable blur heatmap, review all bad detections
3. **Keyboard workflow**: Use only keyboard to triage 20 images
4. **Export workflow**: Filter and export, verify JSON format

### Performance Tests
- **Process** directory with 100+ images (will take several minutes - ~1-2s per image for quality+detection)
  - Progress UI should update smoothly (at least 10 FPS)
  - UI should remain responsive during processing
- Navigate between processed images (should be instant, <16ms)
- Filter 200 detections across images (should be <100ms)
- Render heatmap overlay (should be <50ms for texture update)
- Handle cancellation cleanly (stop within 1-2 seconds of button press)

---

## Technical Architecture Notes

### Directory Mode Data Structures
```rust
struct DirectoryView {
    directory_path: PathBuf,
    images: Vec<ImageState>,
    current_image_idx: usize,
    filter_state: FilterState,
    // Processing state
    processing_receiver: Option<Receiver<ProcessingEvent>>,
    cancel_flag: Arc<AtomicBool>,
    // Flattened list of all detections across images (populated after processing)
    all_detections: Vec<DetectionRef>,
}

struct ImageState {
    image_path: PathBuf,
    status: ProcessingStatus,
    detections: Vec<Detection>,  // Populated after successful processing
    thumbnail: Option<TextureHandle>,
}

enum ProcessingStatus {
    Waiting,
    Processing,
    Success {
        toml_path: PathBuf,  // Generated .beaker.toml file
        processing_time_ms: f64,
    },
    Error {
        message: String,
    },
}

struct DetectionRef {
    image_idx: usize,
    detection_idx: usize,
    // Cached quality for sorting/filtering (from detection result)
    quality: DetectionQuality,
}

struct FilterState {
    show_good: bool,
    show_unknown: bool,
    show_bad: bool,
    min_confidence: f32,
    sort_by: SortMode,
}
```

### Heatmap Integration

**Recommended approach: On-demand generation with caching**

```rust
struct HeatmapCache {
    // Cache heatmaps in memory after generation
    cache: HashMap<PathBuf, DebugHeatmaps>,
}

impl HeatmapCache {
    fn get_or_generate(&mut self, image_path: &Path) -> Result<&DebugHeatmaps> {
        if !self.cache.contains_key(image_path) {
            // Generate heatmaps on first access
            let heatmaps = self.generate_heatmaps(image_path)?;
            self.cache.insert(image_path.to_path_buf(), heatmaps);
        }
        Ok(&self.cache[image_path])
    }

    fn generate_heatmaps(&self, image_path: &Path) -> Result<DebugHeatmaps> {
        // Check if debug images directory already exists
        let stem = image_path.file_stem().unwrap();
        let debug_dir = image_path.parent().unwrap()
            .join(format!("quality_debug_images_{}", stem.to_string_lossy()));

        if debug_dir.exists() {
            // Load pre-existing debug images
            load_debug_heatmaps_from_dir(&debug_dir)
        } else {
            // Generate on-demand by re-running quality with --debug-dump-images
            // (This only happens if user toggles heatmap view)
            run_quality_with_debug_flag(image_path)?;
            load_debug_heatmaps_from_dir(&debug_dir)
        }
    }
}
```

**Why on-demand:**
- Most users won't use heatmaps (they're for debugging/learning)
- Don't slow down initial processing for rare feature
- Can still load pre-existing debug images if available
- Cache in memory after first generation (instant subsequent access)

---

## Summary & Recommendation

**Recommended MVP+1 scope** (8-11 weeks):

**Foundation (Critical - Week 1):**
1. **Proposal 0** - File Navigation & Opening

**Core (Critical - Weeks 2-5):**
- **Weeks 2-3**: Lib changes for progress reporting
- **Weeks 4-5**: GUI implementation
2. **Proposal A** - Bulk/Directory Mode (with progress UI)
3. **Proposal E** - Zoom & Pan

**Quality Triage (High Priority - Weeks 6-9):**
4. **Proposal B** - Quality Triage Workflow
5. **Proposal C** - Heatmap Visualization

**Polish (Important - Weeks 10-11):**
6. **Proposal D** - Rich Metrics (simplified)
7. **Proposal F** - Keyboard Shortcuts (partial)

This delivers a **production-ready quality triage tool** that:
- ✅ **Launches like a real desktop app** (not just CLI testing tool)
- ✅ **Runs detection from GUI** with progress feedback
- ✅ Native file dialogs, drag & drop, recent files
- ✅ Per-image status tracking (checkmarks, errors)
- ✅ Graceful error handling (single failures don't crash whole run)
- ✅ Handles real directory-processing workflows (10-100+ images)
- ✅ Surfaces beaker's unique quality data and visualizations
- ✅ Makes heatmaps accessible (currently hidden in debug output)
- ✅ Dramatically faster than CLI workflow
- ✅ Feels professional and polished

**Want maximum features?** Add Proposal G (Export) and H (Comparison) for a **10-13 week super-app**.

**Want minimal but functional?** Just 0 + A + E (6 weeks including lib changes) for a **basic directory processor with progress UI**.

**Unique value proposition:** No other tool makes quality heatmaps this accessible. This could be a killer feature for beaker-gui.
