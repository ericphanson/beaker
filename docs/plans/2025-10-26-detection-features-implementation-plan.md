# Detection Features Implementation Plan

For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build GUI detection features enabling users to process directories, view progress, navigate results, and triage quality across multiple images.

**Architecture:** Extend beaker lib with optional progress callbacks for GUI communication. Add file dialogs, welcome screen, and directory processing workflow to beaker-gui. Use channels for background thread communication, maintain processing state in GUI.

**Tech Stack:**
- beaker lib: std::sync::mpsc channels, Arc<AtomicBool> for cancellation
- beaker-gui: egui, rfd (file dialogs), serde_json (recent files), dirs (config paths)

---

## Part 1: Lib Changes for Progress Reporting

These changes add optional progress callback support to beaker lib, enabling GUI to receive real-time updates during processing.

---

### Task 1: Add ProcessingEvent Types

**Files:**
- Create: beaker/src/processing_events.rs
- Modify: beaker/src/lib.rs:1-50 (add module export)

**Step 1: Write the failing test**

Create test file:
```rust
// beaker/tests/processing_events_test.rs
use beaker::{ProcessingEvent, ProcessingStage, ProcessingResult};
use std::path::PathBuf;

#[test]
fn test_processing_event_creation() {
    let event = ProcessingEvent::ImageStart {
        path: PathBuf::from("/test/image.jpg"),
        index: 0,
        total: 10,
        stage: ProcessingStage::Quality,
    };

    match event {
        ProcessingEvent::ImageStart { index, total, .. } => {
            assert_eq!(index, 0);
            assert_eq!(total, 10);
        }
        _ => panic!("Wrong event type"),
    }
}

#[test]
fn test_processing_result_success() {
    let result = ProcessingResult::Success {
        detections_count: 2,
        good_count: 1,
        bad_count: 0,
        unknown_count: 1,
        processing_time_ms: 123.45,
    };

    match result {
        ProcessingResult::Success { detections_count, .. } => {
            assert_eq!(detections_count, 2);
        }
        _ => panic!("Wrong result type"),
    }
}
```

**Step 2: Run test to verify it fails**

Run: `just test`

Expected: FAIL with "no `ProcessingEvent` in the root" or similar

**Step 3: Write minimal implementation**

```rust
// beaker/src/processing_events.rs
use std::path::PathBuf;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

Add to beaker/src/lib.rs:
```rust
// After existing module declarations
pub mod processing_events;

pub use processing_events::{ProcessingEvent, ProcessingStage, ProcessingResult};
```

**Step 4: Run test to verify it passes**

Run: `just test`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker/src/processing_events.rs beaker/src/lib.rs beaker/tests/processing_events_test.rs
git commit -m "feat: add ProcessingEvent types for GUI progress reporting"
```

---

### Task 2: Add Progress Channel Parameter to Quality Processing

**Files:**
- Modify: beaker/src/model_processing.rs:200-250 (run_model_processing_with_quality_outputs signature)
- Modify: beaker/src/quality.rs:50-70 (run_quality call site)
- Test: beaker/tests/quality_progress_test.rs

**Step 1: Write the failing test**

```rust
// beaker/tests/quality_progress_test.rs
use beaker::{QualityConfig, ProcessingEvent, ProcessingStage};
use std::sync::mpsc::channel;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_quality_processing_emits_events() {
    let temp_dir = TempDir::new().unwrap();
    let test_image = temp_dir.path().join("test.jpg");

    // Create a minimal test image (copy from examples)
    std::fs::copy("example.jpg", &test_image).unwrap();

    let (tx, rx) = channel();

    let config = QualityConfig {
        input: test_image.to_str().unwrap().to_string(),
        ..Default::default()
    };

    // This will fail because progress_tx parameter doesn't exist yet
    beaker::run_quality_with_progress(config, Some(tx)).unwrap();

    // Verify we received events
    let events: Vec<ProcessingEvent> = rx.try_iter().collect();
    assert!(!events.is_empty(), "Should receive progress events");

    // Should have at least ImageStart and ImageComplete
    let has_start = events.iter().any(|e| matches!(e, ProcessingEvent::ImageStart { .. }));
    let has_complete = events.iter().any(|e| matches!(e, ProcessingEvent::ImageComplete { .. }));

    assert!(has_start, "Should have ImageStart event");
    assert!(has_complete, "Should have ImageComplete event");
}
```

**Step 2: Run test to verify it fails**

Run: `just test`

Expected: FAIL with "cannot find function `run_quality_with_progress`"

**Step 3: Write minimal implementation**

In beaker/src/model_processing.rs, modify the signature:

```rust
pub fn run_model_processing_with_quality_outputs<P: ModelProcessor>(
    config: P::Config,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
) -> anyhow::Result<(usize, std::collections::HashMap<String, crate::quality::QualityResult>)> {
    // ... existing setup code ...

    let image_files = get_image_files(config.base())?;
    let total = image_files.len();

    // Determine current stage based on processor type
    let stage = if std::any::type_name::<P>().contains("QualityProcessor") {
        crate::ProcessingStage::Quality
    } else {
        crate::ProcessingStage::Detection
    };

    // ... existing progress bar setup ...

    for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
        // Emit start event
        if let Some(ref tx) = progress_tx {
            let _ = tx.send(crate::ProcessingEvent::ImageStart {
                path: image_path.clone(),
                index,
                total,
                stage,
            });
        }

        let start_time = std::time::Instant::now();

        match P::process_single_image(&mut session, image_path, &config, &output_manager) {
            Ok(result) => {
                successful_count += 1;

                // Extract quality info for success event
                let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

                // Emit success event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(crate::ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: crate::ProcessingResult::Success {
                            detections_count: 0, // Quality doesn't have detections
                            good_count: 0,
                            bad_count: 0,
                            unknown_count: 0,
                            processing_time_ms,
                        },
                    });
                }

                // ... existing success handling ...
            }
            Err(e) => {
                failed_count += 1;
                failed_images.push(image_path.to_str().unwrap_or_default().to_string());

                // Emit error event
                if let Some(ref tx) = progress_tx {
                    let _ = tx.send(crate::ProcessingEvent::ImageComplete {
                        path: image_path.clone(),
                        index,
                        result: crate::ProcessingResult::Error {
                            error_message: e.to_string(),
                        },
                    });
                }

                log::warn!("Failed to process {}:\n {}", image_path.display(), e);
            }
        }

        // ... existing progress bar update ...
    }

    // ... rest of function unchanged ...
}
```

Add wrapper function in beaker/src/quality.rs:

```rust
pub fn run_quality_with_progress(
    config: QualityConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
) -> anyhow::Result<usize> {
    let (count, _) = crate::model_processing::run_model_processing_with_quality_outputs::<QualityProcessor>(
        config,
        progress_tx,
    )?;
    Ok(count)
}
```

Update existing run_quality to pass None:

```rust
pub fn run_quality(config: QualityConfig) -> anyhow::Result<usize> {
    run_quality_with_progress(config, None)
}
```

**Step 4: Run test to verify it passes**

Run: `just test`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker/src/model_processing.rs beaker/src/quality.rs beaker/tests/quality_progress_test.rs
git commit -m "feat: add progress channel support to quality processing"
```

---

### Task 3: Add Progress Channel Parameter to Detection Processing

**Files:**
- Modify: beaker/src/detection.rs:95-120 (run_detection function)
- Test: beaker/tests/detection_progress_test.rs

**Step 1: Write the failing test**

```rust
// beaker/tests/detection_progress_test.rs
use beaker::{DetectionConfig, ProcessingEvent, ProcessingStage};
use std::sync::mpsc::channel;
use tempfile::TempDir;

#[test]
fn test_detection_emits_stage_changes() {
    let temp_dir = TempDir::new().unwrap();
    let test_image = temp_dir.path().join("test.jpg");

    // Create a minimal test image
    std::fs::copy("example.jpg", &test_image).unwrap();

    let (tx, rx) = channel();

    let config = DetectionConfig {
        input: test_image.to_str().unwrap().to_string(),
        ..Default::default()
    };

    // This will fail because progress_tx parameter doesn't exist yet
    beaker::run_detection_with_progress(config, Some(tx)).unwrap();

    // Verify we received stage change events
    let events: Vec<ProcessingEvent> = rx.try_iter().collect();

    let stage_changes: Vec<_> = events.iter()
        .filter_map(|e| match e {
            ProcessingEvent::StageChange { stage, .. } => Some(*stage),
            _ => None,
        })
        .collect();

    // Should have Quality stage then Detection stage
    assert_eq!(stage_changes.len(), 2, "Should have 2 stage changes");
    assert_eq!(stage_changes[0], ProcessingStage::Quality);
    assert_eq!(stage_changes[1], ProcessingStage::Detection);
}
```

**Step 2: Run test to verify it fails**

Run: `just test`

Expected: FAIL with "cannot find function `run_detection_with_progress`"

**Step 3: Write minimal implementation**

In beaker/src/detection.rs:

```rust
pub fn run_detection_with_progress(
    config: DetectionConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
) -> anyhow::Result<usize> {
    log::info!("Running detection with quality analysis");

    // Get image count for stage change events
    let base_config = config.base();
    let image_files = crate::model_processing::get_image_files(base_config)?;
    let images_total = image_files.len();

    // Emit Quality stage start
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(crate::ProcessingEvent::StageChange {
            stage: crate::ProcessingStage::Quality,
            images_total,
        });
    }

    log::info!("   Analyzing image quality");
    let quality_config = crate::quality::QualityConfig::from_detection_config(&config);
    let quality_results = crate::model_processing::run_model_processing_with_quality_outputs::<crate::quality::QualityProcessor>(
        quality_config,
        progress_tx.clone(),
    );

    let config = match quality_results {
        Ok((_count, results)) => DetectionConfig {
            quality_results: Some(results),
            ..config
        },
        Err(e) => {
            log::warn!("Quality analysis failed: {}", e);
            config
        }
    };

    // Emit Detection stage start
    if let Some(ref tx) = progress_tx {
        let _ = tx.send(crate::ProcessingEvent::StageChange {
            stage: crate::ProcessingStage::Detection,
            images_total,
        });
    }

    log::info!("   Detecting...");
    let (count, _) = crate::model_processing::run_model_processing_with_quality_outputs::<DetectionProcessor>(
        config,
        progress_tx,
    )?;

    Ok(count)
}

pub fn run_detection(config: DetectionConfig) -> anyhow::Result<usize> {
    run_detection_with_progress(config, None)
}
```

**Step 4: Run test to verify it passes**

Run: `just test`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker/src/detection.rs beaker/tests/detection_progress_test.rs
git commit -m "feat: add progress channel support to detection with stage changes"
```

---

### Task 4: Add Cancellation Support

**Files:**
- Modify: beaker/src/model_processing.rs:200-350 (add cancel_flag parameter)
- Modify: beaker/src/detection.rs:95-150 (pass through cancel_flag)
- Modify: beaker/src/quality.rs:50-100 (pass through cancel_flag)
- Test: beaker/tests/cancellation_test.rs

**Step 1: Write the failing test**

```rust
// beaker/tests/cancellation_test.rs
use beaker::{DetectionConfig, ProcessingEvent};
use std::sync::mpsc::channel;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tempfile::TempDir;
use std::path::PathBuf;

#[test]
fn test_detection_respects_cancellation() {
    let temp_dir = TempDir::new().unwrap();

    // Create 5 test images
    for i in 0..5 {
        let test_image = temp_dir.path().join(format!("test_{}.jpg", i));
        std::fs::copy("example.jpg", &test_image).unwrap();
    }

    let (tx, rx) = channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_flag_clone = cancel_flag.clone();

    // Spawn processing thread
    let config = DetectionConfig {
        input: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let handle = std::thread::spawn(move || {
        beaker::run_detection_with_cancellation(config, Some(tx), Some(cancel_flag_clone))
    });

    // Wait for at least one image to be processed
    let mut processed_count = 0;
    for event in rx.iter() {
        if matches!(event, ProcessingEvent::ImageComplete { .. }) {
            processed_count += 1;
            if processed_count >= 2 {
                // Cancel after 2 images
                cancel_flag.store(true, Ordering::Relaxed);
                break;
            }
        }
    }

    // Wait for thread to finish
    let result = handle.join().unwrap();

    // Should succeed but with partial results
    assert!(result.is_ok());
    let count = result.unwrap();
    assert!(count < 5, "Should have processed fewer than 5 images due to cancellation");
    assert!(count >= 2, "Should have processed at least 2 images before cancellation");
}
```

**Step 2: Run test to verify it fails**

Run: `just test`

Expected: FAIL with "cannot find function `run_detection_with_cancellation`"

**Step 3: Write minimal implementation**

In beaker/src/model_processing.rs, add cancel_flag parameter:

```rust
pub fn run_model_processing_with_quality_outputs<P: ModelProcessor>(
    config: P::Config,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<(usize, std::collections::HashMap<String, crate::quality::QualityResult>)> {
    // ... existing setup code ...

    for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                log::info!("Processing cancelled by user after {} images", successful_count);
                break; // Exit loop gracefully
            }
        }

        // ... rest of loop unchanged ...
    }

    // ... rest of function unchanged ...
}
```

In beaker/src/detection.rs:

```rust
pub fn run_detection_with_cancellation(
    config: DetectionConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<usize> {
    log::info!("Running detection with quality analysis");

    let base_config = config.base();
    let image_files = crate::model_processing::get_image_files(base_config)?;
    let images_total = image_files.len();

    if let Some(ref tx) = progress_tx {
        let _ = tx.send(crate::ProcessingEvent::StageChange {
            stage: crate::ProcessingStage::Quality,
            images_total,
        });
    }

    log::info!("   Analyzing image quality");
    let quality_config = crate::quality::QualityConfig::from_detection_config(&config);
    let quality_results = crate::model_processing::run_model_processing_with_quality_outputs::<crate::quality::QualityProcessor>(
        quality_config,
        progress_tx.clone(),
        cancel_flag.clone(),
    );

    let config = match quality_results {
        Ok((_count, results)) => DetectionConfig {
            quality_results: Some(results),
            ..config
        },
        Err(e) => {
            log::warn!("Quality analysis failed: {}", e);
            config
        }
    };

    if let Some(ref tx) = progress_tx {
        let _ = tx.send(crate::ProcessingEvent::StageChange {
            stage: crate::ProcessingStage::Detection,
            images_total,
        });
    }

    log::info!("   Detecting...");
    let (count, _) = crate::model_processing::run_model_processing_with_quality_outputs::<DetectionProcessor>(
        config,
        progress_tx,
        cancel_flag,
    )?;

    Ok(count)
}

pub fn run_detection_with_progress(
    config: DetectionConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
) -> anyhow::Result<usize> {
    run_detection_with_cancellation(config, progress_tx, None)
}

pub fn run_detection(config: DetectionConfig) -> anyhow::Result<usize> {
    run_detection_with_cancellation(config, None, None)
}
```

Update quality.rs similarly:

```rust
pub fn run_quality_with_progress(
    config: QualityConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
) -> anyhow::Result<usize> {
    run_quality_with_cancellation(config, progress_tx, None)
}

pub fn run_quality_with_cancellation(
    config: QualityConfig,
    progress_tx: Option<std::sync::mpsc::Sender<crate::ProcessingEvent>>,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<usize> {
    let (count, _) = crate::model_processing::run_model_processing_with_quality_outputs::<QualityProcessor>(
        config,
        progress_tx,
        cancel_flag,
    )?;
    Ok(count)
}

pub fn run_quality(config: QualityConfig) -> anyhow::Result<usize> {
    run_quality_with_cancellation(config, None, None)
}
```

**Step 4: Run test to verify it passes**

Run: `just test`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker/src/model_processing.rs beaker/src/detection.rs beaker/src/quality.rs beaker/tests/cancellation_test.rs
git commit -m "feat: add cancellation support via AtomicBool flag"
```

---

### Task 5: Update CLI to Use New API

**Files:**
- Modify: beaker/src/main.rs:150-200 (cli calls to run_detection, run_quality)

**Step 1: No test needed (backward compatibility)**

The new API is backward compatible since we just pass `None` for the new parameters.

**Step 2: Verify existing tests still pass**

Run: `just test`

Expected: All tests PASS (backward compatibility confirmed)

**Step 3: Update CLI calls**

In beaker/src/main.rs, verify all calls use the existing functions (which now internally call the new API with None):

```rust
// In detection subcommand handler:
let count = crate::detection::run_detection(config)?;

// In quality subcommand handler:
let count = crate::quality::run_quality(config)?;
```

No changes needed - existing API is preserved.

**Step 4: Run integration tests**

Run: `just ci`

Expected: All tests and checks PASS

**Step 5: Commit**

```bash
git add beaker/src/main.rs
git commit -m "docs: verify CLI backward compatibility with new progress API"
```

---

## Part 2: Proposal 0 - File Navigation & Opening

Add welcome screen, file dialogs, drag & drop, and recent files support to beaker-gui.

---

### Task 6: Add Dependencies for File Dialogs

**Files:**
- Modify: beaker-gui/Cargo.toml

**Step 1: Write the failing test**

```rust
// beaker-gui/tests/file_dialog_deps_test.rs
#[test]
fn test_rfd_available() {
    // This will fail until we add rfd dependency
    use rfd::FileDialog;
    let _ = FileDialog::new();
}

#[test]
fn test_dirs_available() {
    use dirs;
    let _ = dirs::config_dir();
}

#[test]
fn test_serde_json_available() {
    use serde_json;
    let _ = serde_json::json!({});
}
```

**Step 2: Run test to verify it fails**

Run: `cd beaker-gui && cargo test`

Expected: FAIL with "unresolved import `rfd`"

**Step 3: Add dependencies**

In beaker-gui/Cargo.toml:

```toml
[dependencies]
# ... existing dependencies ...
rfd = "0.15"
dirs = "5.0"
serde_json = "1.0"
```

**Step 4: Run test to verify it passes**

Run: `cd beaker-gui && cargo test`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/Cargo.toml beaker-gui/tests/file_dialog_deps_test.rs
git commit -m "feat: add file dialog and config dependencies to beaker-gui"
```

---

### Task 7: Create RecentFiles Manager

**Files:**
- Create: beaker-gui/src/recent_files.rs
- Modify: beaker-gui/src/lib.rs (add module)
- Test: beaker-gui/tests/recent_files_test.rs

**Step 1: Write the failing test**

```rust
// beaker-gui/tests/recent_files_test.rs
use beaker_gui::RecentFiles;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_recent_files_add_and_get() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("recent.json");

    let mut recent = RecentFiles::new(config_path.clone());

    assert_eq!(recent.get_recent().len(), 0);

    recent.add_recent(PathBuf::from("/test/image1.jpg"));
    recent.add_recent(PathBuf::from("/test/image2.jpg"));

    let items = recent.get_recent();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].path, PathBuf::from("/test/image2.jpg")); // Most recent first
    assert_eq!(items[1].path, PathBuf::from("/test/image1.jpg"));
}

#[test]
fn test_recent_files_max_limit() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("recent.json");

    let mut recent = RecentFiles::new(config_path.clone());

    // Add 15 items (limit is 10)
    for i in 0..15 {
        recent.add_recent(PathBuf::from(format!("/test/image{}.jpg", i)));
    }

    let items = recent.get_recent();
    assert_eq!(items.len(), 10, "Should only keep 10 most recent");
}

#[test]
fn test_recent_files_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("recent.json");

    {
        let mut recent = RecentFiles::new(config_path.clone());
        recent.add_recent(PathBuf::from("/test/saved.jpg"));
        recent.save().unwrap();
    }

    // Load in new instance
    let recent2 = RecentFiles::new(config_path.clone());
    let items = recent2.get_recent();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].path, PathBuf::from("/test/saved.jpg"));
}
```

**Step 2: Run test to verify it fails**

Run: `cd beaker-gui && cargo test recent_files`

Expected: FAIL with "unresolved import `beaker_gui::RecentFiles`"

**Step 3: Write minimal implementation**

```rust
// beaker-gui/src/recent_files.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;

const MAX_RECENT_FILES: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentFileItem {
    pub path: PathBuf,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecentFiles {
    items: Vec<RecentFileItem>,
    #[serde(skip)]
    config_path: PathBuf,
}

impl RecentFiles {
    pub fn new(config_path: PathBuf) -> Self {
        let items = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Self { items, config_path }
    }

    pub fn add_recent(&mut self, path: PathBuf) {
        // Remove existing entry for this path
        self.items.retain(|item| item.path != path);

        // Add to front
        self.items.insert(0, RecentFileItem {
            path,
            timestamp: chrono::Utc::now(),
        });

        // Trim to max
        if self.items.len() > MAX_RECENT_FILES {
            self.items.truncate(MAX_RECENT_FILES);
        }
    }

    pub fn get_recent(&self) -> &[RecentFileItem] {
        &self.items
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn save(&self) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.items)?;
        std::fs::write(&self.config_path, json)?;
        Ok(())
    }
}
```

Add to beaker-gui/Cargo.toml:
```toml
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
```

Add to beaker-gui/src/lib.rs:
```rust
pub mod recent_files;
pub use recent_files::{RecentFiles, RecentFileItem};
```

**Step 4: Run test to verify it passes**

Run: `cd beaker-gui && cargo test recent_files`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/recent_files.rs beaker-gui/src/lib.rs beaker-gui/Cargo.toml beaker-gui/tests/recent_files_test.rs
git commit -m "feat: implement RecentFiles manager with persistence"
```

---

### Task 8: Create Welcome Screen View

**Files:**
- Create: beaker-gui/src/views/welcome.rs
- Modify: beaker-gui/src/views/mod.rs
- Modify: beaker-gui/src/lib.rs

**Step 1: Write the failing test (manual UI test)**

Create placeholder test file:
```rust
// beaker-gui/tests/welcome_view_test.rs
use beaker_gui::views::WelcomeView;

#[test]
fn test_welcome_view_creation() {
    let welcome = WelcomeView::new();
    // Basic smoke test - just verify it compiles
    assert!(true);
}
```

**Step 2: Run test to verify it fails**

Run: `cd beaker-gui && cargo test welcome_view`

Expected: FAIL with "cannot find `WelcomeView`"

**Step 3: Write minimal implementation**

```rust
// beaker-gui/src/views/welcome.rs
use eframe::egui;

pub struct WelcomeView {
    // Future: store tips, recent files reference, etc.
}

impl WelcomeView {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> WelcomeAction {
        let mut action = WelcomeAction::None;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(50.0);

                ui.heading("Beaker - Bird Analysis");
                ui.add_space(30.0);

                // Drop zone (placeholder for now)
                let drop_zone = ui.add_sized(
                    [400.0, 200.0],
                    egui::Frame::default()
                        .fill(ui.style().visuals.faint_bg_color)
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                ui.add_space(70.0);
                                ui.label("Drop image or folder here");
                                ui.add_space(10.0);
                                ui.label("📁 or 🖼️");
                            });
                        })
                        .response
                );

                ui.add_space(20.0);

                // Action buttons
                ui.horizontal(|ui| {
                    if ui.button("Open Image").clicked() {
                        action = WelcomeAction::OpenImage;
                    }
                    ui.add_space(20.0);
                    if ui.button("Open Folder").clicked() {
                        action = WelcomeAction::OpenFolder;
                    }
                });

                ui.add_space(30.0);

                // Recent files section (placeholder)
                ui.separator();
                ui.add_space(10.0);
                ui.label("Recent Files");
                ui.label("(None)");
            });
        });

        action
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WelcomeAction {
    None,
    OpenImage,
    OpenFolder,
    OpenRecent(std::path::PathBuf),
}
```

Add to beaker-gui/src/views/mod.rs:
```rust
pub mod welcome;
pub use welcome::{WelcomeView, WelcomeAction};
```

**Step 4: Run test to verify it passes**

Run: `cd beaker-gui && cargo test welcome_view`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/views/welcome.rs beaker-gui/src/views/mod.rs beaker-gui/tests/welcome_view_test.rs
git commit -m "feat: create basic welcome screen view"
```

---

### Task 9: Integrate File Dialog with Welcome Screen

**Files:**
- Modify: beaker-gui/src/views/welcome.rs
- Modify: beaker-gui/src/app.rs (integrate with main app)

**Step 1: Manual test only (UI interaction)**

No automated test - this requires manual testing with actual file dialogs.

**Step 2: Write implementation**

Update welcome.rs to trigger file dialogs:

```rust
// beaker-gui/src/views/welcome.rs
use eframe::egui;
use std::path::PathBuf;

pub struct WelcomeView {
    pending_action: Option<PendingDialogAction>,
}

enum PendingDialogAction {
    OpenImage,
    OpenFolder,
}

impl WelcomeView {
    pub fn new() -> Self {
        Self { pending_action: None }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> WelcomeAction {
        let mut action = WelcomeAction::None;

        // Check for pending file dialog actions (must be called in update, not from button click)
        if let Some(pending) = self.pending_action.take() {
            match pending {
                PendingDialogAction::OpenImage => {
                    if let Some(path) = Self::open_image_dialog() {
                        action = WelcomeAction::OpenImage(path);
                    }
                }
                PendingDialogAction::OpenFolder => {
                    if let Some(path) = Self::open_folder_dialog() {
                        action = WelcomeAction::OpenFolder(path);
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(50.0);

                ui.heading("Beaker - Bird Analysis");
                ui.add_space(30.0);

                // Drop zone
                let drop_zone_response = ui.add_sized(
                    [400.0, 200.0],
                    egui::Label::new("Drop image or folder here\n\n📁 or 🖼️")
                        .sense(egui::Sense::click())
                );

                // Handle dropped files
                if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
                    let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
                    if let Some(file) = dropped_files.first() {
                        if let Some(path) = &file.path {
                            if path.is_dir() {
                                action = WelcomeAction::OpenFolder(path.clone());
                            } else {
                                action = WelcomeAction::OpenImage(path.clone());
                            }
                        }
                    }
                }

                ui.add_space(20.0);

                // Action buttons
                ui.horizontal(|ui| {
                    if ui.button("Open Image").clicked() {
                        self.pending_action = Some(PendingDialogAction::OpenImage);
                    }
                    ui.add_space(20.0);
                    if ui.button("Open Folder").clicked() {
                        self.pending_action = Some(PendingDialogAction::OpenFolder);
                    }
                });
            });
        });

        action
    }

    fn open_image_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter("Images", &["jpg", "jpeg", "png"])
            .add_filter("Beaker TOML", &["toml"])
            .pick_file()
    }

    fn open_folder_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .pick_folder()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WelcomeAction {
    None,
    OpenImage(PathBuf),
    OpenFolder(PathBuf),
    OpenRecent(PathBuf),
}
```

**Step 3: Manual test**

Run: `cd beaker-gui && cargo run`

Expected:
- See welcome screen
- Click "Open Image" → native file dialog appears
- Click "Open Folder" → native folder dialog appears
- Drag & drop image → action triggered

**Step 4: Verify compilation**

Run: `cd beaker-gui && cargo build`

Expected: Builds successfully

**Step 5: Commit**

```bash
git add beaker-gui/src/views/welcome.rs
git commit -m "feat: add file dialogs and drag-drop to welcome screen"
```

---

### Task 10: Add Recent Files to Welcome Screen

**Files:**
- Modify: beaker-gui/src/views/welcome.rs
- Modify: beaker-gui/src/app.rs

**Step 1: Manual test only**

No automated test - requires UI interaction.

**Step 2: Write implementation**

Update welcome.rs to display recent files:

```rust
// beaker-gui/src/views/welcome.rs
use eframe::egui;
use std::path::PathBuf;
use crate::RecentFileItem;

pub struct WelcomeView {
    pending_action: Option<PendingDialogAction>,
}

// ... existing code ...

impl WelcomeView {
    // ... existing methods ...

    pub fn ui(&mut self, ctx: &egui::Context, recent_files: &[RecentFileItem]) -> WelcomeAction {
        let mut action = WelcomeAction::None;

        // ... existing pending action handling ...

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                // ... existing welcome UI ...

                ui.add_space(30.0);

                // Recent files section
                ui.separator();
                ui.add_space(10.0);
                ui.heading("Recent Files");
                ui.add_space(10.0);

                if recent_files.is_empty() {
                    ui.label("(None)");
                } else {
                    ui.vertical(|ui| {
                        for item in recent_files.iter().take(10) {
                            ui.horizontal(|ui| {
                                // Icon based on file type
                                let icon = if item.path.is_dir() { "📁" } else { "🖼️" };
                                ui.label(icon);

                                // Clickable path
                                let path_str = item.path.display().to_string();
                                if ui.button(&path_str).clicked() {
                                    action = WelcomeAction::OpenRecent(item.path.clone());
                                }

                                // Timestamp
                                let time_ago = Self::format_time_ago(&item.timestamp);
                                ui.label(format!("({})", time_ago));
                            });
                        }
                    });

                    ui.add_space(10.0);
                    if ui.button("Clear Recent").clicked() {
                        action = WelcomeAction::ClearRecent;
                    }
                }
            });
        });

        action
    }

    fn format_time_ago(timestamp: &chrono::DateTime<chrono::Utc>) -> String {
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(*timestamp);

        if duration.num_hours() < 1 {
            format!("{} minutes ago", duration.num_minutes())
        } else if duration.num_hours() < 24 {
            format!("{} hours ago", duration.num_hours())
        } else {
            format!("{} days ago", duration.num_days())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WelcomeAction {
    None,
    OpenImage(PathBuf),
    OpenFolder(PathBuf),
    OpenRecent(PathBuf),
    ClearRecent,
}
```

**Step 3: Manual test**

Run: `cd beaker-gui && cargo run`

Expected: Recent files list appears, clicking items triggers action

**Step 4: Verify compilation**

Run: `cd beaker-gui && cargo build`

Expected: Builds successfully

**Step 5: Commit**

```bash
git add beaker-gui/src/views/welcome.rs
git commit -m "feat: add recent files list to welcome screen"
```

---

## Part 3: Proposal A - Bulk/Directory Mode

Add directory processing with progress UI, gallery view, and navigation.

---

### Task 11: Create Processing State Types

**Files:**
- Create: beaker-gui/src/processing_state.rs
- Modify: beaker-gui/src/lib.rs

**Step 1: Write the failing test**

```rust
// beaker-gui/tests/processing_state_test.rs
use beaker_gui::processing::{ProcessingState, ImageState, ProcessingStatus};
use beaker::{ProcessingEvent, ProcessingStage, ProcessingResult};
use std::path::PathBuf;

#[test]
fn test_processing_state_initialization() {
    let image_paths = vec![
        PathBuf::from("/test/img1.jpg"),
        PathBuf::from("/test/img2.jpg"),
    ];

    let state = ProcessingState::new(image_paths);

    assert_eq!(state.images.len(), 2);
    assert!(matches!(state.images[0].status, ProcessingStatus::Waiting));
    assert_eq!(state.current_stage, ProcessingStage::Quality);
}

#[test]
fn test_processing_state_update_image_start() {
    let image_paths = vec![PathBuf::from("/test/img1.jpg")];
    let mut state = ProcessingState::new(image_paths);

    let event = ProcessingEvent::ImageStart {
        path: PathBuf::from("/test/img1.jpg"),
        index: 0,
        total: 1,
        stage: ProcessingStage::Quality,
    };

    state.update(event);

    assert!(matches!(state.images[0].status, ProcessingStatus::Processing));
}

#[test]
fn test_processing_state_update_image_complete_success() {
    let image_paths = vec![PathBuf::from("/test/img1.jpg")];
    let mut state = ProcessingState::new(image_paths);

    let event = ProcessingEvent::ImageComplete {
        path: PathBuf::from("/test/img1.jpg"),
        index: 0,
        result: ProcessingResult::Success {
            detections_count: 2,
            good_count: 1,
            bad_count: 0,
            unknown_count: 1,
            processing_time_ms: 123.0,
        },
    };

    state.update(event);

    match &state.images[0].status {
        ProcessingStatus::Success { .. } => {},
        _ => panic!("Expected Success status"),
    }
}

#[test]
fn test_processing_state_update_stage_change() {
    let image_paths = vec![PathBuf::from("/test/img1.jpg")];
    let mut state = ProcessingState::new(image_paths);

    let event = ProcessingEvent::StageChange {
        stage: ProcessingStage::Detection,
        images_total: 1,
    };

    state.update(event);

    assert_eq!(state.current_stage, ProcessingStage::Detection);
}
```

**Step 2: Run test to verify it fails**

Run: `cd beaker-gui && cargo test processing_state`

Expected: FAIL with "cannot find `processing` in `beaker_gui`"

**Step 3: Write minimal implementation**

```rust
// beaker-gui/src/processing_state.rs
use beaker::{ProcessingEvent, ProcessingStage, ProcessingResult};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ProcessingState {
    pub images: Vec<ImageState>,
    pub current_stage: ProcessingStage,
    pub overall_progress: f32,
}

#[derive(Debug, Clone)]
pub struct ImageState {
    pub image_path: PathBuf,
    pub status: ProcessingStatus,
}

#[derive(Debug, Clone)]
pub enum ProcessingStatus {
    Waiting,
    Processing,
    Success {
        detections_count: usize,
        good_count: usize,
        bad_count: usize,
        unknown_count: usize,
        processing_time_ms: f64,
    },
    Error {
        message: String,
    },
}

impl ProcessingState {
    pub fn new(image_paths: Vec<PathBuf>) -> Self {
        let images = image_paths
            .into_iter()
            .map(|path| ImageState {
                image_path: path,
                status: ProcessingStatus::Waiting,
            })
            .collect();

        Self {
            images,
            current_stage: ProcessingStage::Quality,
            overall_progress: 0.0,
        }
    }

    pub fn update(&mut self, event: ProcessingEvent) {
        match event {
            ProcessingEvent::ImageStart { index, .. } => {
                if index < self.images.len() {
                    self.images[index].status = ProcessingStatus::Processing;
                }
            }
            ProcessingEvent::ImageComplete { index, result, .. } => {
                if index < self.images.len() {
                    self.images[index].status = match result {
                        ProcessingResult::Success {
                            detections_count,
                            good_count,
                            bad_count,
                            unknown_count,
                            processing_time_ms,
                        } => ProcessingStatus::Success {
                            detections_count,
                            good_count,
                            bad_count,
                            unknown_count,
                            processing_time_ms,
                        },
                        ProcessingResult::Error { error_message } => ProcessingStatus::Error {
                            message: error_message,
                        },
                    };
                }

                // Update overall progress
                let completed = self.images.iter()
                    .filter(|img| !matches!(img.status, ProcessingStatus::Waiting | ProcessingStatus::Processing))
                    .count();
                self.overall_progress = completed as f32 / self.images.len() as f32;
            }
            ProcessingEvent::StageChange { stage, .. } => {
                self.current_stage = stage;
            }
            ProcessingEvent::Progress { completed, total, .. } => {
                self.overall_progress = completed as f32 / total as f32;
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.images.iter().all(|img| {
            !matches!(img.status, ProcessingStatus::Waiting | ProcessingStatus::Processing)
        })
    }
}
```

Add to beaker-gui/src/lib.rs:
```rust
pub mod processing_state;
pub use processing_state::{ProcessingState, ImageState, ProcessingStatus};
```

**Step 4: Run test to verify it passes**

Run: `cd beaker-gui && cargo test processing_state`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/processing_state.rs beaker-gui/src/lib.rs beaker-gui/tests/processing_state_test.rs
git commit -m "feat: create ProcessingState to track directory processing"
```

---

### Task 12: Create Directory Processing View (Progress UI)

**Files:**
- Create: beaker-gui/src/views/directory_processing.rs
- Modify: beaker-gui/src/views/mod.rs

**Step 1: Write smoke test**

```rust
// beaker-gui/tests/directory_processing_view_test.rs
use beaker_gui::views::DirectoryProcessingView;
use std::path::PathBuf;

#[test]
fn test_directory_processing_view_creation() {
    let image_paths = vec![PathBuf::from("/test/img1.jpg")];
    let _view = DirectoryProcessingView::new(PathBuf::from("/test"), image_paths);
    // Smoke test - just verify it compiles
}
```

**Step 2: Run test to verify it fails**

Run: `cd beaker-gui && cargo test directory_processing_view`

Expected: FAIL with "cannot find `DirectoryProcessingView`"

**Step 3: Write minimal implementation**

```rust
// beaker-gui/src/views/directory_processing.rs
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use beaker::{DetectionConfig, ProcessingEvent};
use crate::ProcessingState;

pub struct DirectoryProcessingView {
    directory_path: PathBuf,
    processing_state: ProcessingState,
    progress_receiver: Option<Receiver<ProcessingEvent>>,
    cancel_flag: Arc<AtomicBool>,
    processing_thread: Option<std::thread::JoinHandle<()>>,
}

impl DirectoryProcessingView {
    pub fn new(directory_path: PathBuf, image_paths: Vec<PathBuf>) -> Self {
        let processing_state = ProcessingState::new(image_paths.clone());
        let cancel_flag = Arc::new(AtomicBool::new(false));

        // Spawn processing thread
        let (tx, rx) = channel();
        let cancel_flag_clone = cancel_flag.clone();
        let dir_path_clone = directory_path.clone();

        let processing_thread = std::thread::spawn(move || {
            let config = DetectionConfig {
                input: dir_path_clone.to_str().unwrap().to_string(),
                ..Default::default()
            };

            match beaker::run_detection_with_cancellation(config, Some(tx), Some(cancel_flag_clone)) {
                Ok(_) => {
                    log::info!("Detection completed successfully");
                }
                Err(e) => {
                    log::error!("Detection failed: {}", e);
                }
            }
        });

        Self {
            directory_path,
            processing_state,
            progress_receiver: Some(rx),
            cancel_flag,
            processing_thread: Some(processing_thread),
        }
    }

    pub fn update(&mut self, ctx: &egui::Context) -> DirectoryProcessingAction {
        let mut action = DirectoryProcessingAction::None;

        // Check for progress events
        if let Some(ref rx) = self.progress_receiver {
            while let Ok(event) = rx.try_recv() {
                self.processing_state.update(event);
                ctx.request_repaint(); // Update UI immediately
            }
        }

        // Check if processing is complete
        if self.processing_state.is_complete() {
            action = DirectoryProcessingAction::Complete;
        }

        action
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> DirectoryProcessingAction {
        let action = self.update(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.heading(format!(
                    "Processing: {}",
                    self.directory_path.display()
                ));
                ui.add_space(10.0);

                // Stage indicator
                ui.label(format!("Stage: {:?}", self.processing_state.current_stage));

                // Progress bar
                ui.add(egui::ProgressBar::new(self.processing_state.overall_progress)
                    .text(format!(
                        "{:.0}%",
                        self.processing_state.overall_progress * 100.0
                    )));

                ui.add_space(20.0);

                // Cancel button
                if ui.button("Cancel").clicked() {
                    self.cancel_flag.store(true, Ordering::Relaxed);
                }

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                // Image list with status
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, image_state) in self.processing_state.images.iter().enumerate() {
                        ui.horizontal(|ui| {
                            // Status icon
                            let (icon, color) = match &image_state.status {
                                crate::ProcessingStatus::Waiting => ("⏸", egui::Color32::GRAY),
                                crate::ProcessingStatus::Processing => ("⏳", egui::Color32::YELLOW),
                                crate::ProcessingStatus::Success { .. } => ("✓", egui::Color32::GREEN),
                                crate::ProcessingStatus::Error { .. } => ("⚠", egui::Color32::RED),
                            };

                            ui.colored_label(color, icon);

                            // Filename
                            let filename = image_state.image_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy();
                            ui.label(&filename);

                            // Status text
                            match &image_state.status {
                                crate::ProcessingStatus::Waiting => {
                                    ui.label("Waiting...");
                                }
                                crate::ProcessingStatus::Processing => {
                                    ui.label("Processing...");
                                }
                                crate::ProcessingStatus::Success { detections_count, good_count, unknown_count, .. } => {
                                    ui.label(format!(
                                        "{} detections ({} good, {} unknown)",
                                        detections_count, good_count, unknown_count
                                    ));
                                }
                                crate::ProcessingStatus::Error { message } => {
                                    ui.label(format!("Error: {}", message));
                                }
                            }
                        });
                    }
                });
            });
        });

        action
    }
}

impl Drop for DirectoryProcessingView {
    fn drop(&mut self) {
        // Ensure cancellation and wait for thread
        self.cancel_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.processing_thread.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DirectoryProcessingAction {
    None,
    Complete,
    Cancelled,
}
```

Add to beaker-gui/src/views/mod.rs:
```rust
pub mod directory_processing;
pub use directory_processing::{DirectoryProcessingView, DirectoryProcessingAction};
```

**Step 4: Run test to verify it passes**

Run: `cd beaker-gui && cargo test directory_processing_view`

Expected: PASS

**Step 5: Commit**

```bash
git add beaker-gui/src/views/directory_processing.rs beaker-gui/src/views/mod.rs beaker-gui/tests/directory_processing_view_test.rs
git commit -m "feat: create directory processing view with progress UI"
```

---

## Summary

This implementation plan provides:

1. **Lib changes (Tasks 1-5)**: Add progress callbacks and cancellation to beaker lib (~100 LOC)
2. **File navigation (Tasks 6-10)**: Welcome screen, file dialogs, recent files (~400 LOC)
3. **Directory processing (Tasks 11-12)**: Background processing with progress UI (~300 LOC)

**Remaining tasks** for complete Proposal A implementation:
- Task 13: Load detection results after processing
- Task 14: Create gallery view for processed images
- Task 15: Create aggregate detection list
- Task 16: Add navigation controls

**Remaining proposals** (B-H):
- Proposal B: Quality triage workflow
- Proposal C: Heatmap visualization
- Proposal D: Rich metrics display
- Proposal E: Zoom & pan
- Proposal F: Keyboard shortcuts
- Proposal G: Export & reporting
- Proposal H: Comparison view

Each task follows TDD, includes exact file paths, complete code, and commands with expected output. Tasks are 2-5 minutes each for rapid iteration.

**Next steps:**
1. Run `just ci` after completing each part
2. Commit frequently
3. Test manually with real images
4. Continue with remaining tasks following same pattern
