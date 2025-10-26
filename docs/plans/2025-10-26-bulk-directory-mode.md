# Bulk/Directory Mode Implementation Plan

For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable GUI to process directories of images with live progress tracking, gallery view, aggregate detection list, and navigation.

**Architecture:** Create DirectoryView component that spawns background thread to run beaker detection with progress callbacks via channels. Display per-image status during processing, then transition to gallery + aggregate detection list when complete.

**Tech Stack:** egui, std::sync::mpsc channels, std::thread, beaker library (ProcessingEvent API), image thumbnails

**Prerequisites:**
- Phase 0 (File Navigation & Opening) is complete
- Library changes (ProcessingEvent in beaker/src/model_processing.rs) are complete
- run_detection() already accepts `Option<Sender<ProcessingEvent>>` parameter

---

## Task 1: Create DirectoryView Data Structures

**Files:**
- Create: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_directory_view_creation() {
        let dir_path = PathBuf::from("/tmp/test_images");
        let image_paths = vec![
            PathBuf::from("/tmp/test_images/img1.jpg"),
            PathBuf::from("/tmp/test_images/img2.jpg"),
        ];

        let view = DirectoryView::new(dir_path, image_paths);

        assert_eq!(view.images.len(), 2);
        assert_eq!(view.current_image_idx, 0);
        assert!(matches!(view.images[0].status, ProcessingStatus::Waiting));
    }
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `DirectoryView` not defined

### Step 3: Write minimal implementation

```rust
// File: beaker-gui/src/views/directory.rs

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// Status of image processing
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStatus {
    Waiting,
    Processing,
    Success {
        detections_count: usize,
        good_count: usize,
        unknown_count: usize,
        bad_count: usize,
        processing_time_ms: f64,
    },
    Error {
        message: String,
    },
}

/// State for a single image in the directory
pub struct ImageState {
    pub path: PathBuf,
    pub status: ProcessingStatus,
    // Will be populated after processing
    pub detections: Vec<crate::views::detection::Detection>,
    pub thumbnail: Option<egui::TextureHandle>,
}

/// Directory/Bulk mode view
pub struct DirectoryView {
    pub directory_path: PathBuf,
    pub images: Vec<ImageState>,
    pub current_image_idx: usize,
    pub selected_detection_idx: Option<usize>,

    // Processing state
    progress_receiver: Option<Receiver<beaker::ProcessingEvent>>,
    cancel_flag: Arc<AtomicBool>,

    // Aggregate detection list (populated after processing)
    all_detections: Vec<DetectionRef>,

    // Filter state
    show_good: bool,
    show_unknown: bool,
    show_bad: bool,
}

/// Reference to a detection in the flattened list
#[derive(Clone)]
pub struct DetectionRef {
    pub image_idx: usize,
    pub detection_idx: usize,
}

impl DirectoryView {
    pub fn new(directory_path: PathBuf, image_paths: Vec<PathBuf>) -> Self {
        let images = image_paths
            .into_iter()
            .map(|path| ImageState {
                path,
                status: ProcessingStatus::Waiting,
                detections: Vec::new(),
                thumbnail: None,
            })
            .collect();

        Self {
            directory_path,
            images,
            current_image_idx: 0,
            selected_detection_idx: None,
            progress_receiver: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            all_detections: Vec::new(),
            show_good: true,
            show_unknown: true,
            show_bad: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.label("Directory view - under construction");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_directory_view_creation() {
        let dir_path = PathBuf::from("/tmp/test_images");
        let image_paths = vec![
            PathBuf::from("/tmp/test_images/img1.jpg"),
            PathBuf::from("/tmp/test_images/img2.jpg"),
        ];

        let view = DirectoryView::new(dir_path, image_paths);

        assert_eq!(view.images.len(), 2);
        assert_eq!(view.current_image_idx, 0);
        assert!(matches!(view.images[0].status, ProcessingStatus::Waiting));
    }
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): add DirectoryView data structures

- Add ProcessingStatus enum (Waiting, Processing, Success, Error)
- Add ImageState struct for per-image state tracking
- Add DirectoryView struct with image list and processing state
- Add DetectionRef for aggregate detection list"
```

---

## Task 2: Register DirectoryView in Module System

**Files:**
- Modify: beaker-gui/src/views/mod.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/mod.rs (add to existing tests or create new test)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directory_view_module_accessible() {
        use std::path::PathBuf;
        let _view = DirectoryView::new(PathBuf::from("/tmp"), vec![]);
        // If this compiles, the module is properly exported
    }
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `DirectoryView` not in scope

### Step 3: Implement module registration

```rust
// File: beaker-gui/src/views/mod.rs

mod detection;
mod welcome;
mod directory;  // Add this line

pub use detection::{Detection, DetectionView};
pub use welcome::{WelcomeAction, WelcomeView};
pub use directory::{DirectoryView, ProcessingStatus, ImageState};  // Add this line
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/views/mod.rs
git commit -m "feat(gui): export DirectoryView from views module"
```

---

## Task 3: Add DirectoryView to AppState

**Files:**
- Modify: beaker-gui/src/app.rs:23-26

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/app.rs (add to bottom of file)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_supports_directory_view() {
        let dir_view = DirectoryView::new(
            PathBuf::from("/tmp"),
            vec![PathBuf::from("/tmp/img1.jpg")],
        );
        let _state = AppState::Directory(dir_view);
        // If this compiles, AppState has Directory variant
    }
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - no `Directory` variant in `AppState`

### Step 3: Implement AppState changes

```rust
// File: beaker-gui/src/app.rs

// Modify the enum (around line 23-26)
enum AppState {
    Welcome(WelcomeView),
    Detection(DetectionView),
    Directory(DirectoryView),  // Add this line
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/app.rs
git commit -m "feat(gui): add Directory variant to AppState"
```

---

## Task 4: Wire DirectoryView into App Update Loop

**Files:**
- Modify: beaker-gui/src/app.rs:219-236

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/app.rs (add to tests section)

#[test]
fn test_app_renders_directory_view() {
    let dir_view = DirectoryView::new(
        PathBuf::from("/tmp"),
        vec![PathBuf::from("/tmp/img1.jpg")],
    );
    let mut app = BeakerApp {
        state: AppState::Directory(dir_view),
        recent_files: RecentFiles::default(),
        use_native_menu: false,
        pending_menu_file_dialog: Arc::new(Mutex::new(None)),
        #[cfg(target_os = "macos")]
        menu: None,
        #[cfg(target_os = "macos")]
        menu_rx: None,
    };

    // This should compile and not panic
    let _ = format!("{:?}", "Testing directory view in app");
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - missing match arm for `AppState::Directory`

### Step 3: Implement the update loop change

```rust
// File: beaker-gui/src/app.rs (modify around line 219-236)

egui::CentralPanel::default().show(ctx, |ui| match &mut self.state {
    AppState::Welcome(welcome_view) => {
        let action = welcome_view.show(ctx, ui);
        match action {
            WelcomeAction::OpenPaths(paths) => {
                eprintln!(
                    "[BeakerApp] Received action: OpenPaths({} path(s))",
                    paths.len()
                );
                self.open_paths(paths);
            }
            WelcomeAction::None => {}
        }
    }
    AppState::Detection(detection_view) => {
        detection_view.show(ctx, ui);
    }
    AppState::Directory(directory_view) => {  // Add this arm
        directory_view.show(ctx, ui);
    }
});
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/app.rs
git commit -m "feat(gui): wire DirectoryView into app update loop"
```

---

## Task 5: Implement Folder Opening Logic

**Files:**
- Modify: beaker-gui/src/app.rs:155-160

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/app.rs (add to tests)

#[test]
fn test_open_folder_creates_directory_view() {
    use tempfile::TempDir;
    use std::fs::File;

    let temp_dir = TempDir::new().unwrap();
    let img1 = temp_dir.path().join("img1.jpg");
    let img2 = temp_dir.path().join("img2.jpg");

    // Create empty files
    File::create(&img1).unwrap();
    File::create(&img2).unwrap();

    let mut app = BeakerApp::new(false, None);
    app.open_folder(temp_dir.path().to_path_buf());

    // Should transition to Directory state
    assert!(matches!(app.state, AppState::Directory(_)));
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Test fails - `open_folder` still has TODO and doesn't create DirectoryView

### Step 3: Implement folder opening

```rust
// File: beaker-gui/src/app.rs (modify lines 155-160)

fn open_folder(&mut self, path: PathBuf) {
    eprintln!("[BeakerApp] Opening folder: {:?}", path);

    // Collect image files from directory
    let image_paths = Self::collect_image_files(&path);
    eprintln!("[BeakerApp] Found {} image files in folder", image_paths.len());

    if image_paths.is_empty() {
        eprintln!("[BeakerApp] WARNING: No supported image files found in folder");
        return;
    }

    // Create DirectoryView and start processing
    let dir_view = DirectoryView::new(path.clone(), image_paths);
    let _ = self.recent_files.add(path, RecentItemType::Folder);
    self.state = AppState::Directory(dir_view);
}

/// Collect all supported image files from a directory (non-recursive)
fn collect_image_files(dir_path: &PathBuf) -> Vec<PathBuf> {
    let mut image_files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                        image_files.push(path);
                    }
                }
            }
        }
    }

    // Sort for consistent ordering
    image_files.sort();
    image_files
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/app.rs
git commit -m "feat(gui): implement folder opening with image file collection

- Implement open_folder() to create DirectoryView
- Add collect_image_files() helper to find .jpg/.jpeg/.png files
- Transition to Directory state when folder opened
- Add to recent files"
```

---

## Task 6: Start Background Processing Thread

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_start_processing_creates_thread() {
    use tempfile::TempDir;
    use std::fs::File;

    let temp_dir = TempDir::new().unwrap();
    let img1 = temp_dir.path().join("img1.jpg");
    File::create(&img1).unwrap();

    let mut view = DirectoryView::new(
        temp_dir.path().to_path_buf(),
        vec![img1.clone()],
    );

    view.start_processing();

    // Should have receiver set up
    assert!(view.progress_receiver.is_some());
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `start_processing` method doesn't exist

### Step 3: Implement start_processing method

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing new() method ...

    /// Start background processing of all images
    pub fn start_processing(&mut self) {
        use std::sync::mpsc::channel;

        let (tx, rx) = channel();
        self.progress_receiver = Some(rx);

        // Collect paths to process
        let image_paths: Vec<PathBuf> = self.images.iter().map(|img| img.path.clone()).collect();
        let cancel_flag = Arc::clone(&self.cancel_flag);

        // Spawn background thread
        std::thread::spawn(move || {
            eprintln!("[DirectoryView] Background thread started, processing {} images", image_paths.len());

            // Create temp output directory
            let temp_dir = std::env::temp_dir().join(format!("beaker-gui-bulk-{}", std::process::id()));
            if let Err(e) = std::fs::create_dir_all(&temp_dir) {
                eprintln!("[DirectoryView] ERROR: Failed to create temp dir: {}", e);
                return;
            }

            // Build detection config
            let sources: Vec<String> = image_paths.iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            let base_config = beaker::config::BaseModelConfig {
                sources,
                device: "auto".to_string(),
                output_dir: Some(temp_dir.to_str().unwrap().to_string()),
                skip_metadata: false,
                strict: false,  // Don't fail on single image errors
                force: true,
            };

            let config = beaker::config::DetectionConfig {
                base: base_config,
                confidence: 0.5,
                crop_classes: std::collections::HashSet::new(),
                bounding_box: true,
                model_path: None,
                model_url: None,
                model_checksum: None,
                quality_results: None,
            };

            // Run detection with progress callback
            match beaker::detection::run_detection(config, Some(tx.clone()), Some(cancel_flag)) {
                Ok(count) => {
                    eprintln!("[DirectoryView] Processing complete: {} images processed", count);
                }
                Err(e) => {
                    eprintln!("[DirectoryView] ERROR: Processing failed: {}", e);
                }
            }
        });
    }

    // ... existing show() method ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): implement background processing thread

- Add start_processing() method to DirectoryView
- Spawn background thread that runs beaker detect
- Create temp output directory for results
- Pass progress callback channel to beaker
- Support cancellation via cancel_flag"
```

---

## Task 7: Call start_processing When DirectoryView Created

**Files:**
- Modify: beaker-gui/src/app.rs:155-170

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/app.rs (modify existing test)

#[test]
fn test_open_folder_starts_processing() {
    use tempfile::TempDir;
    use std::fs::File;
    use std::thread;
    use std::time::Duration;

    let temp_dir = TempDir::new().unwrap();
    let img1 = temp_dir.path().join("img1.jpg");
    File::create(&img1).unwrap();

    let mut app = BeakerApp::new(false, None);
    app.open_folder(temp_dir.path().to_path_buf());

    // Give processing thread time to start
    thread::sleep(Duration::from_millis(100));

    // Should transition to Directory state with processing started
    match &app.state {
        AppState::Directory(view) => {
            // At least one image should be in Waiting or Processing state
            assert!(!view.images.is_empty());
        }
        _ => panic!("Expected Directory state"),
    }
}
```

### Step 2: Run test to verify behavior

**Run:**
```bash
just test
```

**Expected:** Test may fail if processing hasn't started automatically

### Step 3: Modify open_folder to start processing

```rust
// File: beaker-gui/src/app.rs (modify open_folder method)

fn open_folder(&mut self, path: PathBuf) {
    eprintln!("[BeakerApp] Opening folder: {:?}", path);

    // Collect image files from directory
    let image_paths = Self::collect_image_files(&path);
    eprintln!("[BeakerApp] Found {} image files in folder", image_paths.len());

    if image_paths.is_empty() {
        eprintln!("[BeakerApp] WARNING: No supported image files found in folder");
        return;
    }

    // Create DirectoryView and start processing
    let mut dir_view = DirectoryView::new(path.clone(), image_paths);
    dir_view.start_processing();  // Add this line
    let _ = self.recent_files.add(path, RecentItemType::Folder);
    self.state = AppState::Directory(dir_view);
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/app.rs
git commit -m "feat(gui): automatically start processing when folder opened"
```

---

## Task 8: Handle Progress Events in DirectoryView

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_update_from_progress_event_image_start() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let mut view = DirectoryView::new(dir_path, vec![img1.clone()]);

    let event = beaker::ProcessingEvent::ImageStart {
        path: img1,
        index: 0,
        total: 1,
        stage: beaker::ProcessingStage::Detection,
    };

    view.update_from_event(event);

    assert!(matches!(view.images[0].status, ProcessingStatus::Processing));
}

#[test]
fn test_update_from_progress_event_image_success() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let mut view = DirectoryView::new(dir_path, vec![img1.clone()]);

    let event = beaker::ProcessingEvent::ImageSuccess {
        path: img1,
        index: 0,
    };

    view.update_from_event(event);

    // Status should be Success (with placeholder counts)
    match &view.images[0].status {
        ProcessingStatus::Success { .. } => {},
        _ => panic!("Expected Success status"),
    }
}

#[test]
fn test_update_from_progress_event_image_error() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let mut view = DirectoryView::new(dir_path, vec![img1.clone()]);

    let event = beaker::ProcessingEvent::ImageError {
        path: img1,
        index: 0,
        error: "Test error".to_string(),
    };

    view.update_from_event(event);

    match &view.images[0].status {
        ProcessingStatus::Error { message } => {
            assert_eq!(message, "Test error");
        },
        _ => panic!("Expected Error status"),
    }
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `update_from_event` method doesn't exist

### Step 3: Implement update_from_event method

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Update state based on progress event from background thread
    pub fn update_from_event(&mut self, event: beaker::ProcessingEvent) {
        match event {
            beaker::ProcessingEvent::ImageStart { index, .. } => {
                if index < self.images.len() {
                    self.images[index].status = ProcessingStatus::Processing;
                    eprintln!("[DirectoryView] Image {} started processing", index);
                }
            }
            beaker::ProcessingEvent::ImageSuccess { index, .. } => {
                if index < self.images.len() {
                    // For now, use placeholder values
                    // We'll load actual detection data from TOML in next task
                    self.images[index].status = ProcessingStatus::Success {
                        detections_count: 0,
                        good_count: 0,
                        unknown_count: 0,
                        bad_count: 0,
                        processing_time_ms: 0.0,
                    };
                    eprintln!("[DirectoryView] Image {} completed successfully", index);
                }
            }
            beaker::ProcessingEvent::ImageError { index, error, .. } => {
                if index < self.images.len() {
                    self.images[index].status = ProcessingStatus::Error {
                        message: error.clone(),
                    };
                    eprintln!("[DirectoryView] Image {} failed: {}", index, error);
                }
            }
            beaker::ProcessingEvent::StageChange { stage, .. } => {
                eprintln!("[DirectoryView] Stage changed to {:?}", stage);
            }
        }
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): implement progress event handling

- Add update_from_event() method
- Handle ImageStart -> set status to Processing
- Handle ImageSuccess -> set status to Success
- Handle ImageError -> set status to Error with message
- Add debug logging for progress tracking"
```

---

## Task 9: Poll Progress Events in DirectoryView show() Method

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write failing test (integration test approach)

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_show_polls_progress_events() {
    use std::sync::mpsc::channel;

    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let mut view = DirectoryView::new(dir_path, vec![img1.clone()]);

    // Manually set up receiver and send event
    let (tx, rx) = channel();
    view.progress_receiver = Some(rx);

    let event = beaker::ProcessingEvent::ImageStart {
        path: img1,
        index: 0,
        total: 1,
        stage: beaker::ProcessingStage::Detection,
    };
    tx.send(event).unwrap();

    // Call poll_events (extracted helper method)
    view.poll_events();

    // Event should have been processed
    assert!(matches!(view.images[0].status, ProcessingStatus::Processing));
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `poll_events` method doesn't exist

### Step 3: Implement event polling

```rust
// File: beaker-gui/src/views/directory.rs (modify impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Poll for progress events from background thread
    fn poll_events(&mut self) {
        if let Some(ref rx) = self.progress_receiver {
            // Drain all available events (non-blocking)
            while let Ok(event) = rx.try_recv() {
                self.update_from_event(event);
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Poll progress events
        self.poll_events();

        // Request repaint if processing is active
        let is_processing = self.images.iter().any(|img| {
            matches!(img.status, ProcessingStatus::Waiting | ProcessingStatus::Processing)
        });
        if is_processing {
            ctx.request_repaint();
        }

        ui.label("Directory view - under construction");
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): poll progress events in show() method

- Add poll_events() helper method
- Poll events in show() (non-blocking)
- Request repaint while processing is active
- Ensures UI updates as events arrive"
```

---

## Task 10: Display Processing Progress UI

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the test (visual test, documented)

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_calculate_progress_stats() {
    let dir_path = PathBuf::from("/tmp/test");
    let images = vec![
        PathBuf::from("/tmp/test/img1.jpg"),
        PathBuf::from("/tmp/test/img2.jpg"),
        PathBuf::from("/tmp/test/img3.jpg"),
    ];
    let mut view = DirectoryView::new(dir_path, images);

    // Set various statuses
    view.images[0].status = ProcessingStatus::Success {
        detections_count: 2,
        good_count: 1,
        unknown_count: 1,
        bad_count: 0,
        processing_time_ms: 100.0,
    };
    view.images[1].status = ProcessingStatus::Processing;
    view.images[2].status = ProcessingStatus::Waiting;

    let (completed, total, processing_idx) = view.calculate_progress_stats();

    assert_eq!(completed, 1);
    assert_eq!(total, 3);
    assert_eq!(processing_idx, Some(1));
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `calculate_progress_stats` doesn't exist

### Step 3: Implement progress UI

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Calculate progress statistics
    fn calculate_progress_stats(&self) -> (usize, usize, Option<usize>) {
        let completed = self.images.iter().filter(|img| {
            matches!(img.status, ProcessingStatus::Success { .. } | ProcessingStatus::Error { .. })
        }).count();

        let total = self.images.len();

        let processing_idx = self.images.iter().position(|img| {
            matches!(img.status, ProcessingStatus::Processing)
        });

        (completed, total, processing_idx)
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Poll progress events
        self.poll_events();

        // Request repaint if processing is active
        let is_processing = self.images.iter().any(|img| {
            matches!(img.status, ProcessingStatus::Waiting | ProcessingStatus::Processing)
        });
        if is_processing {
            ctx.request_repaint();
        }

        // Show progress UI if processing
        if is_processing {
            self.show_processing_ui(ui);
        } else {
            self.show_gallery_ui(ctx, ui);
        }
    }

    fn show_processing_ui(&self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(20.0);

            // Title
            ui.heading(format!("Processing: {}", self.directory_path.display()));
            ui.add_space(20.0);

            // Progress stats
            let (completed, total, processing_idx) = self.calculate_progress_stats();
            let progress = completed as f32 / total as f32;

            // Progress bar
            ui.add(egui::ProgressBar::new(progress)
                .text(format!("{}/{} ({:.0}%)", completed, total, progress * 100.0))
                .desired_width(600.0));

            ui.add_space(10.0);

            // Currently processing
            if let Some(idx) = processing_idx {
                let filename = self.images[idx].path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                ui.label(format!("Currently processing: {}", filename));
            }

            ui.add_space(20.0);

            // Cancel button
            if ui.button("Cancel").clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }

            ui.add_space(30.0);

            // Image list with status
            ui.label(egui::RichText::new("Images:").size(16.0));
            ui.add_space(10.0);

            egui::ScrollArea::vertical()
                .max_height(400.0)
                .show(ui, |ui| {
                    for (idx, img) in self.images.iter().enumerate() {
                        let filename = img.path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown");

                        let (icon, status_text, color) = match &img.status {
                            ProcessingStatus::Waiting => {
                                ("⏸", "Waiting...", egui::Color32::GRAY)
                            }
                            ProcessingStatus::Processing => {
                                ("⏳", "Processing...", egui::Color32::BLUE)
                            }
                            ProcessingStatus::Success { detections_count, good_count, unknown_count, .. } => {
                                let text = format!("{} detections ({} good, {} unknown)",
                                    detections_count, good_count, unknown_count);
                                ("✓", text.as_str(), egui::Color32::GREEN)
                            }
                            ProcessingStatus::Error { message } => {
                                ("⚠", message.as_str(), egui::Color32::RED)
                            }
                        };

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(icon).color(color).size(16.0));
                            ui.label(format!("{}: {}", filename, status_text));
                        });
                    }
                });
        });
    }

    fn show_gallery_ui(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.label("Gallery view - coming soon");
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 5: Manual test with real folder

**Run:**
```bash
# Create test folder with images
mkdir -p /tmp/beaker-test-images
# Copy some test images to /tmp/beaker-test-images/

# Run GUI and open folder
just build-release
./target/release/beaker-gui

# Open the test folder and verify progress UI displays
```

**Expected:** Progress UI shows with live updates as images process

### Step 6: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): implement processing progress UI

- Add calculate_progress_stats() helper
- Add show_processing_ui() with progress bar
- Display current processing status per image
- Show icons: ⏸ waiting, ⏳ processing, ✓ success, ⚠ error
- Add Cancel button
- Add scrollable image status list
- Request repaint while processing active"
```

---

## Task 11: Load Detection Data from TOML After Processing

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_load_detection_data_from_toml() {
    use tempfile::TempDir;
    use std::fs;

    let temp_dir = TempDir::new().unwrap();
    let img_path = temp_dir.path().join("test.jpg");
    let toml_path = temp_dir.path().join("test.beaker.toml");

    // Create test TOML file with detection data
    let toml_content = r#"
[detect]
detections = [
    { class_name = "head", confidence = 0.95, x1 = 10.0, y1 = 20.0, x2 = 100.0, y2 = 120.0, quality = { triage_decision = "good" } }
]
"#;
    fs::write(&toml_path, toml_content).unwrap();

    let detections = DirectoryView::load_detections_from_toml(&toml_path).unwrap();

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_name, "head");
    assert_eq!(detections[0].confidence, 0.95);
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `load_detections_from_toml` doesn't exist

### Step 3: Implement TOML loading

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Load detection data from TOML file
    fn load_detections_from_toml(toml_path: &PathBuf) -> anyhow::Result<Vec<crate::views::detection::Detection>> {
        let toml_data = std::fs::read_to_string(toml_path)?;
        let toml_value: toml::Value = toml::from_str(&toml_data)?;

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

                    let blur_score = det_table
                        .get("quality")
                        .and_then(|q| q.as_table())
                        .and_then(|q| q.get("roi_blur_probability_mean"))
                        .and_then(|v| v.as_float())
                        .map(|v| v as f32);

                    let x1 = det_table.get("x1").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                    let y1 = det_table.get("y1").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                    let x2 = det_table.get("x2").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;
                    let y2 = det_table.get("y2").and_then(|v| v.as_float()).unwrap_or(0.0) as f32;

                    detections.push(crate::views::detection::Detection {
                        class_name,
                        confidence,
                        blur_score,
                        x1,
                        y1,
                        x2,
                        y2,
                    });
                }
            }
        }

        Ok(detections)
    }

    /// Count quality triage results from detections
    fn count_triage_results(toml_path: &PathBuf) -> (usize, usize, usize) {
        let toml_data = match std::fs::read_to_string(toml_path) {
            Ok(data) => data,
            Err(_) => return (0, 0, 0),
        };

        let toml_value: toml::Value = match toml::from_str(&toml_data) {
            Ok(val) => val,
            Err(_) => return (0, 0, 0),
        };

        let mut good = 0;
        let mut unknown = 0;
        let mut bad = 0;

        if let Some(dets) = toml_value
            .get("detect")
            .and_then(|v| v.get("detections"))
            .and_then(|v| v.as_array())
        {
            for det_toml in dets {
                if let Some(triage) = det_toml
                    .get("quality")
                    .and_then(|q| q.as_table())
                    .and_then(|q| q.get("triage_decision"))
                    .and_then(|v| v.as_str())
                {
                    match triage {
                        "good" => good += 1,
                        "unknown" => unknown += 1,
                        "bad" => bad += 1,
                        _ => {}
                    }
                }
            }
        }

        (good, unknown, bad)
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): add TOML parsing for detection data

- Add load_detections_from_toml() method
- Add count_triage_results() helper
- Parse detection metadata from .beaker.toml files
- Extract quality triage decisions (good/unknown/bad)"
```

---

## Task 12: Load Detections After ImageSuccess Event

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_update_from_event_loads_detections() {
    use tempfile::TempDir;
    use std::fs;

    let temp_dir = TempDir::new().unwrap();
    let img_path = temp_dir.path().join("test.jpg");
    let toml_path = temp_dir.path().join("test.beaker.toml");

    // Create dummy image and TOML
    fs::write(&img_path, b"fake_image").unwrap();
    let toml_content = r#"
[detect]
detections = [
    { class_name = "head", confidence = 0.95, x1 = 10.0, y1 = 20.0, x2 = 100.0, y2 = 120.0, quality = { triage_decision = "good" } }
]
"#;
    fs::write(&toml_path, toml_content).unwrap();

    let mut view = DirectoryView::new(
        temp_dir.path().to_path_buf(),
        vec![img_path.clone()],
    );

    // Set output directory so we can find TOML
    view.output_dir = Some(temp_dir.path().to_path_buf());

    let event = beaker::ProcessingEvent::ImageSuccess {
        path: img_path,
        index: 0,
    };

    view.update_from_event(event);

    // Should have loaded detections
    assert_eq!(view.images[0].detections.len(), 1);

    // Should have correct counts
    match &view.images[0].status {
        ProcessingStatus::Success { detections_count, good_count, .. } => {
            assert_eq!(*detections_count, 1);
            assert_eq!(*good_count, 1);
        }
        _ => panic!("Expected Success status"),
    }
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Test fails - detections not loaded, or compilation error if `output_dir` field doesn't exist

### Step 3: Implement detection loading on success

```rust
// File: beaker-gui/src/views/directory.rs

// Add output_dir field to DirectoryView struct
pub struct DirectoryView {
    pub directory_path: PathBuf,
    pub images: Vec<ImageState>,
    pub current_image_idx: usize,
    pub selected_detection_idx: Option<usize>,

    // Processing state
    progress_receiver: Option<Receiver<beaker::ProcessingEvent>>,
    cancel_flag: Arc<AtomicBool>,
    output_dir: Option<PathBuf>,  // Add this field

    // ... rest of fields ...
}

// Update new() to initialize output_dir
impl DirectoryView {
    pub fn new(directory_path: PathBuf, image_paths: Vec<PathBuf>) -> Self {
        let images = image_paths
            .into_iter()
            .map(|path| ImageState {
                path,
                status: ProcessingStatus::Waiting,
                detections: Vec::new(),
                thumbnail: None,
            })
            .collect();

        Self {
            directory_path,
            images,
            current_image_idx: 0,
            selected_detection_idx: None,
            progress_receiver: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            output_dir: None,  // Add this
            all_detections: Vec::new(),
            show_good: true,
            show_unknown: true,
            show_bad: true,
        }
    }

    // Update start_processing to store output_dir
    pub fn start_processing(&mut self) {
        use std::sync::mpsc::channel;

        let (tx, rx) = channel();
        self.progress_receiver = Some(rx);

        // Create temp output directory
        let temp_dir = std::env::temp_dir().join(format!("beaker-gui-bulk-{}", std::process::id()));
        self.output_dir = Some(temp_dir.clone());  // Store it

        // ... rest of start_processing method ...
    }

    // Update update_from_event to load detections
    pub fn update_from_event(&mut self, event: beaker::ProcessingEvent) {
        match event {
            beaker::ProcessingEvent::ImageStart { index, .. } => {
                if index < self.images.len() {
                    self.images[index].status = ProcessingStatus::Processing;
                    eprintln!("[DirectoryView] Image {} started processing", index);
                }
            }
            beaker::ProcessingEvent::ImageSuccess { path, index } => {
                if index < self.images.len() {
                    // Find TOML file
                    let stem = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");

                    let toml_path = if let Some(ref output_dir) = self.output_dir {
                        output_dir.join(format!("{}.beaker.toml", stem))
                    } else {
                        path.parent()
                            .map(|p| p.join(format!("{}.beaker.toml", stem)))
                            .unwrap_or_else(|| PathBuf::from(format!("{}.beaker.toml", stem)))
                    };

                    // Load detections
                    let detections = Self::load_detections_from_toml(&toml_path)
                        .unwrap_or_default();

                    let detections_count = detections.len();

                    // Count triage results
                    let (good_count, unknown_count, bad_count) =
                        Self::count_triage_results(&toml_path);

                    self.images[index].detections = detections;
                    self.images[index].status = ProcessingStatus::Success {
                        detections_count,
                        good_count,
                        unknown_count,
                        bad_count,
                        processing_time_ms: 0.0,
                    };

                    eprintln!("[DirectoryView] Image {} completed: {} detections ({} good, {} unknown, {} bad)",
                        index, detections_count, good_count, unknown_count, bad_count);
                }
            }
            beaker::ProcessingEvent::ImageError { index, error, .. } => {
                if index < self.images.len() {
                    self.images[index].status = ProcessingStatus::Error {
                        message: error.clone(),
                    };
                    eprintln!("[DirectoryView] Image {} failed: {}", index, error);
                }
            }
            beaker::ProcessingEvent::StageChange { stage, .. } => {
                eprintln!("[DirectoryView] Stage changed to {:?}", stage);
            }
        }
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): load detection data after successful processing

- Add output_dir field to track temp directory
- Store output_dir when starting processing
- Load detections from TOML on ImageSuccess event
- Count triage results (good/unknown/bad)
- Update ProcessingStatus with actual counts"
```

---

## Task 13: Build Aggregate Detection List After Processing

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_build_aggregate_detection_list() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let img2 = PathBuf::from("/tmp/test/img2.jpg");

    let mut view = DirectoryView::new(dir_path, vec![img1, img2]);

    // Add detections to images
    view.images[0].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.95,
            blur_score: Some(0.1),
            x1: 10.0, y1: 20.0, x2: 100.0, y2: 120.0,
        },
    ];

    view.images[1].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.85,
            blur_score: Some(0.3),
            x1: 15.0, y1: 25.0, x2: 105.0, y2: 125.0,
        },
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.75,
            blur_score: Some(0.5),
            x1: 20.0, y1: 30.0, x2: 110.0, y2: 130.0,
        },
    ];

    view.build_aggregate_detection_list();

    assert_eq!(view.all_detections.len(), 3);
    assert_eq!(view.all_detections[0].image_idx, 0);
    assert_eq!(view.all_detections[1].image_idx, 1);
    assert_eq!(view.all_detections[2].image_idx, 1);
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - `build_aggregate_detection_list` doesn't exist

### Step 3: Implement aggregate list building

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Build flattened list of all detections across all images
    fn build_aggregate_detection_list(&mut self) {
        self.all_detections.clear();

        for (image_idx, image_state) in self.images.iter().enumerate() {
            for (detection_idx, _) in image_state.detections.iter().enumerate() {
                self.all_detections.push(DetectionRef {
                    image_idx,
                    detection_idx,
                });
            }
        }

        eprintln!("[DirectoryView] Built aggregate detection list: {} total detections",
            self.all_detections.len());
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Test passes

### Step 5: Call build_aggregate_detection_list when processing completes

```rust
// File: beaker-gui/src/views/directory.rs (modify poll_events)

impl DirectoryView {
    // ... existing methods ...

    fn poll_events(&mut self) {
        if let Some(ref rx) = self.progress_receiver {
            // Drain all available events (non-blocking)
            while let Ok(event) = rx.try_recv() {
                self.update_from_event(event);
            }
        }

        // Build aggregate list if processing just completed
        let is_processing = self.images.iter().any(|img| {
            matches!(img.status, ProcessingStatus::Waiting | ProcessingStatus::Processing)
        });

        if !is_processing && self.all_detections.is_empty() {
            // Processing complete and we haven't built the list yet
            self.build_aggregate_detection_list();
        }
    }

    // ... rest of implementation ...
}
```

### Step 6: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 7: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): build aggregate detection list after processing

- Add build_aggregate_detection_list() method
- Flatten all detections from all images into single list
- Automatically build list when processing completes
- Store DetectionRef with image_idx and detection_idx"
```

---

## Task 14: Implement Basic Gallery UI

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Document the visual test plan

```rust
// File: beaker-gui/src/views/directory.rs (add comment)

// VISUAL TEST: Gallery UI
// - Should show directory path at top
// - Should show grid of image thumbnails (placeholder boxes for now)
// - Each thumbnail should show filename
// - Should show detection count badge on each thumbnail
// - Clicking thumbnail should update current_image_idx
```

### Step 2: Implement basic gallery UI

```rust
// File: beaker-gui/src/views/directory.rs (modify show_gallery_ui)

impl DirectoryView {
    // ... existing methods ...

    fn show_gallery_ui(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        // Header
        ui.horizontal(|ui| {
            ui.heading(format!("Gallery: {}", self.directory_path.display()));
            ui.label(format!("({} images, {} detections)",
                self.images.len(),
                self.all_detections.len()));
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        // Two-panel layout: Gallery + Current Image
        ui.horizontal(|ui| {
            // Left panel: Thumbnail grid
            egui::ScrollArea::vertical()
                .max_width(300.0)
                .show(ui, |ui| {
                    self.show_thumbnail_grid(ui);
                });

            ui.separator();

            // Right panel: Current image + detections
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.show_current_image(ui);
            });
        });
    }

    fn show_thumbnail_grid(&mut self, ui: &mut egui::Ui) {
        ui.heading("Images");
        ui.add_space(10.0);

        for (idx, image_state) in self.images.iter().enumerate() {
            let filename = image_state.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let is_selected = idx == self.current_image_idx;

            // Thumbnail button (placeholder for now)
            let button_text = if is_selected {
                format!("▶ {}", filename)
            } else {
                format!("  {}", filename)
            };

            let response = ui.selectable_label(is_selected, button_text);

            if response.clicked() {
                self.current_image_idx = idx;
            }

            // Show detection count badge
            let det_count = image_state.detections.len();
            if det_count > 0 {
                ui.label(format!("  {} detections", det_count));
            }

            // Show status icon
            let status_icon = match &image_state.status {
                ProcessingStatus::Success { good_count, unknown_count, bad_count, .. } => {
                    format!("  ✓{} ?{} ✗{}", good_count, unknown_count, bad_count)
                }
                ProcessingStatus::Error { .. } => "  ⚠ Error".to_string(),
                _ => "  ...".to_string(),
            };
            ui.label(egui::RichText::new(status_icon).size(12.0));

            ui.add_space(5.0);
        }
    }

    fn show_current_image(&self, ui: &mut egui::Ui) {
        if self.current_image_idx >= self.images.len() {
            ui.label("No image selected");
            return;
        }

        let current_image = &self.images[self.current_image_idx];
        let filename = current_image.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        ui.heading(filename);
        ui.add_space(10.0);

        // Placeholder for image display
        ui.label("[Image will be displayed here]");
        ui.add_space(20.0);

        // Show detections list
        ui.heading("Detections");
        ui.add_space(10.0);

        if current_image.detections.is_empty() {
            ui.label("No detections");
        } else {
            for (idx, det) in current_image.detections.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}.", idx + 1));
                    ui.label(format!("{}: {:.2}", det.class_name, det.confidence));
                    if let Some(blur) = det.blur_score {
                        ui.label(format!("blur: {:.2}", blur));
                    }
                });
            }
        }
    }

    // ... rest of implementation ...
}
```

### Step 3: Manual visual test

**Run:**
```bash
just build-release
./target/release/beaker-gui

# Open a folder with images
# Wait for processing to complete
# Verify gallery UI displays:
# - Header with directory path and counts
# - Left panel with thumbnail list
# - Clicking thumbnail switches current image
# - Right panel shows current image info and detections
```

**Expected:** Gallery UI displays correctly after processing

### Step 4: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): implement basic gallery UI

- Add show_gallery_ui() method
- Add show_thumbnail_grid() for left panel
- Add show_current_image() for right panel
- Show directory path and counts in header
- Clickable thumbnail list with status badges
- Display current image detections
- Two-panel layout with separators"
```

---

## Task 15: Add Navigation Controls

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_navigate_next_image() {
    let dir_path = PathBuf::from("/tmp/test");
    let images = vec![
        PathBuf::from("/tmp/test/img1.jpg"),
        PathBuf::from("/tmp/test/img2.jpg"),
        PathBuf::from("/tmp/test/img3.jpg"),
    ];
    let mut view = DirectoryView::new(dir_path, images);

    assert_eq!(view.current_image_idx, 0);

    view.navigate_next_image();
    assert_eq!(view.current_image_idx, 1);

    view.navigate_next_image();
    assert_eq!(view.current_image_idx, 2);

    // Should wrap to 0
    view.navigate_next_image();
    assert_eq!(view.current_image_idx, 0);
}

#[test]
fn test_navigate_previous_image() {
    let dir_path = PathBuf::from("/tmp/test");
    let images = vec![
        PathBuf::from("/tmp/test/img1.jpg"),
        PathBuf::from("/tmp/test/img2.jpg"),
        PathBuf::from("/tmp/test/img3.jpg"),
    ];
    let mut view = DirectoryView::new(dir_path, images);

    assert_eq!(view.current_image_idx, 0);

    // Should wrap to last
    view.navigate_previous_image();
    assert_eq!(view.current_image_idx, 2);

    view.navigate_previous_image();
    assert_eq!(view.current_image_idx, 1);
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - navigation methods don't exist

### Step 3: Implement navigation methods

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Navigate to next image (wraps around)
    fn navigate_next_image(&mut self) {
        if self.images.is_empty() {
            return;
        }
        self.current_image_idx = (self.current_image_idx + 1) % self.images.len();
    }

    /// Navigate to previous image (wraps around)
    fn navigate_previous_image(&mut self) {
        if self.images.is_empty() {
            return;
        }
        if self.current_image_idx == 0 {
            self.current_image_idx = self.images.len() - 1;
        } else {
            self.current_image_idx -= 1;
        }
    }

    /// Jump to specific image by index
    fn jump_to_image(&mut self, idx: usize) {
        if idx < self.images.len() {
            self.current_image_idx = idx;
        }
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 5: Add navigation buttons to UI

```rust
// File: beaker-gui/src/views/directory.rs (modify show_current_image)

impl DirectoryView {
    // ... existing methods ...

    fn show_current_image(&mut self, ui: &mut egui::Ui) {
        if self.current_image_idx >= self.images.len() {
            ui.label("No image selected");
            return;
        }

        let current_image = &self.images[self.current_image_idx];
        let filename = current_image.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Navigation controls
        ui.horizontal(|ui| {
            if ui.button("← Previous").clicked() {
                self.navigate_previous_image();
            }

            ui.label(format!("{} / {}", self.current_image_idx + 1, self.images.len()));

            if ui.button("Next →").clicked() {
                self.navigate_next_image();
            }
        });

        ui.add_space(10.0);
        ui.heading(filename);
        ui.add_space(10.0);

        // Placeholder for image display
        ui.label("[Image will be displayed here]");
        ui.add_space(20.0);

        // Show detections list
        ui.heading("Detections");
        ui.add_space(10.0);

        if current_image.detections.is_empty() {
            ui.label("No detections");
        } else {
            for (idx, det) in current_image.detections.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}.", idx + 1));
                    ui.label(format!("{}: {:.2}", det.class_name, det.confidence));
                    if let Some(blur) = det.blur_score {
                        ui.label(format!("blur: {:.2}", blur));
                    }
                });
            }
        }
    }

    // ... rest of implementation ...
}
```

### Step 6: Add keyboard shortcuts

```rust
// File: beaker-gui/src/views/directory.rs (modify show_gallery_ui)

impl DirectoryView {
    // ... existing methods ...

    fn show_gallery_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Handle keyboard shortcuts
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                self.navigate_next_image();
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.navigate_previous_image();
            }
        });

        // ... rest of show_gallery_ui implementation ...
    }

    // ... rest of implementation ...
}
```

### Step 7: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 8: Manual visual test

**Run:**
```bash
just build-release
./target/release/beaker-gui

# Open a folder
# After processing, test:
# - Click "Previous" and "Next" buttons
# - Press arrow keys to navigate
# - Verify counter updates (e.g., "2 / 47")
```

**Expected:** Navigation works smoothly

### Step 9: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): add image navigation controls

- Add navigate_next_image() with wraparound
- Add navigate_previous_image() with wraparound
- Add jump_to_image() helper
- Add Previous/Next buttons in UI
- Add arrow key shortcuts (←/→)
- Display current position (e.g., 2 / 47)"
```

---

## Task 16: Display Aggregate Detection List Sidebar

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Document visual test

```rust
// File: beaker-gui/src/views/directory.rs (add comment)

// VISUAL TEST: Aggregate Detection Sidebar
// - Should show "All Detections (N)" header
// - Should list all detections from all images
// - Format: "filename - class_name #N: confidence"
// - Clicking detection should jump to that image
// - Selected detection should be highlighted
```

### Step 2: Implement aggregate detection sidebar

```rust
// File: beaker-gui/src/views/directory.rs (modify show_gallery_ui)

impl DirectoryView {
    // ... existing methods ...

    fn show_gallery_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Handle keyboard shortcuts
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                self.navigate_next_image();
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.navigate_previous_image();
            }
        });

        // Header
        ui.horizontal(|ui| {
            ui.heading(format!("Gallery: {}", self.directory_path.display()));
            ui.label(format!("({} images, {} detections)",
                self.images.len(),
                self.all_detections.len()));
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        // Three-panel layout: Thumbnails + Current Image + Aggregate Detections
        ui.horizontal(|ui| {
            // Left panel: Thumbnail grid
            egui::ScrollArea::vertical()
                .max_width(250.0)
                .show(ui, |ui| {
                    self.show_thumbnail_grid(ui);
                });

            ui.separator();

            // Middle panel: Current image + detections
            egui::ScrollArea::vertical()
                .max_width(600.0)
                .show(ui, |ui| {
                    self.show_current_image(ui);
                });

            ui.separator();

            // Right panel: Aggregate detection list
            egui::ScrollArea::vertical()
                .show(ui, |ui| {
                    self.show_aggregate_detection_list(ui);
                });
        });
    }

    fn show_aggregate_detection_list(&mut self, ui: &mut egui::Ui) {
        ui.heading(format!("All Detections ({})", self.all_detections.len()));
        ui.add_space(10.0);

        if self.all_detections.is_empty() {
            ui.label("No detections");
            return;
        }

        for (flat_idx, det_ref) in self.all_detections.iter().enumerate() {
            let image_state = &self.images[det_ref.image_idx];
            let detection = &image_state.detections[det_ref.detection_idx];

            let filename = image_state.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let is_selected = Some(flat_idx) == self.selected_detection_idx;

            let label_text = format!(
                "{} - {} #{}: {:.2}",
                filename,
                detection.class_name,
                det_ref.detection_idx + 1,
                detection.confidence
            );

            let response = ui.selectable_label(is_selected, label_text);

            if response.clicked() {
                // Jump to this image and select detection
                self.current_image_idx = det_ref.image_idx;
                self.selected_detection_idx = Some(flat_idx);
            }

            // Show quality info if available
            if let Some(blur) = detection.blur_score {
                ui.label(format!("  blur: {:.2}", blur));
            }
        }
    }

    // ... rest of implementation ...
}
```

### Step 3: Manual visual test

**Run:**
```bash
just build-release
./target/release/beaker-gui

# Open folder with multiple images
# After processing:
# - Verify right sidebar shows "All Detections (N)"
# - Verify all detections listed with format "filename - class #N: conf"
# - Click a detection from different image
# - Verify middle panel switches to that image
```

**Expected:** Aggregate sidebar works correctly

### Step 4: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): add aggregate detection list sidebar

- Add show_aggregate_detection_list() method
- Display all detections from all images in right panel
- Format: filename - class #N: confidence
- Clicking detection jumps to that image
- Highlight selected detection
- Show blur score if available
- Three-panel layout: thumbnails | image | all detections"
```

---

## Task 17: Add Detection Navigation (Next/Previous Detection)

**Files:**
- Modify: beaker-gui/src/views/directory.rs

### Step 1: Write the failing test

```rust
// File: beaker-gui/src/views/directory.rs (add to tests)

#[test]
fn test_navigate_next_detection() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let img2 = PathBuf::from("/tmp/test/img2.jpg");

    let mut view = DirectoryView::new(dir_path, vec![img1.clone(), img2.clone()]);

    // Add detections
    view.images[0].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.95,
            blur_score: None,
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
        },
    ];

    view.images[1].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.85,
            blur_score: None,
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
        },
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.75,
            blur_score: None,
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
        },
    ];

    view.build_aggregate_detection_list();

    // Start at detection 0
    view.selected_detection_idx = Some(0);

    view.navigate_next_detection();
    assert_eq!(view.selected_detection_idx, Some(1));
    assert_eq!(view.current_image_idx, 1); // Should jump to image 1

    view.navigate_next_detection();
    assert_eq!(view.selected_detection_idx, Some(2));
    assert_eq!(view.current_image_idx, 1); // Still on image 1

    // Wrap to beginning
    view.navigate_next_detection();
    assert_eq!(view.selected_detection_idx, Some(0));
    assert_eq!(view.current_image_idx, 0); // Back to image 0
}

#[test]
fn test_navigate_previous_detection() {
    let dir_path = PathBuf::from("/tmp/test");
    let img1 = PathBuf::from("/tmp/test/img1.jpg");
    let img2 = PathBuf::from("/tmp/test/img2.jpg");

    let mut view = DirectoryView::new(dir_path, vec![img1.clone(), img2.clone()]);

    // Add detections
    view.images[0].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.95,
            blur_score: None,
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
        },
    ];

    view.images[1].detections = vec![
        crate::views::detection::Detection {
            class_name: "head".to_string(),
            confidence: 0.85,
            blur_score: None,
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
        },
    ];

    view.build_aggregate_detection_list();

    // Start at detection 0
    view.selected_detection_idx = Some(0);

    // Wrap to last
    view.navigate_previous_detection();
    assert_eq!(view.selected_detection_idx, Some(1));
    assert_eq!(view.current_image_idx, 1); // Should jump to image 1
}
```

### Step 2: Run test to verify it fails

**Run:**
```bash
just test
```

**Expected:** Compilation error - navigation methods don't exist

### Step 3: Implement detection navigation

```rust
// File: beaker-gui/src/views/directory.rs (add to impl DirectoryView)

impl DirectoryView {
    // ... existing methods ...

    /// Navigate to next detection (wraps around)
    fn navigate_next_detection(&mut self) {
        if self.all_detections.is_empty() {
            return;
        }

        let current_idx = self.selected_detection_idx.unwrap_or(0);
        let next_idx = (current_idx + 1) % self.all_detections.len();

        self.selected_detection_idx = Some(next_idx);

        // Jump to the image containing this detection
        self.current_image_idx = self.all_detections[next_idx].image_idx;
    }

    /// Navigate to previous detection (wraps around)
    fn navigate_previous_detection(&mut self) {
        if self.all_detections.is_empty() {
            return;
        }

        let current_idx = self.selected_detection_idx.unwrap_or(0);
        let prev_idx = if current_idx == 0 {
            self.all_detections.len() - 1
        } else {
            current_idx - 1
        };

        self.selected_detection_idx = Some(prev_idx);

        // Jump to the image containing this detection
        self.current_image_idx = self.all_detections[prev_idx].image_idx;
    }

    // ... rest of implementation ...
}
```

### Step 4: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 5: Add UI controls for detection navigation

```rust
// File: beaker-gui/src/views/directory.rs (modify show_aggregate_detection_list)

impl DirectoryView {
    // ... existing methods ...

    fn show_aggregate_detection_list(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(format!("All Detections ({})", self.all_detections.len()));
        });

        // Navigation controls for detections
        if !self.all_detections.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("← Prev Detection").clicked() {
                    self.navigate_previous_detection();
                }

                if let Some(idx) = self.selected_detection_idx {
                    ui.label(format!("{} / {}", idx + 1, self.all_detections.len()));
                }

                if ui.button("Next Detection →").clicked() {
                    self.navigate_next_detection();
                }
            });
        }

        ui.add_space(10.0);

        if self.all_detections.is_empty() {
            ui.label("No detections");
            return;
        }

        // ... rest of show_aggregate_detection_list ...
    }

    // ... rest of implementation ...
}
```

### Step 6: Add keyboard shortcuts (J/K)

```rust
// File: beaker-gui/src/views/directory.rs (modify show_gallery_ui)

impl DirectoryView {
    // ... existing methods ...

    fn show_gallery_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Handle keyboard shortcuts
        ctx.input(|i| {
            // Image navigation
            if i.key_pressed(egui::Key::ArrowRight) {
                self.navigate_next_image();
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.navigate_previous_image();
            }

            // Detection navigation (J/K like vim)
            if i.key_pressed(egui::Key::J) {
                self.navigate_next_detection();
            }
            if i.key_pressed(egui::Key::K) {
                self.navigate_previous_detection();
            }
        });

        // ... rest of show_gallery_ui ...
    }

    // ... rest of implementation ...
}
```

### Step 7: Run test to verify it passes

**Run:**
```bash
just test
```

**Expected:** Tests pass

### Step 8: Manual visual test

**Run:**
```bash
just build-release
./target/release/beaker-gui

# Open folder with multiple images with detections
# Test:
# - Click "Prev Detection" / "Next Detection" buttons
# - Verify image switches when detection is on different image
# - Press J/K keys for detection navigation
# - Verify counter updates
```

**Expected:** Detection navigation works, jumping between images

### Step 9: Commit

```bash
git add beaker-gui/src/views/directory.rs
git commit -m "feat(gui): add detection navigation

- Add navigate_next_detection() with image jumping
- Add navigate_previous_detection() with wraparound
- Add Prev/Next Detection buttons in sidebar
- Add J/K keyboard shortcuts for detection navigation
- Display detection position (e.g., 5 / 89)
- Automatically jump to image containing selected detection"
```

---

## Task 18: Run Full CI and Fix Any Issues

**Files:**
- All modified files

### Step 1: Run full CI locally

**Run:**
```bash
just ci
```

**Expected:** All checks pass (format, lint, build, test)

### Step 2: Fix any issues

If CI fails, fix the specific issues:

- **Format issues:** Run `just fmt`
- **Clippy warnings:** Fix warnings, or add `#[allow(clippy::...)]` if justified
- **Test failures:** Debug and fix the failing tests
- **Build errors:** Fix compilation errors

### Step 3: Re-run CI

**Run:**
```bash
just ci
```

**Expected:** All checks pass

### Step 4: Commit fixes (if any)

```bash
git add -A
git commit -m "fix: resolve CI issues for bulk directory mode"
```

---

## Task 19: Manual Integration Test

**Files:**
- N/A (manual testing)

### Step 1: Create test dataset

```bash
# Create test folder with sample images
mkdir -p /tmp/beaker-bulk-test
cp beaker/tests/data/example-no-bg.png /tmp/beaker-bulk-test/img1.png
cp beaker/tests/data/example-no-bg.png /tmp/beaker-bulk-test/img2.png
cp beaker/tests/data/example-no-bg.png /tmp/beaker-bulk-test/img3.png

# Optional: Copy more real bird images if available
```

### Step 2: Test complete workflow

**Run:**
```bash
just build-release
./target/release/beaker-gui
```

**Test checklist:**
1. ✅ Welcome screen displays
2. ✅ Click "Open..." or drag folder
3. ✅ Directory processing starts automatically
4. ✅ Progress UI shows:
   - Overall progress bar
   - Current stage (Quality/Detection)
   - Per-image status list with icons
   - Cancel button
5. ✅ Processing completes (all images show ✓ or ⚠)
6. ✅ Gallery UI appears:
   - Left: Thumbnail list with status badges
   - Middle: Current image with detections
   - Right: Aggregate detection list
7. ✅ Navigation works:
   - Prev/Next image buttons
   - Arrow keys (←/→)
   - Clicking thumbnails
8. ✅ Detection navigation works:
   - Prev/Next detection buttons
   - J/K keys
   - Clicking detection in sidebar jumps to image
9. ✅ Recent files updated (folder appears in recent)

### Step 3: Document any issues

If any issues found, create follow-up tasks in a separate commit.

### Step 4: Mark complete

**Run:**
```bash
echo "Manual integration test passed" > /tmp/beaker-test-result.txt
```

---

## Task 20: Commit Final Changes and Create Documentation

**Files:**
- Modify: beaker-gui/README.md (if exists) or create it

### Step 1: Document the new feature

```markdown
# Beaker GUI

Bird image analysis tool with graphical interface.

## Features

### Single Image Mode
- Open single image for analysis
- View detections with bounding boxes
- Inspect quality metrics

### Bulk/Directory Mode (NEW)
- Process entire directories of images
- Live progress tracking with per-image status
- Gallery view with thumbnails
- Aggregate detection list across all images
- Navigation: Next/Previous image (← →)
- Detection navigation: J/K to jump between detections
- Automatic processing on folder open

## Usage

### Open Single Image
```bash
beaker-gui --image path/to/image.jpg
```

### Open Folder (Bulk Mode)
1. Launch GUI: `beaker-gui`
2. Click "Open..." or drag & drop folder
3. Wait for processing to complete
4. Browse results in gallery view

### Keyboard Shortcuts
- `←` / `→`: Navigate between images
- `J` / `K`: Navigate between detections (across all images)
- `Cmd+O` (macOS) or `Ctrl+O`: Open file dialog

## Architecture

- `WelcomeView`: Welcome screen with file/folder opening
- `DetectionView`: Single image analysis view
- `DirectoryView`: Bulk processing with gallery (NEW)
  - Background thread runs beaker detection
  - Progress events via `std::sync::mpsc` channels
  - Aggregate detection list across all images

## Development

Build:
```bash
just build-release
```

Run tests:
```bash
just test
```

Run CI locally:
```bash
just ci
```
```

### Step 2: Create the file

**Run:**
```bash
# If README doesn't exist, create it with the content above
# If it exists, update it to add Bulk/Directory Mode section
```

### Step 3: Commit documentation

```bash
git add beaker-gui/README.md
git commit -m "docs: document bulk/directory mode feature

- Add Bulk/Directory Mode section to README
- Document keyboard shortcuts
- Document architecture changes
- Add usage instructions for folder processing"
```

---

## Task 21: Final Commit and Push

**Files:**
- All changes

### Step 1: Verify all tests pass

**Run:**
```bash
just ci
```

**Expected:** All checks pass

### Step 2: Review all changes

**Run:**
```bash
git log --oneline origin/main..HEAD
git diff origin/main..HEAD --stat
```

**Expected:** ~20 commits for the implementation

### Step 3: Push to remote branch

**Run:**
```bash
git push -u origin claude/implementation-plan-detection-011CUWKfh3FWbULFuympfqaQ
```

**Expected:** Push succeeds

### Step 4: Verify CI passes on remote

Check GitHub Actions or CI system to ensure all checks pass.

---

## Summary

**Implemented features:**
✅ DirectoryView data structures and module registration
✅ Background processing thread with progress events
✅ Progress UI with per-image status tracking
✅ Detection data loading from TOML
✅ Aggregate detection list across all images
✅ Gallery view with thumbnail grid
✅ Image navigation (Prev/Next, ← →)
✅ Detection navigation (J/K, cross-image jumping)
✅ Cancellation support
✅ Integration with folder opening dialog

**Total effort:** ~800 LOC across 21 tasks, each 2-5 minutes

**Testing:**
- Unit tests for core logic (navigation, event handling, TOML parsing)
- Manual integration testing
- CI validation

**Next steps (future work):**
- Load and display actual images (not just placeholders)
- Generate thumbnails for gallery
- Display bounding boxes on current image
- Add filtering by quality (good/unknown/bad)
- Export functionality
- Heatmap visualization

---

## Troubleshooting

**Issue: Compilation errors with `beaker::ProcessingEvent`**
- **Solution:** Ensure beaker library changes are complete and ProcessingEvent is exported in `beaker/src/lib.rs`

**Issue: Tests fail due to missing ONNX models**
- **Solution:** Tests that actually run detection will download models on first run. Ensure network access.

**Issue: GUI freezes during processing**
- **Solution:** Verify background thread is spawned correctly and ctx.request_repaint() is called in poll_events()

**Issue: Detection counts are all zero**
- **Solution:** Check TOML parsing - verify path to .beaker.toml is correct and file exists

**Issue: Navigation doesn't work**
- **Solution:** Verify all_detections list is built after processing completes (check poll_events logic)

---

## Performance Notes

**Expected performance:**
- Processing: ~1-2 seconds per image (quality + detection)
- Directory of 50 images: ~1-2 minutes total
- UI responsiveness: Should remain smooth during processing (background thread)
- Memory: ~100MB for model cache + image data

**Optimization opportunities (future):**
- Parallel processing (process multiple images concurrently)
- Thumbnail generation and caching
- Lazy loading of images (only load current + adjacent)
- Incremental TOML parsing (parse as images complete, not after)
