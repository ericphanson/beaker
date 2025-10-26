// File: beaker-gui/src/views/directory.rs

// VISUAL TEST: Gallery UI
// - Should show directory path at top
// - Should show grid of image thumbnails (placeholder boxes for now)
// - Each thumbnail should show filename
// - Should show detection count badge on each thumbnail
// - Clicking thumbnail should update current_image_idx

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
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
    output_dir: Option<PathBuf>,

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
            output_dir: None,
            all_detections: Vec::new(),
            show_good: true,
            show_unknown: true,
            show_bad: true,
        }
    }

    /// Start background processing of all images
    pub fn start_processing(&mut self) {
        use std::sync::mpsc::channel;

        let (tx, rx) = channel();
        self.progress_receiver = Some(rx);

        // Create temp output directory
        let temp_dir = std::env::temp_dir()
            .join(format!("beaker-gui-bulk-{}", std::process::id()));
        self.output_dir = Some(temp_dir.clone());

        // Collect paths to process
        let image_paths: Vec<PathBuf> = self.images.iter().map(|img| img.path.clone()).collect();
        let cancel_flag = Arc::clone(&self.cancel_flag);

        // Spawn background thread
        std::thread::spawn(move || {
            eprintln!(
                "[DirectoryView] Background thread started, processing {} images",
                image_paths.len()
            );

            // Create temp output directory
            if let Err(e) = std::fs::create_dir_all(&temp_dir) {
                eprintln!("[DirectoryView] ERROR: Failed to create temp dir: {}", e);
                return;
            }

            // Build detection config
            let sources: Vec<String> = image_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            let base_config = beaker::config::BaseModelConfig {
                sources,
                device: "auto".to_string(),
                output_dir: Some(temp_dir.to_str().unwrap().to_string()),
                skip_metadata: false,
                strict: false, // Don't fail on single image errors
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
            match beaker::detection::run_detection_with_options(
                config,
                Some(tx.clone()),
                Some(cancel_flag),
            ) {
                Ok(count) => {
                    eprintln!(
                        "[DirectoryView] Processing complete: {} images processed",
                        count
                    );
                }
                Err(e) => {
                    eprintln!("[DirectoryView] ERROR: Processing failed: {}", e);
                }
            }
        });
    }

    /// Update state based on progress event from background thread
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

    /// Poll for progress events from background thread
    fn poll_events(&mut self) {
        // Collect events first to avoid borrow checker issues
        let mut events = Vec::new();
        if let Some(ref rx) = self.progress_receiver {
            // Drain all available events (non-blocking)
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }

        // Process collected events
        for event in events {
            self.update_from_event(event);
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

    /// Calculate progress statistics
    fn calculate_progress_stats(&self) -> (usize, usize, Option<usize>) {
        let completed = self
            .images
            .iter()
            .filter(|img| {
                matches!(
                    img.status,
                    ProcessingStatus::Success { .. } | ProcessingStatus::Error { .. }
                )
            })
            .count();

        let total = self.images.len();

        let processing_idx = self
            .images
            .iter()
            .position(|img| matches!(img.status, ProcessingStatus::Processing));

        (completed, total, processing_idx)
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // Poll progress events
        self.poll_events();

        // Request repaint if processing is active
        let is_processing = self.images.iter().any(|img| {
            matches!(
                img.status,
                ProcessingStatus::Waiting | ProcessingStatus::Processing
            )
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
            ui.add(
                egui::ProgressBar::new(progress)
                    .text(format!("{}/{} ({:.0}%)", completed, total, progress * 100.0))
                    .desired_width(600.0),
            );

            ui.add_space(10.0);

            // Currently processing
            if let Some(idx) = processing_idx {
                let filename = self.images[idx]
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                ui.label(format!("Currently processing: {}", filename));
            }

            ui.add_space(20.0);

            // Cancel button
            if ui.button("Cancel").clicked() {
                use std::sync::atomic::Ordering;
                self.cancel_flag.store(true, Ordering::Relaxed);
            }

            ui.add_space(30.0);

            // Image list with status
            ui.label(egui::RichText::new("Images:").size(16.0));
            ui.add_space(10.0);

            egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                for img in &self.images {
                    let filename = img
                        .path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    ui.horizontal(|ui| {
                        match &img.status {
                            ProcessingStatus::Waiting => {
                                ui.label(egui::RichText::new("⏸").color(egui::Color32::GRAY).size(16.0));
                                ui.label(format!("{}: Waiting...", filename));
                            }
                            ProcessingStatus::Processing => {
                                ui.label(egui::RichText::new("⏳").color(egui::Color32::BLUE).size(16.0));
                                ui.label(format!("{}: Processing...", filename));
                            }
                            ProcessingStatus::Success {
                                detections_count,
                                good_count,
                                unknown_count,
                                ..
                            } => {
                                ui.label(egui::RichText::new("✓").color(egui::Color32::GREEN).size(16.0));
                                ui.label(format!(
                                    "{}: {} detections ({} good, {} unknown)",
                                    filename, detections_count, good_count, unknown_count
                                ));
                            }
                            ProcessingStatus::Error { message } => {
                                ui.label(egui::RichText::new("⚠").color(egui::Color32::RED).size(16.0));
                                ui.label(format!("{}: {}", filename, message));
                            }
                        }
                    });
                }
            });
        });
    }

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

    fn show_current_image(&mut self, ui: &mut egui::Ui) {
        if self.current_image_idx >= self.images.len() {
            ui.label("No image selected");
            return;
        }

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

        // Get current image data (after navigation buttons which may have changed index)
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

    #[test]
    fn test_start_processing_creates_thread() {
        use std::fs::File;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let img1 = temp_dir.path().join("img1.jpg");
        File::create(&img1).unwrap();

        let mut view = DirectoryView::new(temp_dir.path().to_path_buf(), vec![img1.clone()]);

        view.start_processing();

        // Should have receiver set up
        assert!(view.progress_receiver.is_some());
    }

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
            ProcessingStatus::Success { .. } => {}
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
            }
            _ => panic!("Expected Error status"),
        }
    }

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

    #[test]
    fn test_load_detection_data_from_toml() {
        use tempfile::TempDir;
        use std::fs;

        let temp_dir = TempDir::new().unwrap();
        let _img_path = temp_dir.path().join("test.jpg");
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
}
