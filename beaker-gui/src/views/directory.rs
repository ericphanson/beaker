// File: beaker-gui/src/views/directory.rs

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
            eprintln!(
                "[DirectoryView] Background thread started, processing {} images",
                image_paths.len()
            );

            // Create temp output directory
            let temp_dir = std::env::temp_dir()
                .join(format!("beaker-gui-bulk-{}", std::process::id()));
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

    fn show_gallery_ui(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.label("Gallery view - coming soon");
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
}
