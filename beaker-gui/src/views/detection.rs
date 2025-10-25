use anyhow::Result;
use egui::ColorImage;
use image::DynamicImage;

pub struct DetectionView {
    image: Option<DynamicImage>,
    detections: Vec<Detection>,
    texture: Option<egui::TextureHandle>,
    selected_detection: Option<usize>,
}

// Simplified detection structure for GUI
#[derive(Clone)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    #[allow(dead_code)] // May be used for highlighting in future
    pub x1: f32,
    #[allow(dead_code)]
    pub y1: f32,
    #[allow(dead_code)]
    pub x2: f32,
    #[allow(dead_code)]
    pub y2: f32,
    pub blur_score: Option<f32>,
}

impl DetectionView {
    pub fn new(image_path: &str) -> Result<Self> {
        // Run detection and get both the bounding box image and detection data
        let (bbox_image_path, detections) = Self::run_detection(image_path)?;

        // Load the bounding box image (already has boxes drawn by beaker lib)
        let image = image::open(&bbox_image_path)?;

        Ok(Self {
            image: Some(image),
            detections,
            texture: None,
            selected_detection: None,
        })
    }

    /// Get the detections for testing
    #[allow(dead_code)]
    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }

    fn run_detection(image_path: &str) -> Result<(String, Vec<Detection>)> {
        use std::collections::HashSet;

        // Create temp directory for output
        let temp_dir = std::env::temp_dir().join(format!("beaker-gui-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        eprintln!("Running detection for: {}", image_path);
        eprintln!("Output directory: {}", temp_dir.display());

        let base_config = beaker::config::BaseModelConfig {
            sources: vec![image_path.to_string()],
            device: "auto".to_string(),
            output_dir: Some(temp_dir.to_str().unwrap().to_string()),
            skip_metadata: false,
            strict: true,
            force: true,
        };

        let config = beaker::config::DetectionConfig {
            base: base_config,
            confidence: 0.5,
            crop_classes: HashSet::new(),
            bounding_box: true, // Enable bounding box output
            model_path: None,
            model_url: None,
            model_checksum: None,
            quality_results: None,
        };

        // Run detection
        beaker::detection::run_detection(config)?;

        // List files created in temp directory
        eprintln!("Files in output directory:");
        if let Ok(entries) = std::fs::read_dir(&temp_dir) {
            for entry in entries.flatten() {
                eprintln!("  - {}", entry.file_name().to_string_lossy());
            }
        }

        // Find the bounding box image
        let image_stem = std::path::Path::new(image_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        // Try to find the actual bounding box file
        let bbox_image_path = temp_dir.join(format!("{}_bounding-box.jpg", image_stem));
        eprintln!(
            "Looking for bounding box image: {}",
            bbox_image_path.display()
        );

        if !bbox_image_path.exists() {
            anyhow::bail!(
                "Bounding box image not found at: {}",
                bbox_image_path.display()
            );
        }

        // Read the TOML metadata to get detection info
        let toml_path = temp_dir.join(format!("{}.beaker.toml", image_stem));
        eprintln!("Looking for TOML: {}", toml_path.display());

        let toml_data = std::fs::read_to_string(&toml_path)?;

        // Parse TOML to extract detection info
        let toml_value: toml::Value = toml::from_str(&toml_data)?;
        let mut detections = Vec::new();

        // Detections are under [detect] section with core results flattened
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
                        .unwrap_or("unknown");
                    let confidence = det_table
                        .get("confidence")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;

                    let blur_score = det_table
                        .get("quality")
                        .and_then(|q| q.as_table())
                        .and_then(|q| q.get("blur_score"))
                        .and_then(|v| v.as_float())
                        .map(|v| v as f32);

                    let x1 = det_table
                        .get("x1")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let y1 = det_table
                        .get("y1")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let x2 = det_table
                        .get("x2")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let y2 = det_table
                        .get("y2")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;

                    detections.push(Detection {
                        class_name: class_name.to_string(),
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        blur_score,
                    });
                }
            }
        }

        eprintln!("Found {} detections", detections.len());

        Ok((bbox_image_path.to_str().unwrap().to_string(), detections))
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // RUNTIME ASSERT: View must have an image
        assert!(
            self.image.is_some(),
            "DetectionView invariant violated: no image loaded"
        );

        egui::SidePanel::right("detections_panel")
            .default_width(crate::style::DETECTION_PANEL_WIDTH)
            .show_inside(ui, |ui| {
                self.show_detections_list(ui);
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_image_with_bboxes(ui, ctx);
        });
    }

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

        ui.label(format!("Found {} objects", self.detections.len()));
        ui.separator();

        for (idx, det) in self.detections.iter().enumerate() {
            let is_selected = self.selected_detection == Some(idx);

            // RUNTIME ASSERT: Confidence must be valid
            assert!(
                det.confidence >= 0.0 && det.confidence <= 1.0,
                "Detection confidence out of range: {}",
                det.confidence
            );

            // Styled card for each detection
            egui::Frame::none()
                .fill(if is_selected {
                    egui::Color32::from_rgb(230, 240, 255)
                } else {
                    egui::Color32::WHITE
                })
                .rounding(6.0)
                .inner_margin(12.0)
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(200)))
                .show(ui, |ui| {
                    if ui.selectable_label(false, &det.class_name).clicked() {
                        self.selected_detection = Some(idx);
                    }
                    ui.label(format!("Confidence: {:.2}", det.confidence));
                    if let Some(blur) = det.blur_score {
                        ui.label(format!("Blur: {:.2}", blur));
                    }
                });

            ui.add_space(8.0);
        }
    }

    fn show_image_with_bboxes(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(img) = &self.image {
            // RUNTIME ASSERT: Image dimensions must be non-zero
            assert!(
                img.width() > 0 && img.height() > 0,
                "DetectionView invariant violated: image has zero dimensions"
            );

            // Load texture on first render
            if self.texture.is_none() {
                let rendered = self.render_image_with_bboxes(img);

                // RUNTIME ASSERT: Rendered image matches original dimensions
                assert_eq!(rendered.size[0], img.width() as usize);
                assert_eq!(rendered.size[1], img.height() as usize);

                self.texture =
                    Some(ctx.load_texture("detection_view", rendered, Default::default()));
            }

            if let Some(texture) = &self.texture {
                // Scale image to fit panel
                let available_size = ui.available_size();
                let image_aspect = texture.size()[0] as f32 / texture.size()[1] as f32;
                let panel_aspect = available_size.x / available_size.y;

                let size = if image_aspect > panel_aspect {
                    egui::vec2(available_size.x, available_size.x / image_aspect)
                } else {
                    egui::vec2(available_size.y * image_aspect, available_size.y)
                };

                ui.centered_and_justified(|ui| {
                    ui.image((texture.id(), size));
                });
            }
        }
    }

    fn render_image_with_bboxes(&self, img: &DynamicImage) -> ColorImage {
        // Image already has bounding boxes drawn by beaker lib
        // Just convert to egui ColorImage format
        let img_rgba = img.to_rgba8();
        let size = [img_rgba.width() as usize, img_rgba.height() as usize];
        let pixels = img_rgba.into_raw();
        ColorImage::from_rgba_unmultiplied(size, &pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_parsing_detections() {
        // Test that we correctly parse detections from beaker TOML format
        // The TOML structure has detections under [detect] section with core results flattened
        let toml_data = r#"
[detect]
model_version = "test-v1.0.0"
input_img_width = 1024
input_img_height = 768

[[detect.detections]]
class_name = "head"
confidence = 0.95
x1 = 100.0
y1 = 150.0
x2 = 200.0
y2 = 250.0

[[detect.detections]]
class_name = "head"
confidence = 0.87
x1 = 300.0
y1 = 350.0
x2 = 400.0
y2 = 450.0

[detect.detections.quality]
blur_score = 0.12

[detect.config]
confidence = 0.5

[detect.execution]
timestamp = "2025-10-25T12:00:00Z"
beaker_version = "0.1.0"
"#;

        // Parse TOML
        let toml_value: toml::Value = toml::from_str(toml_data).unwrap();
        let mut detections = Vec::new();

        // Extract detections using the same logic as run_detection
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
                        .unwrap_or("unknown");
                    let confidence = det_table
                        .get("confidence")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;

                    let blur_score = det_table
                        .get("quality")
                        .and_then(|q| q.as_table())
                        .and_then(|q| q.get("blur_score"))
                        .and_then(|v| v.as_float())
                        .map(|v| v as f32);

                    let x1 = det_table
                        .get("x1")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let y1 = det_table
                        .get("y1")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let x2 = det_table
                        .get("x2")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;
                    let y2 = det_table
                        .get("y2")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0) as f32;

                    detections.push(Detection {
                        class_name: class_name.to_string(),
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        blur_score,
                    });
                }
            }
        }

        // Verify we parsed the detections correctly
        assert_eq!(detections.len(), 2, "Should parse 2 detections");
        assert_eq!(detections[0].class_name, "head");
        assert_eq!(detections[0].confidence, 0.95);
        assert_eq!(detections[0].x1, 100.0);
        assert_eq!(detections[0].y1, 150.0);
        assert_eq!(detections[0].x2, 200.0);
        assert_eq!(detections[0].y2, 250.0);
        assert!(detections[0].blur_score.is_none());

        assert_eq!(detections[1].class_name, "head");
        assert_eq!(detections[1].confidence, 0.87);
        assert_eq!(detections[1].x1, 300.0);
        assert_eq!(detections[1].y1, 350.0);
    }
}
