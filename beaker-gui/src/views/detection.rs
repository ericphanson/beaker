use egui::ColorImage;
use image::DynamicImage;
use anyhow::Result;

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
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub blur_score: Option<f32>,
}

impl DetectionView {
    pub fn new(image_path: &str) -> Result<Self> {
        let image = image::open(image_path)?;

        // Run detection using beaker lib
        let detections = Self::run_detection(image_path)?;

        Ok(Self {
            image: Some(image),
            detections,
            texture: None,
            selected_detection: None,
        })
    }

    fn run_detection(_image_path: &str) -> Result<Vec<Detection>> {
        // TODO: Integrate with beaker detection once API is stabilized
        // For MVP, return mock data to demonstrate GUI functionality

        // Mock detections for demonstration
        let detections = vec![
            Detection {
                class_name: "bird".to_string(),
                confidence: 0.95,
                x1: 100.0,
                y1: 100.0,
                x2: 300.0,
                y2: 300.0,
                blur_score: Some(0.15),
            },
            Detection {
                class_name: "bird".to_string(),
                confidence: 0.87,
                x1: 400.0,
                y1: 150.0,
                x2: 600.0,
                y2: 350.0,
                blur_score: Some(0.22),
            },
        ];

        Ok(detections)
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // RUNTIME ASSERT: View must have an image
        assert!(self.image.is_some(), "DetectionView invariant violated: no image loaded");

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
            assert!(selected < self.detections.len(),
                    "DetectionView invariant violated: selected_detection {} >= detection count {}",
                    selected, self.detections.len());
        }

        ui.heading("Detections");

        ui.label(format!("Found {} objects", self.detections.len()));
        ui.separator();

        for (idx, det) in self.detections.iter().enumerate() {
            let is_selected = self.selected_detection == Some(idx);

            // RUNTIME ASSERT: Confidence must be valid
            assert!(det.confidence >= 0.0 && det.confidence <= 1.0,
                    "Detection confidence out of range: {}", det.confidence);

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
            assert!(img.width() > 0 && img.height() > 0,
                    "DetectionView invariant violated: image has zero dimensions");

            // Load texture on first render
            if self.texture.is_none() {
                let rendered = self.render_image_with_bboxes(img);

                // RUNTIME ASSERT: Rendered image matches original dimensions
                assert_eq!(rendered.size[0], img.width() as usize);
                assert_eq!(rendered.size[1], img.height() as usize);

                self.texture = Some(ctx.load_texture(
                    "detection_view",
                    rendered,
                    Default::default(),
                ));
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
        let mut img_rgba = img.to_rgba8();

        for (idx, det) in self.detections.iter().enumerate() {
            // RUNTIME ASSERT: Bounding box must be within image bounds
            assert!(det.x1 >= 0.0 && det.x1 < img.width() as f32,
                    "Bbox x1={} out of image width={}", det.x1, img.width());
            assert!(det.y1 >= 0.0 && det.y1 < img.height() as f32,
                    "Bbox y1={} out of image height={}", det.y1, img.height());
            assert!(det.x2 > det.x1 && det.x2 <= img.width() as f32,
                    "Bbox x2={} invalid (x1={}, width={})", det.x2, det.x1, img.width());
            assert!(det.y2 > det.y1 && det.y2 <= img.height() as f32,
                    "Bbox y2={} invalid (y1={}, height={})", det.y2, det.y1, img.height());

            // Color based on selection
            let color = if self.selected_detection == Some(idx) {
                image::Rgba([255u8, 0, 0, 255])  // Red for selected
            } else {
                image::Rgba([0, 255, 0, 255])    // Green for others
            };

            // Draw bounding box (3 pixels thick)
            let rect = imageproc::rect::Rect::at(det.x1 as i32, det.y1 as i32)
                .of_size((det.x2 - det.x1) as u32, (det.y2 - det.y1) as u32);

            for thickness in 0..3 {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img_rgba,
                    imageproc::rect::Rect::at(rect.left() - thickness, rect.top() - thickness)
                        .of_size(rect.width() + 2 * thickness as u32, rect.height() + 2 * thickness as u32),
                    color,
                );
            }

            // Draw label with background
            let label = format!("{} {:.2}", det.class_name, det.confidence);
            let label_x = det.x1.max(5.0) as i32;
            let label_y = (det.y1 - 20.0).max(5.0) as i32;

            // Draw text background
            let bg_rect = imageproc::rect::Rect::at(label_x - 2, label_y - 2)
                .of_size((label.len() * 8 + 4) as u32, 18);
            imageproc::drawing::draw_filled_rect_mut(
                &mut img_rgba,
                bg_rect,
                image::Rgba([0, 0, 0, 200]),
            );

            // Load font for text rendering
            let font_data = include_bytes!("../../fonts/NotoSans-Regular.ttf");
            let font = ab_glyph::FontArc::try_from_slice(font_data)
                .expect("Failed to load font");

            imageproc::drawing::draw_text_mut(
                &mut img_rgba,
                image::Rgba([255, 255, 255, 255]),
                label_x,
                label_y,
                14.0,
                &font,
                &label,
            );
        }

        // Convert to egui ColorImage
        let size = [img_rgba.width() as usize, img_rgba.height() as usize];
        let pixels = img_rgba.into_raw();
        ColorImage::from_rgba_unmultiplied(size, &pixels)
    }
}
