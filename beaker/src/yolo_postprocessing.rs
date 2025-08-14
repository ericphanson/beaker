use std::path::Path;

use crate::detection_obj::Detection;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array;

pub fn nms(detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    // Group detections by class_id
    use std::collections::HashMap;
    let mut class_groups: HashMap<u32, Vec<Detection>> = HashMap::new();
    for detection in detections {
        class_groups
            .entry(detection.class_id)
            .or_default()
            .push(detection);
    }

    let mut all_results = Vec::new();

    // Apply NMS separately to each class
    for (_, mut class_detections) in class_groups {
        // Sort by confidence score in descending order
        class_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; class_detections.len()];

        for i in 0..class_detections.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(class_detections[i].clone());

            // Suppress overlapping detections within the same class
            for j in (i + 1)..class_detections.len() {
                if !suppressed[j] && class_detections[i].iou(&class_detections[j]) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        all_results.extend(keep);
    }

    all_results
}

pub fn postprocess_output(
    output: &Array<f32, ndarray::IxDyn>,
    confidence_threshold: f32,
    iou_threshold: f32,
    img_width: u32,
    img_height: u32,
    model_size: u32,
    is_legacy_detect_model: bool,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    // Output shape should be [1, num_classes + 4, num_boxes]
    let shape = output.shape();
    if shape.len() != 3 {
        return Err(anyhow::anyhow!("Expected 3D output, got {}D", shape.len()));
    }
    let num_boxes = shape[2];

    for i in 0..num_boxes {
        // Get box coordinates (first 4 values)
        let x_center = output[[0, 0, i]];
        let y_center = output[[0, 1, i]];
        let width = output[[0, 2, i]];
        let height = output[[0, 3, i]];

        if is_legacy_detect_model {
            // Legacy single-class head detection model (class 0 = head)
            let confidence = output[[0, 4, i]];

            if confidence > confidence_threshold {
                // Convert from center coordinates to corner coordinates
                let x1 = x_center - width / 2.0;
                let y1 = y_center - height / 2.0;
                let x2 = x_center + width / 2.0;
                let y2 = y_center + height / 2.0;

                // Scale coordinates back to original image size
                let scale_x = img_width as f32 / model_size as f32;
                let scale_y = img_height as f32 / model_size as f32;

                detections.push(Detection {
                    angle_radians: f32::NAN, // No angle for legacy model
                    x1: x1 * scale_x,
                    y1: y1 * scale_y,
                    x2: x2 * scale_x,
                    y2: y2 * scale_y,
                    confidence,
                    class_id: 1, // Map old head detection to new class ID 1 (head)
                    class_name: "head".to_string(),
                });
            }
        } else {
            // New multi-class model (class 0 = bird, 1 = head, 2 = eye, 3 = beak)
            // Find the class with highest confidence
            let mut max_confidence = 0.0;
            let mut best_class_id = 0;

            // Assume classes start at index 4
            let num_classes = shape[1] - 4; // Subtract 4 for bbox coordinates
            for class_idx in 0..num_classes {
                let class_confidence = output[[0, 4 + class_idx, i]];
                if class_confidence > max_confidence {
                    max_confidence = class_confidence;
                    best_class_id = class_idx as u32;
                }
            }

            if max_confidence > confidence_threshold {
                // Convert from center coordinates to corner coordinates
                let x1 = x_center - width / 2.0;
                let y1 = y_center - height / 2.0;
                let x2 = x_center + width / 2.0;
                let y2 = y_center + height / 2.0;

                // Scale coordinates back to original image size
                let scale_x = img_width as f32 / model_size as f32;
                let scale_y = img_height as f32 / model_size as f32;

                let class_name = match best_class_id {
                    0 => "bird",
                    1 => "head",
                    2 => "eye",
                    3 => "beak",
                    _ => "unknown",
                }
                .to_string();

                detections.push(Detection {
                    angle_radians: f32::NAN, // No angle
                    x1: x1 * scale_x,
                    y1: y1 * scale_y,
                    x2: x2 * scale_x,
                    y2: y2 * scale_y,
                    confidence: max_confidence,
                    class_id: best_class_id,
                    class_name,
                });
            }
        }
    }

    // Apply Non-Maximum Suppression
    let nms_detections = nms(detections, iou_threshold);

    // Sort detections by confidence in descending order
    let mut nms_detections = nms_detections;
    nms_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(nms_detections)
}

pub fn create_square_crop(
    img: &DynamicImage,
    bbox: &Detection,
    output_path: &Path,
    padding: f32,
) -> Result<()> {
    let (img_width, img_height) = img.dimensions();

    // Extract bounding box coordinates
    let x1 = bbox.x1.max(0.0) as u32;
    let y1 = bbox.y1.max(0.0) as u32;
    let x2 = bbox.x2.min(img_width as f32) as u32;
    let y2 = bbox.y2.min(img_height as f32) as u32;

    // Calculate center and dimensions with padding
    let bbox_width = x2 - x1;
    let bbox_height = y2 - y1;
    let expand_w = (bbox_width as f32 * padding) as u32;
    let expand_h = (bbox_height as f32 * padding) as u32;

    let expanded_x1 = x1.saturating_sub(expand_w);
    let expanded_y1 = y1.saturating_sub(expand_h);
    let expanded_x2 = (x2 + expand_w).min(img_width);
    let expanded_y2 = (y2 + expand_h).min(img_height);

    // Calculate center and make square
    let center_x = (expanded_x1 + expanded_x2) / 2;
    let center_y = (expanded_y1 + expanded_y2) / 2;
    let new_width = expanded_x2 - expanded_x1;
    let new_height = expanded_y2 - expanded_y1;
    let size = new_width.max(new_height);
    let half_size = size / 2;

    // Calculate square crop bounds
    let crop_x1 = center_x.saturating_sub(half_size);
    let crop_y1 = center_y.saturating_sub(half_size);
    let crop_x2 = (center_x + half_size).min(img_width);
    let crop_y2 = (center_y + half_size).min(img_height);

    // Adjust if we hit image boundaries to maintain square
    let actual_width = crop_x2 - crop_x1;
    let actual_height = crop_y2 - crop_y1;

    let (final_x1, final_y1, final_x2, final_y2) = if actual_width < actual_height {
        let diff = actual_height - actual_width;
        (
            crop_x1.saturating_sub(diff / 2),
            crop_y1,
            (crop_x2 + diff / 2).min(img_width),
            crop_y2,
        )
    } else if actual_height < actual_width {
        let diff = actual_width - actual_height;
        (
            crop_x1,
            crop_y1.saturating_sub(diff / 2),
            crop_x2,
            (crop_y2 + diff / 2).min(img_height),
        )
    } else {
        (crop_x1, crop_y1, crop_x2, crop_y2)
    };

    // Crop the image
    let cropped = image::imageops::crop_imm(
        img,
        final_x1,
        final_y1,
        final_x2 - final_x1,
        final_y2 - final_y1,
    );

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert to appropriate format based on output file extension
    let cropped_dynamic = DynamicImage::ImageRgba8(cropped.to_image());
    let output_img = if let Some(ext) = output_path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        if ext_lower == "png" {
            // For PNG, preserve alpha channel if present
            cropped_dynamic
        } else {
            // For JPEG and other formats, convert to RGB
            DynamicImage::ImageRgb8(cropped_dynamic.to_rgb8())
        }
    } else {
        // Default to RGB if no extension
        DynamicImage::ImageRgb8(cropped_dynamic.to_rgb8())
    };

    // Save the cropped image
    output_img.save(output_path)?;

    Ok(())
}

pub fn save_bounding_box_image(
    img: &DynamicImage,
    detections: &[Detection],
    output_path: &Path,
) -> Result<()> {
    // Create a copy of the image for drawing bounding boxes
    let mut output_img = img.clone();

    // Determine if we should preserve alpha channel based on output format
    let preserve_alpha = if let Some(ext) = output_path.extension() {
        ext.to_string_lossy().to_lowercase() == "png"
    } else {
        false
    };

    // Convert to appropriate format for drawing
    if preserve_alpha {
        // For PNG output, work with RGBA to preserve transparency
        let mut rgba_img = output_img.to_rgba8();

        // Draw bounding boxes on the image
        for detection in detections {
            let x1 = detection.x1.max(0.0) as u32;
            let y1 = detection.y1.max(0.0) as u32;
            let x2 = detection.x2.min(rgba_img.width() as f32) as u32;
            let y2 = detection.y2.min(rgba_img.height() as f32) as u32;

            // Draw bounding box - bright green with full opacity
            let green = image::Rgba([0, 255, 0, 255]);

            // Draw horizontal lines
            for x in x1..=x2 {
                if x < rgba_img.width() {
                    if y1 < rgba_img.height() {
                        rgba_img.put_pixel(x, y1, green);
                    }
                    if y2 < rgba_img.height() {
                        rgba_img.put_pixel(x, y2, green);
                    }
                }
            }

            // Draw vertical lines
            for y in y1..=y2 {
                if y < rgba_img.height() {
                    if x1 < rgba_img.width() {
                        rgba_img.put_pixel(x1, y, green);
                    }
                    if x2 < rgba_img.width() {
                        rgba_img.put_pixel(x2, y, green);
                    }
                }
            }
        }

        output_img = DynamicImage::ImageRgba8(rgba_img);
    } else {
        // For JPEG output, work with RGB
        let mut rgb_img = output_img.to_rgb8();

        // Draw bounding boxes on the image
        for detection in detections {
            let x1 = detection.x1.max(0.0) as u32;
            let y1 = detection.y1.max(0.0) as u32;
            let x2 = detection.x2.min(rgb_img.width() as f32) as u32;
            let y2 = detection.y2.min(rgb_img.height() as f32) as u32;

            // Draw bounding box - bright green
            let green = image::Rgb([0, 255, 0]);

            // Draw horizontal lines
            for x in x1..=x2 {
                if x < rgb_img.width() {
                    if y1 < rgb_img.height() {
                        rgb_img.put_pixel(x, y1, green);
                    }
                    if y2 < rgb_img.height() {
                        rgb_img.put_pixel(x, y2, green);
                    }
                }
            }

            // Draw vertical lines
            for y in y1..=y2 {
                if y < rgb_img.height() {
                    if x1 < rgb_img.width() {
                        rgb_img.put_pixel(x1, y, green);
                    }
                    if x2 < rgb_img.width() {
                        rgb_img.put_pixel(x2, y, green);
                    }
                }
            }
        }

        output_img = DynamicImage::ImageRgb8(rgb_img);
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the image with bounding boxes
    output_img.save(output_path)?;

    Ok(())
}
