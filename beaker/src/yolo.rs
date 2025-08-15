use crate::detection_obj::Detection;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Array;

pub fn preprocess_image(
    img: &DynamicImage,
    target_size: u32,
) -> Result<Array<f32, ndarray::IxDyn>> {
    // Convert to RGB if needed
    let rgb_img = img.to_rgb8();
    let (orig_width, orig_height) = rgb_img.dimensions();

    // Calculate letterbox resize dimensions
    let max_dim = if orig_width > orig_height {
        orig_width
    } else {
        orig_height
    };
    let scale = (target_size as f32) / (max_dim as f32);
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    // Resize image
    let resized = image::imageops::resize(
        &rgb_img,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    );

    // Create letterboxed image with gray padding (114, 114, 114)
    let mut letterboxed = image::RgbImage::new(target_size, target_size);
    for pixel in letterboxed.pixels_mut() {
        *pixel = image::Rgb([114, 114, 114]);
    }

    // Calculate offsets to center the resized image
    let x_offset = (target_size - new_width) / 2;
    let y_offset = (target_size - new_height) / 2;

    // Copy resized image to center of letterboxed image
    for y in 0..new_height {
        for x in 0..new_width {
            let src_pixel = resized.get_pixel(x, y);
            letterboxed.put_pixel(x + x_offset, y + y_offset, *src_pixel);
        }
    }

    // Convert to NCHW format and normalize
    let mut input_data = Vec::with_capacity((3 * target_size * target_size) as usize);

    // Fill in NCHW order: batch, channel, height, width
    for c in 0..3 {
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = letterboxed.get_pixel(x, y);
                let value = pixel[c] as f32 / 255.0;
                input_data.push(value);
            }
        }
    }

    // Create ndarray with dynamic shape
    let input = Array::from_shape_vec(
        ndarray::IxDyn(&[1, 3, target_size as usize, target_size as usize]),
        input_data,
    )?;

    Ok(input)
}

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
