use beaker::config::DetectionClass;
use beaker::gui_api::{detect_single_image, SimpleDetectionParams};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectRequest {
    pub image_path: String,
    pub confidence: f32,
    pub iou_threshold: f32,
    pub classes: Vec<String>,
    pub bounding_box: bool,
    pub output_dir: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectResponse {
    pub detections: Vec<DetectionInfo>,
    pub processing_time_ms: f64,
    pub bounding_box_path: Option<String>,
    pub input_img_width: u32,
    pub input_img_height: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionInfo {
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BBox,
    pub crop_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[tauri::command]
pub async fn detect_objects(request: DetectRequest) -> Result<DetectResponse, String> {
    // Validate inputs with assertions
    assert!(
        request.confidence >= 0.0 && request.confidence <= 1.0,
        "Confidence must be between 0.0 and 1.0"
    );
    assert!(
        request.iou_threshold >= 0.0 && request.iou_threshold <= 1.0,
        "IoU threshold must be between 0.0 and 1.0"
    );
    assert!(
        !request.classes.is_empty(),
        "At least one detection class must be specified"
    );
    assert!(
        Path::new(&request.image_path).exists(),
        "Image path does not exist: {}",
        request.image_path
    );

    // Run detection in a blocking task to avoid blocking the async runtime
    let result = tokio::task::spawn_blocking(move || {
        // Parse classes
        let crop_classes: HashSet<DetectionClass> = request
            .classes
            .iter()
            .filter_map(|s| s.parse::<DetectionClass>().ok())
            .collect();

        if crop_classes.is_empty() {
            return Err("No valid detection classes provided".to_string());
        }

        // Create detection params
        let params = SimpleDetectionParams {
            image_path: PathBuf::from(&request.image_path),
            confidence: request.confidence,
            iou_threshold: request.iou_threshold,
            crop_classes,
            bounding_box: request.bounding_box,
            output_dir: request.output_dir.map(PathBuf::from),
        };

        // Run detection
        detect_single_image(params).map_err(|e| format!("Detection failed: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))??;

    // Convert beaker DetectionResult to our DetectResponse
    let detections = result
        .detections
        .into_iter()
        .map(|d| DetectionInfo {
            class_name: d.detection.class.clone(),
            confidence: d.detection.confidence,
            bbox: BBox {
                x: d.detection.bbox.x,
                y: d.detection.bbox.y,
                width: d.detection.bbox.width,
                height: d.detection.bbox.height,
            },
            crop_path: d.crop_path,
        })
        .collect();

    // Validate output: all detections should have valid bboxes
    for det in &detections {
        assert!(
            det.bbox.width > 0.0 && det.bbox.height > 0.0,
            "Invalid bounding box dimensions"
        );
    }

    // Verify output files exist
    if let Some(ref bbox_path) = result.bounding_box_path {
        assert!(
            Path::new(bbox_path).exists(),
            "Bounding box image not created at expected path"
        );
    }

    for det in &detections {
        if let Some(ref crop_path) = det.crop_path {
            assert!(
                Path::new(crop_path).exists(),
                "Crop image not created at expected path: {}",
                crop_path
            );
        }
    }

    Ok(DetectResponse {
        detections,
        processing_time_ms: result.processing_time_ms,
        bounding_box_path: result.bounding_box_path,
        input_img_width: result.input_img_width,
        input_img_height: result.input_img_height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detect_objects_validates_confidence() {
        let req = DetectRequest {
            image_path: "tests/fixtures/sparrow.jpg".to_string(),
            confidence: 1.5, // Invalid
            iou_threshold: 0.45,
            classes: vec!["bird".to_string()],
            bounding_box: false,
            output_dir: None,
        };

        // This should panic due to assertion
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(detect_objects(req))
        }));

        assert!(result.is_err());
    }
}
