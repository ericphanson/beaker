use ndarray::Array;

use crate::config::DetectionConfig;
use crate::detection_obj::Detection;
use anyhow::Result;

/// Convert ONNX output value to ndarray
pub fn to_array(output: &ort::value::Value) -> Result<ndarray::ArrayD<f32>> {
    let output_view: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>> =
        output
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;
    let output_array =
        Array::from_shape_vec(output_view.shape(), output_view.iter().cloned().collect())?;
    Ok(output_array)
}

// Python version:
// prob = out_logits.sigmoid()  # per-class probs
// B, Q, C = prob.shape
// topk_values, topk_indexes = torch.topk(prob.view(B, -1), self.num_select, dim=1)
// scores = topk_values
// topk_boxes = topk_indexes // C  # [B, N] -> query indices
// labels = topk_indexes % C  # [B, N] -> class indices

// # gather boxes for those queries
// boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)  # [B, Q, 4]
// boxes = torch.gather(
//     boxes_xyxy,
//     1,
//     topk_boxes.unsqueeze(-1).repeat(1, 1, 4),  # [B, N, 4]
// )

// # gather orientations for those queries
// orients = torch.gather(
//     out_orient,
//     1,
//     topk_boxes.unsqueeze(-1).repeat(1, 1, 2),  # [B, N, 2]
// )

// # (optional) normalize and/or convert to angle
// orients_unit = torch.nn.functional.normalize(orients, dim=-1)  # keep (cos,sin)
// angles = torch.atan2(
//     orients_unit[..., 1], orients_unit[..., 0]
// )  # [B, N] radians

// # scale boxes to absolute pixels
// img_h, img_w = target_sizes.unbind(1)
// scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)  # [B, 4]
// boxes = boxes * scale_fct[:, None, :]

/// Postprocess RF-DETR model outputs to extract bird detections with orientation
///
/// This function implements the RF-DETR postprocessing pipeline:
/// 1. Applies sigmoid to convert logits to probabilities
/// 2. Performs top-k selection (k=10) across all query-class combinations
/// 3. Extracts bounding boxes and converts from cxcywh to xyxy format
/// 4. Scales coordinates to original image dimensions
/// 5. Filters by confidence threshold
///
/// # Arguments
/// * `outputs` - Session outputs containing [boxes, logits, orientations]
/// * `orig_width` - Original image width for coordinate scaling
/// * `orig_height` - Original image height for coordinate scaling
/// * `_model_size` - Model input size (unused for RF-DETR)
/// * `config` - Detection configuration with confidence threshold
pub fn postprocess_output(
    outputs: &ort::session::SessionOutputs,
    orig_width: u32,
    orig_height: u32,
    _model_size: u32,
    config: &DetectionConfig,
) -> Result<Vec<Detection>, anyhow::Error> {
    let out_bbox = to_array(&outputs[0])?; // [B, Q, 4] - boxes in cxcywh format
    let out_logits = to_array(&outputs[1])?; // [B, Q, C] - class logits
    let out_orient = to_array(&outputs[2])?; // [B, Q, 2] - orientation vectors

    log::debug!(
        "Postprocessing RF-DETR outputs. Got: {:?}",
        out_bbox.shape()
    );

    // Apply sigmoid to get per-class probabilities
    let prob = out_logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));

    let shape = prob.shape();
    let b = shape[0]; // batch size (should be 1)
    let q = shape[1]; // number of queries
    let c = shape[2]; // number of classes

    let k = 10; // topk k=10
    let mut detections = Vec::new();

    for batch_idx in 0..b {
        // Collect all probabilities with their flat indices for this batch
        let mut indexed_probs = Vec::new();
        for query_idx in 0..q {
            for class_idx in 0..c {
                let prob_val = prob[[batch_idx, query_idx, class_idx]];
                let flat_idx = query_idx * c + class_idx;
                indexed_probs.push((flat_idx, prob_val));
            }
        }

        // Sort by probability (descending) and take top k
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_probs.truncate(k);
        for (flat_idx, score) in indexed_probs {
            // Skip if below confidence threshold
            if score < config.confidence {
                continue;
            }

            // Convert flat index back to query and class indices
            let query_idx = flat_idx / c; // topk_boxes = topk_indexes // C
            let class_idx = flat_idx % c; // labels = topk_indexes % C

            // Get box coordinates for this query (cxcywh format)
            let cx = out_bbox[[batch_idx, query_idx, 0]];
            let cy = out_bbox[[batch_idx, query_idx, 1]];
            let w = out_bbox[[batch_idx, query_idx, 2]];
            let h = out_bbox[[batch_idx, query_idx, 3]];

            // Convert from cxcywh to xyxy format
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Scale boxes to original image size
            let scale_x = orig_width as f32;
            let scale_y = orig_height as f32;

            let scaled_x1 = x1 * scale_x;
            let scaled_y1 = y1 * scale_y;
            let scaled_x2 = x2 * scale_x;
            let scaled_y2 = y2 * scale_y;

            // Get orientation for this query
            let orient_cos = out_orient[[batch_idx, query_idx, 0]];
            let orient_sin = out_orient[[batch_idx, query_idx, 1]];
            let angle = orient_sin.atan2(orient_cos); // radians

            // // Map class index to class name
            // let class_name = match class_idx {
            //     0 => "bird",
            //     1 => "head",
            //     2 => "eye",
            //     3 => "beak",
            //     _ => "unknown",
            // }
            // .to_string();

            // Map class index to class name
            let class_name = match class_idx {
                0 => "background",
                1 => "bird",
                2 => "head",
                3 => "eye",
                4 => "beak",
                _ => "unknown",
            }
            .to_string();

            if ["bird", "head", "eye", "beak"].contains(&class_name.as_str()) {
                detections.push(Detection {
                    angle_radians: angle,
                    x1: scaled_x1,
                    y1: scaled_y1,
                    x2: scaled_x2,
                    y2: scaled_y2,
                    confidence: score,
                    class_id: class_idx as u32,
                    class_name,
                });
            }
        }
    }
    let classes = detections
        .iter()
        .map(|d| d.class_name.clone())
        .collect::<Vec<_>>();
    log::debug!(
        "Postprocessed {} detections with classes {:?}",
        detections.len(),
        classes
    );

    Ok(detections)
}
