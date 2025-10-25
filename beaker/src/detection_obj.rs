// Separate file because of circular import issues
// (there is probably a better way)
use serde::Serialize;

use crate::blur_detection::DetectionQuality;

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub angle_radians: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<DetectionQuality>,
}
