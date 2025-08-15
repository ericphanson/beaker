// Separate file because of circular import issues
// (there is probably a better way)
use serde::Serialize;

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
    pub quality: Option<f32>,
}

impl Detection {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn intersection_area(&self, other: &Detection) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        if x2 > x1 && y2 > y1 {
            (x2 - x1) * (y2 - y1)
        } else {
            0.0
        }
    }

    pub fn iou(&self, other: &Detection) -> f32 {
        let intersection = self.intersection_area(other);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}
