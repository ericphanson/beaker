// TypeScript types matching the Rust backend commands

export interface DetectRequest {
  imagePath: string;
  confidence: number;
  iouThreshold: number;
  classes: string[];
  boundingBox: boolean;
  outputDir?: string;
}

export interface DetectResponse {
  detections: DetectionInfo[];
  processingTimeMs: number;
  boundingBoxPath?: string;
  inputImgWidth: number;
  inputImgHeight: number;
}

export interface DetectionInfo {
  className: string;
  confidence: number;
  bbox: BBox;
  cropPath?: string;
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}
