//! Beaker: Bird detection and analysis toolkit
//!
//! This library provides unified infrastructure for AI-powered bird image analysis,
//! including head detection and background removal capabilities.

pub mod color_utils;
pub mod config;
pub mod cutout_postprocessing;
pub mod cutout_preprocessing;
pub mod cutout_processing;
pub mod head_detection;
pub mod image_input;
pub mod model_cache;
pub mod model_processing;
pub mod onnx_session;
pub mod output_manager;
pub mod shared_metadata;
pub mod yolo_postprocessing;
pub mod yolo_preprocessing;
