//! Beaker: Bird detection and analysis toolkit
//!
//! This library provides unified infrastructure for AI-powered bird image analysis,
//! including head detection and background removal capabilities.

pub mod blur_detection;
pub mod cache_common;
pub mod color_utils;
pub mod config;
pub mod cutout_postprocessing;
pub mod cutout_preprocessing;
pub mod cutout_processing;
pub mod detection;
pub mod detection_obj;
pub mod image_input;
pub mod mask_encoding;
pub mod model_access;
pub mod model_processing;
pub mod onnx_session;
pub mod output_manager;
pub mod progress;
pub mod quality_processing;
pub mod quality_types;
pub mod quality_visualization;
pub mod rfdetr;
pub mod shared_metadata;
