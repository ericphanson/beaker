//! Configuration layer providing clean separation between CLI arguments and internal model configurations.
//!
//! This module defines the shared configuration structures used throughout the beaker toolkit:
//! - `BaseModelConfig`: Common configuration options shared by all models
//! - Model-specific configurations that embed the base config
//! - Conversion traits from CLI commands to internal configurations
//!
//! The design separates CLI concerns (argument parsing, help text, validation) from
//! business logic (processing parameters, feature flags, internal state).

use clap::Parser;
use clap_verbosity_flag::Verbosity;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

/// Supported detection classes for multi-class detection
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum DetectionClass {
    Bird,
    Head,
    Eye,
    Beak,
}

impl std::str::FromStr for DetectionClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bird" => Ok(DetectionClass::Bird),
            "head" => Ok(DetectionClass::Head),
            "eye" => Ok(DetectionClass::Eye),
            "beak" => Ok(DetectionClass::Beak),
            _ => Err(format!("Unknown detection class: {s}")),
        }
    }
}

impl DetectionClass {
    /// Convert DetectionClass to string
    pub fn to_string(&self) -> &'static str {
        match self {
            DetectionClass::Bird => "bird",
            DetectionClass::Head => "head",
            DetectionClass::Eye => "eye",
            DetectionClass::Beak => "beak",
        }
    }

    /// Get all available classes
    pub fn all_classes() -> Vec<DetectionClass> {
        vec![
            DetectionClass::Bird,
            DetectionClass::Head,
            DetectionClass::Eye,
            DetectionClass::Beak,
        ]
    }
}

/// Parse crop classes from comma-separated string
pub fn parse_crop_classes(crop_str: &str) -> Result<HashSet<DetectionClass>, String> {
    if crop_str.trim().to_lowercase() == "all" {
        return Ok(DetectionClass::all_classes().into_iter().collect());
    }

    let mut classes = HashSet::new();
    for class_str in crop_str.split(',') {
        let class_str = class_str.trim();
        if !class_str.is_empty() {
            classes.insert(class_str.parse()?);
        }
    }

    if classes.is_empty() {
        return Err("No valid classes specified".to_string());
    }

    Ok(classes)
}

/// Parse RGBA color from string like "255,255,255,255"
pub fn parse_rgba_color(s: &str) -> Result<[u8; 4], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("Color must be in format 'R,G,B,A' (e.g., '255,255,255,255')".to_string());
    }

    let mut color = [0u8; 4];
    for (i, part) in parts.iter().enumerate() {
        color[i] = part
            .trim()
            .parse::<u8>()
            .map_err(|_| format!("Invalid color component: '{part}'"))?;
    }

    Ok(color)
}

/// Parse probability value (must be between 0.0 and 1.0)
pub fn parse_probability(s: &str) -> Result<f32, String> {
    let val = s
        .parse::<f32>()
        .map_err(|_| format!("Invalid number: '{s}'"))?;
    if !(0.0..=1.0).contains(&val) {
        return Err(format!("Must be between 0.0 and 1.0, got {val}"));
    }
    Ok(val)
}

/// Global CLI arguments that apply to all beaker commands
#[derive(Parser, Debug, Clone)]
pub struct GlobalArgs {
    /// Global output directory (overrides default placement next to input)
    #[arg(long, global = true)]
    pub output_dir: Option<String>,

    /// Create metadata output file(s)
    #[arg(long, global = true)]
    pub metadata: bool,

    /// Verbosity level (-q/--quiet, -v/-vv/-vvv/-vvvv for info/debug/trace)
    #[command(flatten)]
    pub verbosity: Verbosity,

    /// Use permissive mode for input validation (warn instead of error for unsupported files)
    #[arg(long, global = true)]
    pub permissive: bool,

    /// Device to use for inference (auto, cpu, coreml)
    #[arg(long, default_value = "auto", global = true)]
    pub device: String,

    /// Disable colored output (also respects NO_COLOR and BEAKER_NO_COLOR env vars)
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Allow overwriting outputs when basename collisions are detected
    #[arg(long, global = true)]
    pub force: bool,
}

/// Base configuration common to all models
#[derive(Debug, Clone, Serialize)]
pub struct BaseModelConfig {
    /// Input sources (images or directories)
    pub sources: Vec<String>,
    /// Device for inference
    pub device: String,
    /// Optional output directory override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    /// Whether to skip metadata generation
    pub skip_metadata: bool,
    /// Use strict mode (fail if files are not found or are unsupported). Opposite of `--permissive`.
    pub strict: bool,
    /// Allow overwriting outputs when basename collisions are detected
    pub force: bool,
}

/// CLI command for object detection (only command-specific arguments)
#[derive(Parser, Debug, Clone)]
pub struct DetectCommand {
    /// Path(s) to input images or directories. Supports glob patterns like *.jpg
    #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
    pub sources: Vec<String>,

    /// Confidence threshold for detections (0.0-1.0)
    #[arg(short, long, default_value = "0.5", value_parser = parse_probability)]
    pub confidence: f32,

    /// Classes to crop as comma-separated list (bird,head,eye,beak) or 'all' for all classes.
    /// Leave empty to disable cropping.
    #[arg(long, value_name = "CLASSES")]
    pub crop: Option<String>,

    /// Save an image with bounding boxes drawn
    #[arg(long)]
    pub bounding_box: bool,

    /// Path to custom head detection model file
    #[arg(long)]
    pub model_path: Option<String>,

    /// URL to download custom head detection model from
    #[arg(long)]
    pub model_url: Option<String>,

    /// MD5 checksum for model verification (used with --model-url)
    #[arg(long)]
    pub model_checksum: Option<String>,
}

/// Internal configuration for detection processing
#[derive(Debug, Clone, Serialize)]
pub struct DetectionConfig {
    #[serde(skip)]
    pub base: BaseModelConfig,
    pub confidence: f32,
    pub crop_classes: HashSet<DetectionClass>,
    pub bounding_box: bool,
    /// CLI-provided model path override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// CLI-provided model URL override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_url: Option<String>,
    /// CLI-provided model checksum override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
    // Stores local quality grid results
    #[serde(skip)]
    pub quality_results: Option<HashMap<String, QualityResult>>,
}

/// CLI command for cutout processing (only command-specific arguments)
#[derive(Parser, Debug, Clone)]
pub struct CutoutCommand {
    /// Path(s) to input images or directories. Supports glob patterns like *.jpg
    #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
    pub sources: Vec<String>,

    /// Apply post-processing to smooth mask edges
    #[arg(long)]
    pub post_process: bool,

    /// Use alpha matting for better edge quality
    #[arg(long)]
    pub alpha_matting: bool,

    /// Foreground threshold for alpha matting (0-255)
    #[arg(long, default_value = "240")]
    pub alpha_matting_foreground_threshold: u8,

    /// Background threshold for alpha matting (0-255)
    #[arg(long, default_value = "10")]
    pub alpha_matting_background_threshold: u8,

    /// Erosion size for alpha matting
    #[arg(long, default_value = "10")]
    pub alpha_matting_erode_size: u32,

    /// Background color as RGBA (e.g., "255,255,255,255" for white)
    #[arg(long, value_parser = parse_rgba_color)]
    pub background_color: Option<[u8; 4]>,

    /// Save the segmentation mask as a separate image
    #[arg(long)]
    pub save_mask: bool,

    /// Path to custom cutout model file
    #[arg(long)]
    pub model_path: Option<String>,

    /// URL to download custom cutout model from
    #[arg(long)]
    pub model_url: Option<String>,

    /// MD5 checksum for model verification (used with --model-url)
    #[arg(long)]
    pub model_checksum: Option<String>,
}

/// CLI command for quality assessment processing (only command-specific arguments)
#[derive(Parser, Debug, Clone)]
pub struct QualityCommand {
    /// Path(s) to input images or directories. Supports glob patterns like *.jpg
    #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
    pub sources: Vec<String>,

    /// Path to custom quality model file
    #[arg(long)]
    pub model_path: Option<String>,

    /// URL to download custom quality model from
    #[arg(long)]
    pub model_url: Option<String>,

    /// MD5 checksum for model verification (used with --model-url)
    #[arg(long)]
    pub model_checksum: Option<String>,

    /// Dump debug heatmap and overlay images for blur analysis
    #[arg(long)]
    pub debug_dump_images: bool,
}

/// Internal configuration for cutout processing
#[derive(Debug, Clone, Serialize)]
pub struct CutoutConfig {
    #[serde(skip)]
    pub base: BaseModelConfig,
    pub post_process_mask: bool,
    pub alpha_matting: bool,
    pub alpha_matting_foreground_threshold: u8,
    pub alpha_matting_background_threshold: u8,
    pub alpha_matting_erode_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_color: Option<[u8; 4]>,
    pub save_mask: bool,
    /// CLI-provided model path override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// CLI-provided model URL override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_url: Option<String>,
    /// CLI-provided model checksum override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
}

/// Internal configuration for quality assessment processing
#[derive(Debug, Clone, Serialize)]
pub struct QualityConfig {
    #[serde(skip)]
    pub base: BaseModelConfig,
    /// CLI-provided model path override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// CLI-provided model URL override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_url: Option<String>,
    /// CLI-provided model checksum override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
    /// Whether to dump debug heatmap and overlay images
    pub debug_dump_images: bool,
}

impl QualityConfig {
    /// Create configuration from detection config
    pub fn from_detection_config(config: &DetectionConfig) -> Self {
        Self {
            base: config.base.clone(), // Clone the base config
            model_checksum: None,
            model_url: None,
            model_path: None,
            debug_dump_images: false,
        }
    }
}

// Conversion traits from CLI commands to internal configurations

impl From<GlobalArgs> for BaseModelConfig {
    fn from(global: GlobalArgs) -> Self {
        Self {
            sources: Vec::new(), // Sources come from command, not global args
            device: global.device,
            output_dir: global.output_dir,
            skip_metadata: !global.metadata, // Note: CLI uses metadata flag, internal uses skip_metadata
            strict: !global.permissive,      // Note: CLI uses permissive, internal uses strict
            force: global.force,
        }
    }
}

impl DetectionConfig {
    /// Create configuration from global args and command-specific args
    pub fn from_args(global: GlobalArgs, cmd: DetectCommand) -> Result<Self, String> {
        let mut base: BaseModelConfig = global.into();
        base.sources = cmd.sources; // Add sources from command

        let crop_classes = match cmd.crop {
            Some(crop_str) => parse_crop_classes(&crop_str)?,
            None => HashSet::new(), // No cropping if not specified
        };

        Ok(Self {
            base,
            confidence: cmd.confidence,
            crop_classes,
            bounding_box: cmd.bounding_box,
            model_path: cmd.model_path,
            model_url: cmd.model_url,
            model_checksum: cmd.model_checksum,
            quality_results: None,
        })
    }
}

impl CutoutConfig {
    /// Create configuration from global args and command-specific args
    pub fn from_args(global: GlobalArgs, cmd: CutoutCommand) -> Result<Self, String> {
        // Validate conflicting flags (UX-02)
        if cmd.alpha_matting && cmd.background_color.is_some() {
            return Err(
                "Cannot use both --alpha-matting and --background-color. Choose one.".to_string(),
            );
        }

        let mut base: BaseModelConfig = global.into();
        base.sources = cmd.sources; // Add sources from command

        Ok(Self {
            base,
            post_process_mask: cmd.post_process,
            alpha_matting: cmd.alpha_matting,
            alpha_matting_foreground_threshold: cmd.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold: cmd.alpha_matting_background_threshold,
            alpha_matting_erode_size: cmd.alpha_matting_erode_size,
            background_color: cmd.background_color,
            save_mask: cmd.save_mask,
            model_path: cmd.model_path,
            model_url: cmd.model_url,
            model_checksum: cmd.model_checksum,
        })
    }
}

impl QualityConfig {
    /// Create configuration from global args and command-specific args
    pub fn from_args(global: GlobalArgs, cmd: QualityCommand) -> Result<Self, String> {
        let mut base: BaseModelConfig = global.into();
        base.sources = cmd.sources; // Add sources from command

        Ok(Self {
            base,
            model_path: cmd.model_path,
            model_url: cmd.model_url,
            model_checksum: cmd.model_checksum,
            debug_dump_images: cmd.debug_dump_images,
        })
    }
}

// ModelConfig trait implementations for model_processing integration
use crate::{model_processing::ModelConfig, quality_processing::QualityResult};

impl ModelConfig for DetectionConfig {
    fn base(&self) -> &BaseModelConfig {
        &self.base
    }

    fn tool_name(&self) -> &'static str {
        "detect"
    }
}

impl ModelConfig for CutoutConfig {
    fn base(&self) -> &BaseModelConfig {
        &self.base
    }

    fn tool_name(&self) -> &'static str {
        "cutout"
    }
}

impl ModelConfig for QualityConfig {
    fn base(&self) -> &BaseModelConfig {
        &self.base
    }

    fn tool_name(&self) -> &'static str {
        "quality"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_args_conversion() {
        let global_args = GlobalArgs {
            device: "cpu".to_string(),
            output_dir: Some("/tmp".to_string()),
            metadata: false,
            verbosity: Verbosity::new(2, 0), // -vv level (info level enables verbose)
            permissive: true,
            no_color: false,
            force: false,
        };

        let config: BaseModelConfig = global_args.into();

        assert_eq!(config.sources, Vec::<String>::new()); // Sources come from command
        assert_eq!(config.device, "cpu");
        assert_eq!(config.output_dir, Some("/tmp".to_string()));
        assert!(config.skip_metadata); // metadata=false -> skip_metadata=true
                                       // Note: verbosity is now handled directly by the logging system via env_logger
        assert!(!config.strict); // permissive=true -> strict=false
        assert!(!config.force);
    }

    #[test]
    fn test_detect_command_conversion() {
        let global_args = GlobalArgs {
            device: "auto".to_string(),
            output_dir: None,
            metadata: false,
            verbosity: Verbosity::new(0, 0), // Default level (warnings and errors only)
            permissive: false,
            no_color: false,
            force: false,
        };

        let detect_cmd = DetectCommand {
            sources: vec!["bird.jpg".to_string()],
            confidence: 0.8,
            crop: Some("head,bird".to_string()),
            bounding_box: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        let config = DetectionConfig::from_args(global_args, detect_cmd).unwrap();

        assert_eq!(config.base.sources, vec!["bird.jpg"]);
        assert_eq!(config.confidence, 0.8);
        assert!(config
            .crop_classes
            .contains(&crate::config::DetectionClass::Head));
        assert!(config
            .crop_classes
            .contains(&crate::config::DetectionClass::Bird));
        assert!(!config.bounding_box);
        assert!(config.base.strict); // permissive=false -> strict=true
        assert_eq!(config.model_path, None);
        assert_eq!(config.model_url, None);
        assert_eq!(config.model_checksum, None);
    }

    #[test]
    fn test_cutout_command_conversion() {
        let global_args = GlobalArgs {
            device: "coreml".to_string(),
            output_dir: Some("/output".to_string()),
            metadata: false,
            verbosity: Verbosity::new(1, 0), // -v level (info)
            permissive: false,
            no_color: false,
            force: false,
        };

        let cutout_cmd = CutoutCommand {
            sources: vec!["photo.png".to_string()],
            post_process: true,
            alpha_matting: false,
            alpha_matting_foreground_threshold: 240,
            alpha_matting_background_threshold: 10,
            alpha_matting_erode_size: 10,
            background_color: Some([255, 255, 255, 255]),
            save_mask: true,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        let config = CutoutConfig::from_args(global_args, cutout_cmd).unwrap();

        assert_eq!(config.base.sources, vec!["photo.png"]);
        assert_eq!(config.base.device, "coreml");
        assert!(config.post_process_mask);
        assert!(!config.alpha_matting);
        assert_eq!(config.background_color, Some([255, 255, 255, 255]));
        assert!(config.save_mask);
        assert_eq!(config.model_path, None);
        assert_eq!(config.model_url, None);
        assert_eq!(config.model_checksum, None);
    }

    #[test]
    fn test_backward_compatibility_methods() {
        let config = DetectionConfig {
            base: BaseModelConfig {
                sources: vec!["test.jpg".to_string()],
                device: "cpu".to_string(),
                output_dir: Some("/tmp".to_string()),
                skip_metadata: true,
                strict: true,
                force: false,
            },
            confidence: 0.25,
            crop_classes: HashSet::new(),
            bounding_box: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
            quality_results: None,
        };

        // Test field access through base config
        assert_eq!(config.base.sources, vec!["test.jpg".to_string()]);
        assert_eq!(config.base.device, "cpu");
        assert_eq!(config.base.output_dir, Some("/tmp".to_string()));
        assert!(config.base.skip_metadata);
        assert!(config.base.strict);
        assert!(!config.base.force);
    }

    #[test]
    fn test_quality_command_conversion() {
        let global_args = GlobalArgs {
            device: "auto".to_string(),
            output_dir: Some("/quality_output".to_string()),
            metadata: true,
            verbosity: Verbosity::new(0, 0), // Default level
            permissive: false,
            no_color: false,
            force: false,
        };

        let quality_cmd = QualityCommand {
            sources: vec!["test_image.jpg".to_string()],
            model_path: Some("/custom/model.onnx".to_string()),
            model_url: None,
            model_checksum: None,
            debug_dump_images: true,
        };

        let config = QualityConfig::from_args(global_args, quality_cmd).unwrap();

        // Test conversion from command to config
        assert_eq!(config.base.sources, vec!["test_image.jpg"]);
        assert_eq!(config.base.device, "auto");
        assert_eq!(config.base.output_dir, Some("/quality_output".to_string()));
        assert!(!config.base.skip_metadata); // metadata=true -> skip_metadata=false
        assert!(config.base.strict); // permissive=false -> strict=true
        assert_eq!(config.model_path, Some("/custom/model.onnx".to_string()));
        assert_eq!(config.model_url, None);
        assert_eq!(config.model_checksum, None);
        assert!(config.debug_dump_images);
    }

    #[test]
    fn test_cutout_conflicting_flags_validation() {
        let global_args = GlobalArgs {
            device: "auto".to_string(),
            output_dir: None,
            metadata: false,
            verbosity: Verbosity::new(0, 0),
            permissive: false,
            no_color: false,
            force: false,
        };

        let cutout_cmd = CutoutCommand {
            sources: vec!["photo.png".to_string()],
            post_process: false,
            alpha_matting: true,
            alpha_matting_foreground_threshold: 240,
            alpha_matting_background_threshold: 10,
            alpha_matting_erode_size: 10,
            background_color: Some([255, 0, 0, 255]), // Conflicting with alpha_matting
            save_mask: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        let result = CutoutConfig::from_args(global_args, cutout_cmd);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot use both"));
    }

    #[test]
    fn test_cutout_valid_alpha_matting() {
        let global_args = GlobalArgs {
            device: "auto".to_string(),
            output_dir: None,
            metadata: false,
            verbosity: Verbosity::new(0, 0),
            permissive: false,
            no_color: false,
            force: false,
        };

        let cutout_cmd = CutoutCommand {
            sources: vec!["photo.png".to_string()],
            post_process: false,
            alpha_matting: true,
            alpha_matting_foreground_threshold: 240,
            alpha_matting_background_threshold: 10,
            alpha_matting_erode_size: 10,
            background_color: None, // Valid - no background color
            save_mask: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        let result = CutoutConfig::from_args(global_args, cutout_cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rgba_color() {
        // Valid color
        assert_eq!(parse_rgba_color("255,128,0,255"), Ok([255, 128, 0, 255]));
        assert_eq!(parse_rgba_color("0,0,0,0"), Ok([0, 0, 0, 0]));

        // Invalid formats
        assert!(parse_rgba_color("255,128,0").is_err()); // Too few components
        assert!(parse_rgba_color("255,128,0,255,128").is_err()); // Too many components
        assert!(parse_rgba_color("256,128,0,255").is_err()); // Out of range
        assert!(parse_rgba_color("invalid,128,0,255").is_err()); // Non-numeric
    }

    #[test]
    fn test_parse_probability() {
        // Valid probabilities
        assert_eq!(parse_probability("0.0"), Ok(0.0));
        assert_eq!(parse_probability("0.5"), Ok(0.5));
        assert_eq!(parse_probability("1.0"), Ok(1.0));

        // Invalid probabilities
        assert!(parse_probability("-0.5").is_err()); // Below range
        assert!(parse_probability("2.0").is_err()); // Above range
        assert!(parse_probability("1.5").is_err()); // Above range
        assert!(parse_probability("invalid").is_err()); // Non-numeric
    }
}
