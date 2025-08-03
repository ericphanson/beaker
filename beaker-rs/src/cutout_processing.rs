use anyhow::Result;
use chrono::{DateTime, Utc};
use image::GenericImageView;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::cutout_postprocessing::{
    apply_alpha_matting, create_cutout, create_cutout_with_background, postprocess_mask,
};
use crate::cutout_preprocessing::preprocess_image_for_isnet_v2;
use crate::model_cache::{get_or_download_model, ISNET_GENERAL_MODEL};
use crate::onnx_session::{
    create_onnx_session, determine_optimal_device, ModelSource, SessionConfig, VerboseOutput,
};
use crate::shared_metadata::{get_metadata_path, load_or_create_metadata, save_metadata};

/// Macro to print only when verbose mode is enabled
macro_rules! verbose_println {
    ($config:expr, $($arg:tt)*) => {
        if $config.verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Serialize, Clone)]
pub struct CutoutResult {
    pub input_path: String,
    pub output_path: String,
    pub model_version: String,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
}

#[derive(Serialize)]
pub struct CutoutOutput {
    pub cutout: CutoutSection,
}

#[derive(Serialize)]
pub struct CutoutSection {
    pub timestamp: DateTime<Utc>,
    pub model_version: String,
    pub post_process_mask: bool,
    pub alpha_matting: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_color: Option<[u8; 4]>,
    pub input_path: String,
    pub output_path: String,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CutoutConfig {
    pub sources: Vec<String>,
    pub device: String,
    pub output_dir: Option<String>,
    pub post_process_mask: bool,
    pub alpha_matting: bool,
    pub alpha_matting_foreground_threshold: u8,
    pub alpha_matting_background_threshold: u8,
    pub alpha_matting_erode_size: u32,
    pub background_color: Option<[u8; 4]>,
    pub save_mask: bool,
    pub skip_metadata: bool,
    pub verbose: bool,
}

impl VerboseOutput for CutoutConfig {
    fn verbose_println(&self, msg: String) {
        if self.verbose {
            println!("{msg}");
        }
    }
}

/// Check if a file is a supported image format
fn is_image_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        matches!(
            ext_lower.as_str(),
            "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff" | "tif"
        )
    } else {
        false
    }
}

/// Find all image files in a directory (non-recursive)
fn find_image_files(dir_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut image_files = Vec::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && is_image_file(&path) {
            image_files.push(path);
        }
    }

    // Sort for consistent ordering
    image_files.sort();
    Ok(image_files)
}

/// Collect all image files from multiple sources (files, directories, or glob patterns)
fn collect_image_files_from_sources(sources: &[String]) -> Result<Vec<std::path::PathBuf>> {
    let mut all_image_files = Vec::new();

    for source in sources {
        let source_path = Path::new(source);

        if source_path.is_file() {
            if is_image_file(source_path) {
                all_image_files.push(source_path.to_path_buf());
            }
        } else if source_path.is_dir() {
            let mut dir_files = find_image_files(source_path)?;
            all_image_files.append(&mut dir_files);
        } else if source.contains('*') || source.contains('?') {
            // Handle glob patterns
            for entry in glob::glob(source)? {
                let path = entry?;
                if path.is_file() && is_image_file(&path) {
                    all_image_files.push(path);
                }
            }
        }
    }

    // Sort all collected files for consistent ordering
    all_image_files.sort();

    // Remove duplicates (in case same file is specified multiple ways)
    all_image_files.dedup();

    Ok(all_image_files)
}

/// Process a single image with an existing session
fn process_single_image(
    config: &CutoutConfig,
    session: &mut Session,
    image_path: &Path,
    _session_creation_time: f64,
) -> Result<CutoutResult> {
    let start_time = Instant::now();

    verbose_println!(config, "🖼️  Processing: {}", image_path.display());

    // Load and preprocess the image
    let img = image::open(image_path)?;
    let original_size = img.dimensions();

    let input_array = preprocess_image_for_isnet_v2(&img)?;

    // Prepare input for the model
    let input_name = session.inputs[0].name.clone();
    let output_name = session.outputs[0].name.clone();
    let input_value = Value::from_array(input_array)
        .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;

    // Run inference
    let outputs = session
        .run(ort::inputs![input_name.as_str() => &input_value])
        .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

    // Extract the output tensor using ORT v2 API
    let output_view = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;

    // Extract the mask from the output (shape should be [1, 1, 1024, 1024])
    let mask_2d = output_view.slice(ndarray::s![0, 0, .., ..]);

    // Post-process the mask
    let mask = postprocess_mask(&mask_2d, original_size, config.post_process_mask)?;

    // Generate output paths
    let output_path = generate_output_path(image_path, config, "cutout", "png")?;
    let mask_path = if config.save_mask {
        Some(generate_output_path(image_path, config, "mask", "png")?)
    } else {
        None
    };

    // Create the cutout
    let cutout_result = if config.alpha_matting {
        apply_alpha_matting(
            &img,
            &mask,
            config.alpha_matting_foreground_threshold,
            config.alpha_matting_background_threshold,
            config.alpha_matting_erode_size,
        )?
    } else if let Some(bg_color) = config.background_color {
        create_cutout_with_background(&img, &mask, bg_color)?
    } else {
        create_cutout(&img, &mask)?
    };

    // Save the cutout (always PNG for transparency)
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    cutout_result.save(&output_path)?;

    // Save mask if requested
    if let Some(mask_path_val) = &mask_path {
        if let Some(parent) = Path::new(mask_path_val).parent() {
            fs::create_dir_all(parent)?;
        }
        mask.save(mask_path_val)?;
    }

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    verbose_println!(
        config,
        "✅ Processed {} in {:.1}ms → {}",
        image_path.display(),
        processing_time,
        output_path.display()
    );

    let cutout_result = CutoutResult {
        input_path: image_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        model_version: "isnet-general-use".to_string(),
        processing_time_ms: processing_time,
        mask_path: mask_path.map(|p| p.to_string_lossy().to_string()),
    };

    // Handle individual metadata output
    if !config.skip_metadata {
        handle_individual_metadata_output(config, image_path, &cutout_result)?;
    }

    Ok(cutout_result)
}
/// Generate output path for processed files
fn generate_output_path(
    input_path: &Path,
    config: &CutoutConfig,
    suffix: &str,
    extension: &str,
) -> Result<std::path::PathBuf> {
    let input_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let output_filename = format!("{input_stem}_{suffix}.{extension}");

    let output_path = if let Some(output_dir) = &config.output_dir {
        Path::new(output_dir).join(&output_filename)
    } else {
        input_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&output_filename)
    };

    Ok(output_path)
}

/// Handle individual metadata output for each processed image
fn handle_individual_metadata_output(
    config: &CutoutConfig,
    image_path: &Path,
    cutout_result: &CutoutResult,
) -> Result<()> {
    let metadata_path = get_metadata_path(image_path, config.output_dir.as_deref())?;

    // Load existing metadata or create new
    let mut metadata = load_or_create_metadata(&metadata_path)?;

    // Create cutout section with timestamp
    let cutout_section = CutoutSection {
        timestamp: Utc::now(),
        model_version: "isnet-general-use".to_string(),
        post_process_mask: config.post_process_mask,
        alpha_matting: config.alpha_matting,
        background_color: config.background_color,
        input_path: cutout_result.input_path.clone(),
        output_path: cutout_result.output_path.clone(),
        processing_time_ms: cutout_result.processing_time_ms,
        mask_path: cutout_result.mask_path.clone(),
    };

    // Update metadata with cutout section
    metadata.cutout = Some(serde_json::to_value(cutout_section)?);

    // Save updated metadata
    save_metadata(&metadata, &metadata_path)?;

    verbose_println!(config, "📋 Saved metadata to: {}", metadata_path.display());

    Ok(())
}

/// Process multiple images sequentially

pub fn run_cutout_processing(config: CutoutConfig) -> Result<usize> {
    // Collect all image files to process
    let image_files = collect_image_files_from_sources(&config.sources)?;

    if image_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No image files found in the specified sources"
        ));
    }

    verbose_println!(config, "🎯 Found {} image(s) to process", image_files.len());

    // Download model if needed
    let model_path = get_or_download_model(&ISNET_GENERAL_MODEL, config.verbose)?;

    // Determine optimal device
    let device_selection = determine_optimal_device(&config.device, image_files.len(), &config);
    verbose_println!(
        config,
        "🔧 Using device: {} ({})",
        device_selection.device,
        device_selection.reason
    );

    // Create session using unified configuration
    let session_config = SessionConfig {
        device: &device_selection.device,
        num_images: image_files.len(),
        verbose: config.verbose,
        suppress_warnings: false, // Allow warnings in cutout processing
    };
    let (mut session, session_creation_time) = create_onnx_session(
        ModelSource::FilePath(model_path.to_str().unwrap()),
        &session_config,
        &config,
    )?;

    // Process all images
    let mut results = Vec::new();
    let total_start = Instant::now();

    for image_path in &image_files {
        match process_single_image(&config, &mut session, image_path, session_creation_time) {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("❌ Failed to process {}: {}", image_path.display(), e);
                // Continue with other images
            }
        }
    }

    let total_time = total_start.elapsed().as_secs_f64() * 1000.0;

    // Individual metadata files are already created during processing
    // No need for batch metadata output with the new shared system

    if config.verbose && results.len() > 1 {
        println!(
            "🏁 Processed {} images in {:.1}ms (avg: {:.1}ms per image)",
            results.len(),
            total_time,
            total_time / results.len() as f64
        );
    }

    Ok(results.len())
}
