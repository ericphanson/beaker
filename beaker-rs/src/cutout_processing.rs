use anyhow::Result;
use image::GenericImageView;
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    logging::LogLevel,
    session::Session,
    value::Value,
};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::cutout_postprocessing::{
    apply_alpha_matting, create_cutout, create_cutout_with_background, postprocess_mask,
};
use crate::cutout_preprocessing::preprocess_image_for_isnet_v2;
use crate::model_cache::{get_or_download_model, ISNET_GENERAL_MODEL};

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
    pub model_version: String,
    pub post_process_mask: bool,
    pub alpha_matting: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_color: Option<[u8; 4]>,
    pub results: Vec<CutoutResult>,
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

/// Determine optimal device based on number of images and user preference
fn determine_optimal_device(config: &CutoutConfig, num_images: usize) -> String {
    const COREML_THRESHOLD: usize = 3; // Use CoreML for 3+ images

    match config.device.as_str() {
        "auto" => {
            if num_images >= COREML_THRESHOLD {
                "coreml".to_string()
            } else {
                "cpu".to_string()
            }
        }
        device => device.to_string(),
    }
}

/// Create an ONNX Runtime session with the specified device
fn create_session(
    config: &CutoutConfig,
    device: &str,
    model_path: &Path,
) -> Result<(Session, f64)> {
    // Determine execution provider based on device with availability checking
    let execution_providers = match device {
        "coreml" => match CoreMLExecutionProvider::default().is_available() {
            Ok(true) => vec![
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ],
            _ => {
                verbose_println!(config, "⚠️  CoreML not available, falling back to CPU");
                vec![CPUExecutionProvider::default().build()]
            }
        },
        "cpu" => vec![CPUExecutionProvider::default().build()],
        _ => {
            verbose_println!(config, "⚠️  Unknown device '{}', using CPU", device);
            vec![CPUExecutionProvider::default().build()]
        }
    };

    // Store EP info for logging before moving the vector
    let ep_names: Vec<String> = execution_providers
        .iter()
        .map(|ep| format!("{ep:?}"))
        .collect();

    // Set log level to suppress warnings unless verbose mode is enabled
    let log_level = if config.verbose {
        LogLevel::Warning // Show warnings in verbose mode
    } else {
        LogLevel::Error // Only show errors in normal mode
    };

    // Load the model using ORT v2 API
    let session_start = Instant::now();
    let model_bytes =
        fs::read(model_path).map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;

    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
        .with_log_level(log_level)
        .map_err(|e| anyhow::anyhow!("Failed to set log level: {}", e))?
        .with_execution_providers(execution_providers)
        .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {}", e))?
        .commit_from_memory(&model_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load model from memory: {}", e))?;

    let session_time = session_start.elapsed().as_secs_f64() * 1000.0;

    verbose_println!(
        config,
        "🤖 Loaded ISNet model ({} bytes) in {:.3}ms using {:?}",
        fs::metadata(model_path)?.len(),
        session_time,
        ep_names
    );

    Ok((session, session_time))
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

    Ok(CutoutResult {
        input_path: image_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        model_version: "isnet-general-use".to_string(),
        processing_time_ms: processing_time,
        mask_path: mask_path.map(|p| p.to_string_lossy().to_string()),
    })
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

/// Handle metadata output for processed images
fn handle_metadata_output(config: &CutoutConfig, results: &[CutoutResult]) -> Result<()> {
    if config.skip_metadata || results.is_empty() {
        return Ok(());
    }

    let output = CutoutOutput {
        cutout: CutoutSection {
            model_version: "isnet-general-use".to_string(),
            post_process_mask: config.post_process_mask,
            alpha_matting: config.alpha_matting,
            background_color: config.background_color,
            results: results.to_vec(),
        },
    };

    // For single image, save metadata next to the output
    // For multiple images, save in output directory or first image's directory
    let metadata_path = if results.len() == 1 {
        let first_result = &results[0];
        let output_path = Path::new(&first_result.output_path);
        let stem = output_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        output_path.with_file_name(format!("{stem}-beaker.toml"))
    } else if let Some(output_dir) = &config.output_dir {
        Path::new(output_dir).join("cutout-batch-beaker.toml")
    } else {
        Path::new("cutout-batch-beaker.toml").to_path_buf()
    };

    let toml_content = toml::to_string_pretty(&output)?;
    fs::write(&metadata_path, toml_content)?;

    verbose_println!(config, "📋 Saved metadata to: {}", metadata_path.display());

    Ok(())
}

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
    let device = determine_optimal_device(&config, image_files.len());
    verbose_println!(config, "🔧 Using device: {}", device);

    // Create session
    let (mut session, session_creation_time) = create_session(&config, &device, &model_path)?;

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

    // Handle metadata output
    handle_metadata_output(&config, &results)?;

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
