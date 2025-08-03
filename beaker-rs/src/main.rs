use clap::Parser;

mod cutout_postprocessing;
mod cutout_preprocessing;
mod cutout_processing;
mod head_detection;
mod model_cache;
mod onnx_session;
mod shared_metadata;
mod yolo_postprocessing;
mod yolo_preprocessing;

use cutout_processing::{run_cutout_processing, CutoutConfig};
use head_detection::{run_head_detection, HeadDetectionConfig, MODEL_VERSION};

/// Parse RGBA color from string like "255,255,255,255"
fn parse_rgba_color(s: &str) -> Result<[u8; 4], String> {
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

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Detect bird heads in images
    Head {
        /// Path(s) to input images or directories. Supports glob patterns like *.jpg
        #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
        sources: Vec<String>,

        /// Confidence threshold for detections (0.0-1.0)
        #[arg(short, long, default_value = "0.25")]
        confidence: f32,

        /// IoU threshold for non-maximum suppression (0.0-1.0)
        #[arg(long, default_value = "0.45")]
        iou_threshold: f32,

        /// Create a square crop of the detected head
        #[arg(long)]
        crop: bool,

        /// Save an image with bounding boxes drawn
        #[arg(long)]
        bounding_box: bool,

        /// Device to use for inference (auto, cpu, coreml)
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Remove backgrounds from images using AI segmentation
    Cutout {
        /// Path(s) to input images or directories. Supports glob patterns like *.jpg
        #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
        sources: Vec<String>,

        /// Device to use for inference (auto, cpu, coreml)
        #[arg(long, default_value = "auto")]
        device: String,

        /// Apply post-processing to smooth mask edges
        #[arg(long)]
        post_process: bool,

        /// Use alpha matting for better edge quality
        #[arg(long)]
        alpha_matting: bool,

        /// Foreground threshold for alpha matting (0-255)
        #[arg(long, default_value = "240")]
        alpha_matting_foreground_threshold: u8,

        /// Background threshold for alpha matting (0-255)
        #[arg(long, default_value = "10")]
        alpha_matting_background_threshold: u8,

        /// Erosion size for alpha matting
        #[arg(long, default_value = "10")]
        alpha_matting_erode_size: u32,

        /// Background color as RGBA (e.g., "255,255,255,255" for white)
        #[arg(long, value_parser = parse_rgba_color)]
        background_color: Option<[u8; 4]>,

        /// Save the segmentation mask as a separate image
        #[arg(long)]
        save_mask: bool,
    },

    /// Show version information
    Version,
}

#[derive(Parser)]
#[command(name = "beaker")]
#[command(about = "Bird detection and analysis toolkit")]
struct Cli {
    /// Global output directory (overrides default placement next to input)
    #[arg(long, global = true)]
    output_dir: Option<String>,

    /// Skip creating metadata output files
    #[arg(long, global = true)]
    no_metadata: bool,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

fn main() {
    // Initialize tracing subscriber for ORT logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Head {
            sources,
            confidence,
            iou_threshold,
            device,
            crop,
            bounding_box,
        }) => {
            // Display what we're processing (only if verbose)
            if cli.verbose {
                if sources.len() == 1 {
                    println!("🔍 Running head detection on: {}", sources[0]);
                } else {
                    println!("🔍 Running head detection on {} inputs:", sources.len());
                    for source in sources {
                        println!("   • {source}");
                    }
                }

                println!(
                    "   Model: embedded ONNX model (version: {})",
                    MODEL_VERSION.trim()
                );
                println!("   Confidence threshold: {confidence}");
                println!("   IoU threshold: {iou_threshold}");
                println!("   Device: {device}");
                if *crop {
                    println!("   Will create head crops");
                }
                if *bounding_box {
                    println!("   Will save bounding box images");
                }
                if !cli.no_metadata {
                    println!("   Will create metadata output");
                }
                if let Some(output_dir) = &cli.output_dir {
                    println!("   Output directory: {output_dir}");
                }
            }

            // Run actual detection
            let config = HeadDetectionConfig {
                sources: sources.clone(),
                confidence: *confidence,
                iou_threshold: *iou_threshold,
                device: device.to_string(),
                output_dir: cli.output_dir,
                crop: *crop,
                bounding_box: *bounding_box,
                skip_metadata: cli.no_metadata,
                verbose: cli.verbose,
            };
            match run_head_detection(config) {
                Ok(detections) => {
                    if cli.verbose {
                        println!("✅ Found {detections} detections");
                    }
                }
                Err(e) => {
                    eprintln!("❌ Detection failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Cutout {
            sources,
            device,
            post_process,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            background_color,
            save_mask,
        }) => {
            // Display what we're processing (only if verbose)
            if cli.verbose {
                if sources.len() == 1 {
                    println!("✂️  Running background removal on: {}", sources[0]);
                } else {
                    println!(
                        "✂️  Running background removal on {} inputs:",
                        sources.len()
                    );
                    for source in sources {
                        println!("   • {source}");
                    }
                }

                println!("   Model: ISNet General Use");
                println!("   Device: {device}");
                if *post_process {
                    println!("   Will apply mask post-processing");
                }
                if *alpha_matting {
                    println!("   Will use alpha matting (fg: {alpha_matting_foreground_threshold}, bg: {alpha_matting_background_threshold}, erode: {alpha_matting_erode_size})");
                }
                if let Some(bg_color) = background_color {
                    println!(
                        "   Background color: RGBA({}, {}, {}, {})",
                        bg_color[0], bg_color[1], bg_color[2], bg_color[3]
                    );
                }
                if *save_mask {
                    println!("   Will save segmentation masks");
                }
                if !cli.no_metadata {
                    println!("   Will create metadata output");
                }
                if let Some(output_dir) = &cli.output_dir {
                    println!("   Output directory: {output_dir}");
                }
            }

            // Run cutout processing
            let config = CutoutConfig {
                sources: sources.clone(),
                device: device.to_string(),
                output_dir: cli.output_dir,
                post_process_mask: *post_process,
                alpha_matting: *alpha_matting,
                alpha_matting_foreground_threshold: *alpha_matting_foreground_threshold,
                alpha_matting_background_threshold: *alpha_matting_background_threshold,
                alpha_matting_erode_size: *alpha_matting_erode_size,
                background_color: *background_color,
                save_mask: *save_mask,
                skip_metadata: cli.no_metadata,
                verbose: cli.verbose,
            };
            match run_cutout_processing(config) {
                Ok(processed) => {
                    if cli.verbose {
                        println!("✅ Processed {processed} images");
                    }
                }
                Err(e) => {
                    eprintln!("❌ Background removal failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Version) => {
            // Print version information
            println!("beaker {}", env!("CARGO_PKG_VERSION"));
            println!("Head model version: {}", MODEL_VERSION.trim());
            println!("Repository: {}", env!("CARGO_PKG_REPOSITORY"));
        }
        None => {
            // Show help if no command specified
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            cmd.print_help().unwrap();
        }
    }
}
