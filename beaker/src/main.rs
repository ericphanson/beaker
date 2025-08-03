use clap::Parser;

mod config;
mod cutout_postprocessing;
mod cutout_preprocessing;
mod cutout_processing;
mod head_detection;
mod image_input;
mod model_cache;
mod onnx_session;
mod shared_metadata;
mod yolo_postprocessing;
mod yolo_preprocessing;

use config::{CutoutCommand, CutoutConfig, GlobalArgs, HeadCommand, HeadDetectionConfig};
use cutout_processing::run_cutout_processing;
use head_detection::{run_head_detection, MODEL_VERSION};

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Detect bird heads in images
    Head(HeadCommand),

    /// Remove backgrounds from images using AI segmentation
    Cutout(CutoutCommand),

    /// Show version information
    Version,
}

#[derive(Parser)]
#[command(name = "beaker")]
#[command(about = "Bird detection and analysis toolkit")]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[command(subcommand)]
    command: Option<Commands>,
}

fn main() {
    // Initialize tracing subscriber for ORT logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Head(head_cmd)) => {
            // Display what we're processing (only if verbose)
            if cli.global.verbose {
                if head_cmd.sources.len() == 1 {
                    println!("🔍 Running head detection on: {}", head_cmd.sources[0]);
                } else {
                    println!(
                        "🔍 Running head detection on {} inputs:",
                        head_cmd.sources.len()
                    );
                    for source in &head_cmd.sources {
                        println!("   • {source}");
                    }
                }

                println!(
                    "   Model: embedded ONNX model (version: {})",
                    MODEL_VERSION.trim()
                );
                println!("   Confidence threshold: {}", head_cmd.confidence);
                println!("   IoU threshold: {}", head_cmd.iou_threshold);
                println!("   Device: {}", cli.global.device);
                if head_cmd.crop {
                    println!("   Will create head crops");
                }
                if head_cmd.bounding_box {
                    println!("   Will save bounding box images");
                }
                if !cli.global.no_metadata {
                    println!("   Will create metadata output");
                }
                if let Some(output_dir) = &cli.global.output_dir {
                    println!("   Output directory: {output_dir}");
                }
            }

            // Convert CLI command to internal config and run detection
            let internal_config =
                HeadDetectionConfig::from_args(cli.global.clone(), head_cmd.clone());
            match run_head_detection(internal_config) {
                Ok(detections) => {
                    if cli.global.verbose {
                        println!("✅ Found {detections} detections");
                    }
                }
                Err(e) => {
                    eprintln!("❌ Detection failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Cutout(cutout_cmd)) => {
            // Display what we're processing (only if verbose)
            if cli.global.verbose {
                if cutout_cmd.sources.len() == 1 {
                    println!(
                        "✂️  Running background removal on: {}",
                        cutout_cmd.sources[0]
                    );
                } else {
                    println!(
                        "✂️  Running background removal on {} inputs:",
                        cutout_cmd.sources.len()
                    );
                    for source in &cutout_cmd.sources {
                        println!("   • {source}");
                    }
                }

                println!("   Model: ISNet General Use");
                println!("   Device: {}", cli.global.device);
                if cutout_cmd.post_process {
                    println!("   Will apply mask post-processing");
                }
                if cutout_cmd.alpha_matting {
                    println!(
                        "   Will use alpha matting (fg: {}, bg: {}, erode: {})",
                        cutout_cmd.alpha_matting_foreground_threshold,
                        cutout_cmd.alpha_matting_background_threshold,
                        cutout_cmd.alpha_matting_erode_size
                    );
                }
                if let Some(bg_color) = &cutout_cmd.background_color {
                    println!(
                        "   Background color: RGBA({}, {}, {}, {})",
                        bg_color[0], bg_color[1], bg_color[2], bg_color[3]
                    );
                }
                if cutout_cmd.save_mask {
                    println!("   Will save segmentation masks");
                }
                if !cli.global.no_metadata {
                    println!("   Will create metadata output");
                }
                if let Some(output_dir) = &cli.global.output_dir {
                    println!("   Output directory: {output_dir}");
                }
            }

            // Convert CLI command to internal config and run cutout processing
            let internal_config = CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone());
            match run_cutout_processing(internal_config) {
                Ok(processed) => {
                    if cli.global.verbose {
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
