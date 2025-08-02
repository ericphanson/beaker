use clap::Parser;

mod head_detection;
mod raw_processing;
mod yolo_postprocessing;
mod yolo_preprocessing;

use head_detection::{run_head_detection, HeadDetectionConfig, MODEL_VERSION};

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
