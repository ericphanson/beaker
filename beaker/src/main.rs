use clap::Parser;
use env_logger::Builder;
use env_logger::Env;
use log::{error, info, Level};

mod config;
mod cutout_postprocessing;
mod cutout_preprocessing;
mod cutout_processing;
mod head_detection;
mod image_input;
mod model_cache;
mod model_processing;
mod onnx_session;
mod output_manager;
mod shared_metadata;
mod yolo_postprocessing;
mod yolo_preprocessing;

use colored::*;
use config::{CutoutCommand, CutoutConfig, GlobalArgs, HeadCommand, HeadDetectionConfig};
use cutout_processing::run_cutout_processing;
use head_detection::{run_head_detection, MODEL_VERSION};
use std::io::Write;

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

fn get_log_level_from_verbosity(
    verbosity: clap_verbosity_flag::Verbosity<clap_verbosity_flag::ErrorLevel>,
) -> log::LevelFilter {
    let base_level = verbosity.log_level_filter();
    let adjusted_level = match base_level {
        log::LevelFilter::Off => log::LevelFilter::Off, // -qq -> OFF
        log::LevelFilter::Error => log::LevelFilter::Warn, // default -> WARN
        log::LevelFilter::Warn => log::LevelFilter::Info, // -v -> INFO
        log::LevelFilter::Info => log::LevelFilter::Debug, // -vv -> DEBUG
        log::LevelFilter::Debug => log::LevelFilter::Trace, // -vvv -> TRACE
        log::LevelFilter::Trace => log::LevelFilter::Trace, // -vvvv -> TRACE (max)
    };

    // But we also need to handle -q -> ERROR
    // clap-verbosity-flag doesn't give us a way to distinguish between default and -q
    // So we need to check the quiet flag directly
    if verbosity.is_silent() {
        log::LevelFilter::Error // -q -> ERROR
    } else {
        adjusted_level
    }
}

fn main() {
    let cli = Cli::parse();

    // If user didn't pass -v/-q and RUST_LOG is set, honor the env var.
    let use_env = !cli.global.verbosity.is_present() && std::env::var_os("RUST_LOG").is_some();

    let mut logger = if use_env {
        Builder::from_env(Env::default())
    } else {
        let level_filter = get_log_level_from_verbosity(cli.global.verbosity.clone());

        let mut b = Builder::new();
        b.filter_level(level_filter);
        b
    };

    logger
        .format(|buf, record| {
            let level_str = match record.level() {
                Level::Error => "ERROR".red().bold().to_string(),
                Level::Warn => "WARN".yellow().to_string(),
                Level::Info => "INFO".green().to_string(),
                Level::Debug => "DEBUG".blue().to_string(),
                Level::Trace => "TRACE".magenta().to_string(),
            };
            writeln!(buf, "[{}] {}", level_str, record.args())
        })
        .init();

    match &cli.command {
        Some(Commands::Head(head_cmd)) => {
            // Display what we're processing (use info! instead of verbose check)
            if head_cmd.sources.len() == 1 {
                info!("🔍 Running head detection on: {}", head_cmd.sources[0]);
            } else {
                info!(
                    "🔍 Running head detection on {} inputs:",
                    head_cmd.sources.len()
                );
                for source in &head_cmd.sources {
                    info!("   • {source}");
                }
            }

            info!(
                "   Model: embedded ONNX model (version: {})",
                MODEL_VERSION.trim()
            );
            info!("   Confidence threshold: {}", head_cmd.confidence);
            info!("   IoU threshold: {}", head_cmd.iou_threshold);
            info!("   Device: {}", cli.global.device);
            if head_cmd.crop {
                info!("   Will create head crops");
            }
            if head_cmd.bounding_box {
                info!("   Will save bounding box images");
            }
            if !cli.global.no_metadata {
                log::debug!("   Will create metadata output");
            }
            if let Some(output_dir) = &cli.global.output_dir {
                info!("   Output directory: {output_dir}");
            }

            // Convert CLI command to internal config and run detection
            let internal_config =
                HeadDetectionConfig::from_args(cli.global.clone(), head_cmd.clone());
            match run_head_detection(internal_config) {
                Ok(_detections) => {
                    // Detection results already logged by the processing framework
                }
                Err(e) => {
                    error!("❌ Detection failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Cutout(cutout_cmd)) => {
            // Display what we're processing (use info! instead of verbose check)
            if cutout_cmd.sources.len() == 1 {
                info!(
                    "✂️  Running background removal on: {}",
                    cutout_cmd.sources[0]
                );
            } else {
                info!(
                    "✂️  Running background removal on {} inputs:",
                    cutout_cmd.sources.len()
                );
                for source in &cutout_cmd.sources {
                    info!("   • {source}");
                }
            }

            info!("   Model: ISNet General Use");
            info!("   Device: {}", cli.global.device);
            if cutout_cmd.post_process {
                info!("   Will apply mask post-processing");
            }
            if cutout_cmd.alpha_matting {
                info!(
                    "   Will use alpha matting (fg: {}, bg: {}, erode: {})",
                    cutout_cmd.alpha_matting_foreground_threshold,
                    cutout_cmd.alpha_matting_background_threshold,
                    cutout_cmd.alpha_matting_erode_size
                );
            }
            if let Some(bg_color) = &cutout_cmd.background_color {
                info!(
                    "   Background color: RGBA({}, {}, {}, {})",
                    bg_color[0], bg_color[1], bg_color[2], bg_color[3]
                );
            }
            if cutout_cmd.save_mask {
                info!("   Will save segmentation masks");
            }
            if !cli.global.no_metadata {
                log::debug!("   Will create metadata output");
            }
            if let Some(output_dir) = &cli.global.output_dir {
                info!("   Output directory: {output_dir}");
            }

            // Convert CLI command to internal config and run cutout processing
            let internal_config = CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone());
            match run_cutout_processing(internal_config) {
                Ok(_processed) => {
                    // Processing results already logged by the processing framework
                }
                Err(e) => {
                    error!("❌ Background removal failed: {e}");
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
