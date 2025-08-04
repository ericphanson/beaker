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
use cutout_processing::{run_cutout_processing, CUTOUT_MODEL_INFO};
use head_detection::{run_head_detection, MODEL_VERSION};
use std::io::Write;

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Detect bird heads in images
    Head(HeadCommand),

    /// Remove backgrounds from images
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
            let sources_desc = if head_cmd.sources.len() == 1 {
                head_cmd.sources[0].clone()
            } else {
                format!("{} inputs", head_cmd.sources.len())
            };

            info!(
                "🔍 Head detection: {} | conf: {} | IoU: {} | device: {}",
                sources_desc, head_cmd.confidence, head_cmd.iou_threshold, cli.global.device
            );

            // Build outputs list
            let mut outputs = Vec::new();
            if head_cmd.crop {
                outputs.push("crops");
            }
            if head_cmd.bounding_box {
                outputs.push("bounding-boxes");
            }
            if !cli.global.no_metadata {
                outputs.push("metadata");
            }

            if outputs.is_empty() {
                info!("   Outputs: none");
            } else {
                info!("   Outputs: {}", outputs.join(", "));
            }

            let internal_config =
                HeadDetectionConfig::from_args(cli.global.clone(), head_cmd.clone());
            match run_head_detection(internal_config) {
                Ok(_) => {}
                Err(e) => {
                    error!("❌ Detection failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Cutout(cutout_cmd)) => {
            let sources_desc = if cutout_cmd.sources.len() == 1 {
                cutout_cmd.sources[0].clone()
            } else {
                format!("{} inputs", cutout_cmd.sources.len())
            };

            info!(
                "✂️  Background removal: {} | device: {}",
                sources_desc, cli.global.device
            );

            // Build features list
            let mut features = Vec::new();
            if cutout_cmd.post_process {
                features.push("post-process");
            }
            if cutout_cmd.alpha_matting {
                features.push("alpha-matting");
            }
            if cutout_cmd.save_mask {
                features.push("save-mask");
            }
            if !features.is_empty() {
                info!("   Features: {}", features.join(", "));
            }

            // Build outputs list
            let mut outputs = Vec::new();
            outputs.push("cutout"); // Always produces cutout
            if cutout_cmd.save_mask {
                outputs.push("mask");
            }
            if !cli.global.no_metadata {
                outputs.push("metadata");
            }
            info!("   Outputs: {}", outputs.join(", "));

            let internal_config = CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone());
            match run_cutout_processing(internal_config) {
                Ok(_) => {}
                Err(e) => {
                    error!("❌ Background removal failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Version) => {
            // Print version information
            println!("beaker v{}", env!("CARGO_PKG_VERSION"));
            println!("Head model version: {}", MODEL_VERSION.trim());
            println!("Cutout model version: {}", CUTOUT_MODEL_INFO.name.trim());
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
