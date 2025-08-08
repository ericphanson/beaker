use clap::Parser;
use env_logger::Builder;
use env_logger::Env;
use log::{error, info, Level};

mod cache_common;
mod color_utils;
mod config;
mod cutout_postprocessing;
mod cutout_preprocessing;
mod cutout_processing;
mod head_detection;
mod image_input;
mod model_access;
mod model_processing;
mod onnx_session;
mod output_manager;
mod progress;
mod shared_metadata;
mod yolo_postprocessing;
mod yolo_preprocessing;

use color_utils::{colors, symbols};
use config::{CutoutCommand, CutoutConfig, GlobalArgs, HeadCommand, HeadDetectionConfig};
use cutout_processing::{get_default_cutout_model_info, run_cutout_processing};
use head_detection::{run_head_detection, MODEL_VERSION};
use progress::global_mp;
use std::io::Write;

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Detect & crop bird heads in bird images
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

    // Initialize color configuration early
    color_utils::init_color_config(cli.global.no_color);

    // If user didn't pass -v/-q and RUST_LOG is set, honor the env var.
    let use_env = !cli.global.verbosity.is_present() && std::env::var_os("RUST_LOG").is_some();

    let mut logger = if use_env {
        Builder::from_env(Env::default())
    } else {
        let level_filter = get_log_level_from_verbosity(cli.global.verbosity);

        let mut b = Builder::new();
        b.filter_level(level_filter);
        b
    };

    let base_logger = logger
        .format(|buf, record| {
            let level_str = match record.level() {
                Level::Error => colors::error_level("ERROR"),
                Level::Warn => colors::warning_level("WARN"),
                Level::Info => colors::info_level("INFO"),
                Level::Debug => colors::debug_level("DEBUG"),
                Level::Trace => colors::trace_level("TRACE"),
            };
            if record.level() == Level::Info {
                // We skip printing [INFO] for info level to reduce noise
                writeln!(buf, "{}", record.args())
            } else {
                // For other levels, include the level in the output
                writeln!(buf, "[{}] {}", level_str, record.args())
            }
        })
        .build();

    // this will suspend the global multi-progress bar when logging
    indicatif_log_bridge::LogWrapper::new((*global_mp()).clone(), base_logger)
        .try_init()
        .unwrap();

    // if !use_env {
    // log::set_max_level(get_log_level_from_verbosity(cli.global.verbosity));
    // }

    match &cli.command {
        Some(Commands::Head(head_cmd)) => {
            // Build outputs list
            let mut outputs = Vec::new();
            if head_cmd.crop {
                outputs.push("crops");
            }
            if head_cmd.bounding_box {
                outputs.push("bounding-boxes");
            }
            if cli.global.metadata {
                outputs.push("metadata");
            }

            let output_str = if outputs.is_empty() {
                "".to_string()
            } else {
                format!(" | outputs: {}", outputs.join(", "))
            };

            info!(
                "{} Head detection | conf: {} | IoU: {} | device: {}{}",
                symbols::head_detection_start(),
                head_cmd.confidence,
                head_cmd.iou_threshold,
                cli.global.device,
                output_str
            );

            if outputs.is_empty() {
                error!("No outputs requested! Pass at least one of `--metadata`, `--crop`, or `--bounding-box`.");
                std::process::exit(1);
            } else {
                let internal_config =
                    HeadDetectionConfig::from_args(cli.global.clone(), head_cmd.clone());
                match run_head_detection(internal_config) {
                    Ok(_) => {}
                    Err(e) => {
                        error!("{} Detection failed: {e}", symbols::operation_failed());
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Cutout(cutout_cmd)) => {
            // Build features list
            let mut features = Vec::new();
            if cutout_cmd.post_process {
                features.push("post-process");
            }
            if cutout_cmd.alpha_matting {
                features.push("alpha-matting");
            }

            let feature_str = if features.is_empty() {
                "".to_string()
            } else {
                format!(" | features: {}", features.join(", "))
            };
            // Build outputs list
            let mut outputs = Vec::new();
            outputs.push("cutout"); // Always produces cutout
            if cutout_cmd.save_mask {
                outputs.push("mask");
            }
            if cli.global.metadata {
                outputs.push("metadata");
            }

            let output_str = if outputs.is_empty() {
                "".to_string()
            } else {
                format!(" | outputs: {}", outputs.join(", "))
            };

            info!(
                "{} Background removal | device: {}{}{}",
                symbols::background_removal_start(),
                cli.global.device,
                feature_str,
                output_str
            );

            let internal_config = CutoutConfig::from_args(cli.global.clone(), cutout_cmd.clone());
            match run_cutout_processing(internal_config) {
                Ok(_) => {}
                Err(e) => {
                    error!(
                        "{} Background removal failed: {e}",
                        symbols::operation_failed()
                    );
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Version) => {
            // Print version information
            println!("beaker v{}", env!("CARGO_PKG_VERSION"));
            println!("Head model version: {}", MODEL_VERSION.trim());
            println!(
                "Cutout model version: {}",
                get_default_cutout_model_info().name.trim()
            );
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
