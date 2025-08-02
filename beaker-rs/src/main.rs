use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "beaker")]
#[command(about = "Detect bird heads in images using YOLOv8")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input image path (when not using subcommands)
    #[arg(short, long)]
    image: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run bird head detection on an image
    Detect {
        /// Input image or directory path
        source: String,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<String>,

        /// Confidence threshold
        #[arg(short, long, default_value = "0.25")]
        confidence: f32,

        /// Device to use for inference
        #[arg(short, long, default_value = "auto")]
        device: String,

        /// Skip creating square crops around detected heads
        #[arg(long)]
        skip_crop: bool,

        /// Save detection results with bounding boxes
        #[arg(long)]
        save_bounding_box: bool,

        /// Show detection results
        #[arg(long)]
        show: bool,
    },
    /// Run benchmark tests
    Benchmark,
    /// Show version information
    Version,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Detect {
            source,
            output,
            confidence,
            device,
            skip_crop,
            save_bounding_box,
            show,
        }) => {
            println!("🔍 Would detect bird heads in: {}", source);
            println!("   Confidence threshold: {}", confidence);
            println!("   Device: {}", device);
            if *skip_crop {
                println!("   Skipping crop generation");
            }
            if *save_bounding_box {
                println!("   Will save bounding box images");
            }
            if *show {
                println!("   Will show results");
            }
            if let Some(output) = output {
                println!("   Output directory: {}", output);
            }
            println!("   [Not implemented yet - this is a skeleton]");
        }
        Some(Commands::Benchmark) => {
            println!("🏃 Would run benchmark tests");
            println!("   [Not implemented yet - this is a skeleton]");
        }
        Some(Commands::Version) => {
            println!("beaker {}", env!("CARGO_PKG_VERSION"));
            println!("Rust implementation of YOLOv8 bird head detection");
        }
        None => {
            if let Some(image) = &cli.image {
                println!("🔍 Would detect bird heads in: {}", image);
                println!("   [Not implemented yet - this is a skeleton]");
                println!("   Hint: Use 'beaker detect {}' for more options", image);
            } else {
                // Show help if no command or image specified
                use clap::CommandFactory;
                let mut cmd = Cli::command();
                cmd.print_help().unwrap();
            }
        }
    }
}
