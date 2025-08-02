use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let model_path = Path::new(&out_dir).join("bird-head-detector.onnx");

    // Only download if the model doesn't exist or if we're in CI
    let force_download = env::var("CI").is_ok() || !model_path.exists();

    if force_download {
        println!("cargo:warning=Downloading latest ONNX model from GitHub releases...");
        download_latest_model(&model_path)?;
    } else {
        println!("cargo:warning=Using cached ONNX model");
    }

    // Tell cargo to rebuild if this script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to rebuild if we're in CI (to always get latest model)
    if env::var("CI").is_ok() {
        println!("cargo:rerun-if-env-changed=CI");
    }

    Ok(())
}

fn download_latest_model(output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // GitHub API to get the latest release
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases/latest";

    println!(
        "cargo:warning=Fetching latest release info from: {}",
        api_url
    );

    let client = ureq::Agent::new();
    let response = client
        .get(api_url)
        .set("User-Agent", "beaker-rs-build-script/0.1.0")
        .call()?;

    let release_info: serde_json::Value = response.into_json()?;

    // Find the ONNX model asset
    let assets = release_info["assets"]
        .as_array()
        .ok_or("No assets found in release")?;

    let onnx_asset = assets
        .iter()
        .find(|asset| {
            asset["name"]
                .as_str()
                .map(|name| name.ends_with(".onnx"))
                .unwrap_or(false)
        })
        .ok_or("No ONNX model found in latest release")?;

    let download_url = onnx_asset["browser_download_url"]
        .as_str()
        .ok_or("No download URL found for ONNX asset")?;

    let model_name = onnx_asset["name"]
        .as_str()
        .unwrap_or("bird-head-detector.onnx");

    println!(
        "cargo:warning=Downloading {} from: {}",
        model_name, download_url
    );

    // Download the model
    let response = client
        .get(download_url)
        .set("User-Agent", "beaker-rs-build-script/0.1.0")
        .call()?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(output_path)?;
    std::io::copy(&mut reader, &mut file)?;

    println!(
        "cargo:warning=Successfully downloaded model to: {}",
        output_path.display()
    );

    Ok(())
}
