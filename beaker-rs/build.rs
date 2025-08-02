use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let model_path = Path::new(&out_dir).join("bird-head-detector.onnx");

    // Check if we have a cached model from CI
    let cache_dir = env::var("ONNX_MODEL_CACHE_DIR").ok();
    let cached_model = cache_dir
        .as_ref()
        .map(|dir| Path::new(dir).join("bird-head-detector.onnx"));
    let cached_version = cache_dir
        .as_ref()
        .map(|dir| Path::new(dir).join("bird-head-detector.version"));

    // Check if cached model is up-to-date
    let mut use_cache = false;
    if let (Some(cached_path), Some(version_path)) = (&cached_model, &cached_version) {
        if cached_path.exists() && version_path.exists() {
            if let Ok(cached_tag) = fs::read_to_string(version_path) {
                if let Ok(latest_tag) = get_latest_release_tag() {
                    if cached_tag.trim() == latest_tag.trim() {
                        println!(
                            "cargo:warning=Using cached ONNX model (version: {})",
                            cached_tag.trim()
                        );
                        fs::copy(cached_path, &model_path)?;
                        use_cache = true;
                    } else {
                        println!(
                            "cargo:warning=Cache outdated (cached: {}, latest: {}), downloading new model",
                            cached_tag.trim(), latest_tag.trim()
                        );
                    }
                } else {
                    println!("cargo:warning=Could not check latest release, using cached model");
                    fs::copy(cached_path, &model_path)?;
                    use_cache = true;
                }
            }
        }
    }

    if use_cache {
        return Ok(());
    }

    // Only download if the model doesn't exist or if we're in CI
    let force_download = env::var("CI").is_ok() || !model_path.exists();

    if force_download {
        println!("cargo:warning=Downloading latest ONNX model from GitHub releases...");
        let release_tag = download_latest_model(&model_path)?;

        // Cache the model and version for CI if cache directory is set
        if let Some(cached_path) = cached_model {
            if let Some(parent) = cached_path.parent() {
                fs::create_dir_all(parent)?;
                fs::copy(&model_path, &cached_path)?;

                // Save the version information
                if let Some(version_path) = cached_version {
                    fs::write(&version_path, &release_tag)?;
                }

                println!(
                    "cargo:warning=Cached model (version: {}) for future builds at: {}",
                    release_tag,
                    cached_path.display()
                );
            }
        }
    } else {
        println!("cargo:warning=Using existing ONNX model");
    }

    // Tell cargo to rebuild if this script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to rebuild if we're in CI (to check for cache updates)
    if env::var("CI").is_ok() {
        println!("cargo:rerun-if-env-changed=CI");
        println!("cargo:rerun-if-env-changed=ONNX_MODEL_CACHE_DIR");
    }

    Ok(())
}
fn get_latest_release_tag() -> Result<String, Box<dyn std::error::Error>> {
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases/latest";

    let client = ureq::Agent::new();
    let response = client
        .get(api_url)
        .set("User-Agent", "beaker-rs-build-script/0.1.0")
        .call()?;

    let release_info: serde_json::Value = response.into_json()?;

    let tag_name = release_info["tag_name"]
        .as_str()
        .ok_or("No tag_name found in release")?;

    Ok(tag_name.to_string())
}

fn download_latest_model(output_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // GitHub API to get the latest release
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases/latest";

    println!("cargo:warning=Fetching latest release info from: {api_url}");

    let client = ureq::Agent::new();
    let response = client
        .get(api_url)
        .set("User-Agent", "beaker-rs-build-script/0.1.0")
        .call()?;

    let release_info: serde_json::Value = response.into_json()?;

    // Get the release tag
    let tag_name = release_info["tag_name"]
        .as_str()
        .ok_or("No tag_name found in release")?
        .to_string();

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

    println!("cargo:warning=Downloading {model_name} (version: {tag_name}) from: {download_url}");

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

    Ok(tag_name)
}
