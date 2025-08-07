use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let model_path = Path::new(&out_dir).join("bird-head-detector.onnx");
    let version_path = Path::new(&out_dir).join("bird-head-detector.version");

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
    if let (Some(cached_path), Some(cached_version_path)) = (&cached_model, &cached_version) {
        if cached_path.exists() && cached_version_path.exists() {
            if let Ok(cached_tag) = fs::read_to_string(cached_version_path) {
                if let Ok(latest_tag) = get_latest_release_tag() {
                    if cached_tag.trim() == latest_tag.trim() {
                        println!("Using cached ONNX model (version: {})", cached_tag.trim());
                        fs::copy(cached_path, &model_path)?;
                        fs::copy(cached_version_path, &version_path)?;
                        use_cache = true;
                    } else {
                        println!(
                            "Cache outdated (cached: {}, latest: {}), downloading new model",
                            cached_tag.trim(),
                            latest_tag.trim()
                        );
                    }
                } else {
                    println!("Could not check latest release, using cached model");
                    fs::copy(cached_path, &model_path)?;
                    fs::copy(cached_version_path, &version_path)?;
                    use_cache = true;
                }
            }
        }
    }

    if use_cache {
        return Ok(());
    }

    // Only download if the model doesn't exist or if we're in CI or if forced
    let force_download =
        env::var("CI").is_ok() || env::var("FORCE_DOWNLOAD").is_ok() || !model_path.exists();

    if force_download {
        println!("Downloading latest ONNX model from GitHub releases...");
        let release_tag = download_latest_model(&model_path)?;

        // Write version file
        fs::write(&version_path, &release_tag)?;

        // Cache the model and version for CI if cache directory is set
        if let Some(cached_path) = cached_model {
            if let Some(parent) = cached_path.parent() {
                fs::create_dir_all(parent)?;
                fs::copy(&model_path, &cached_path)?;

                // Save the version information
                if let Some(cached_version_path) = cached_version {
                    fs::copy(&version_path, &cached_version_path)?;
                }

                println!(
                    "Cached model (version: {}) for future builds at: {}",
                    release_tag,
                    cached_path.display()
                );
            }
        }
    } else {
        println!("Using existing ONNX model");

        // If we have an existing model but no version file, try to get version
        if !version_path.exists() {
            if let Ok(latest_tag) = get_latest_release_tag() {
                fs::write(&version_path, &latest_tag)?;
            } else {
                fs::write(&version_path, "unknown")?;
            }
        }
    }

    // Tell cargo to rebuild if this script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to rebuild if we're in CI (to check for cache updates)
    if env::var("CI").is_ok() {
        println!("cargo:rerun-if-env-changed=CI");
        println!("cargo:rerun-if-env-changed=ONNX_MODEL_CACHE_DIR");
    }

    // Tell cargo to rebuild if FORCE_DOWNLOAD is set
    println!("cargo:rerun-if-env-changed=FORCE_DOWNLOAD");

    Ok(())
}
fn get_latest_release_tag() -> Result<String, Box<dyn std::error::Error>> {
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases";

    let client = ureq::Agent::new();
    let mut request = client
        .get(api_url)
        .set("User-Agent", "beaker-build-script/0.1.0");

    // Use GitHub token if available for higher rate limits
    if let Ok(token) = env::var("GITHUB_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }

    let response = request.call()?;

    let releases: serde_json::Value = response.into_json()?;

    let releases_array = releases.as_array().ok_or("No releases found")?;

    // Find the latest release that matches the bird-head-detector pattern
    for release in releases_array {
        let tag_name = release["tag_name"]
            .as_str()
            .ok_or("No tag_name found in release")?;

        if tag_name.starts_with("bird-head-detector-v") {
            return Ok(tag_name.to_string());
        }
    }

    Err("No bird-head-detector release found".into())
}

fn download_latest_model(output_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // GitHub API to get all releases
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases";

    println!("Fetching releases from: {api_url}");

    let client = ureq::Agent::new();
    let mut request = client
        .get(api_url)
        .set("User-Agent", "beaker-build-script/0.1.0");

    // Use GitHub token if available for higher rate limits
    if let Ok(token) = env::var("GITHUB_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
        println!("Using authenticated GitHub API request");
    } else {
        println!("Using unauthenticated GitHub API request (rate limited)");
    }

    let response = request.call()?;

    let releases: serde_json::Value = response.into_json()?;

    let releases_array = releases.as_array().ok_or("No releases found")?;

    // Find the latest release that matches the bird-head-detector pattern
    for release in releases_array {
        let tag_name = release["tag_name"]
            .as_str()
            .ok_or("No tag_name found in release")?;

        if tag_name.starts_with("bird-head-detector-v") {
            // Find the ONNX model asset in this release
            let assets = release["assets"]
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
                .ok_or("No ONNX model found in bird-head-detector release")?;

            let download_url = onnx_asset["browser_download_url"]
                .as_str()
                .ok_or("No download URL found for ONNX asset")?;

            let model_name = onnx_asset["name"]
                .as_str()
                .unwrap_or("bird-head-detector.onnx");

            println!("Downloading {model_name} (version: {tag_name}) from: {download_url}");

            // Download the model
            let response = client
                .get(download_url)
                .set("User-Agent", "beaker-build-script/0.1.0")
                .call()?;

            let mut reader = response.into_reader();
            let mut file = fs::File::create(output_path)?;
            std::io::copy(&mut reader, &mut file)?;

            println!(
                "Successfully downloaded model to: {}",
                output_path.display()
            );

            return Ok(tag_name.to_string());
        }
    }

    Err("No bird-head-detector release found".into())
}
