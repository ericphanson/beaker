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
                // Try to get latest tag, but don't fail if network is unavailable
                match get_latest_release_tag() {
                    Ok(latest_tag) => {
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
                    }
                    Err(_) => {
                        println!("Could not check latest release (likely network/TLS issue), using cached model");
                        fs::copy(cached_path, &model_path)?;
                        fs::copy(cached_version_path, &version_path)?;
                        use_cache = true;
                    }
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
        
        // Try to download, but handle network failures gracefully
        match download_latest_model(&model_path) {
            Ok(release_tag) => {
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
            }
            Err(e) => {
                eprintln!("Failed to download model: {e}");
                
                // In CI environments, check if we have a cache directory setup
                // If so, we can proceed with a placeholder for now
                let has_cache_setup = env::var("ONNX_MODEL_CACHE_DIR").is_ok();
                
                if env::var("CI").is_ok() && !model_path.exists() && !has_cache_setup {
                    eprintln!("ERROR: In CI environment, model download failed and no existing model found.");
                    eprintln!("This is likely due to network restrictions or firewall issues.");
                    eprintln!("Please ensure the following URLs are accessible:");
                    eprintln!("  - api.github.com");
                    eprintln!("  - github.com");
                    eprintln!("  - objects.githubusercontent.com");
                    return Err(e);
                }
                
                // For CI with cache setup or local development, create a placeholder model file if none exists
                if !model_path.exists() {
                    if env::var("CI").is_ok() {
                        println!("CI environment detected with cache setup, creating placeholder model...");
                        println!("NOTE: The actual model should be cached from a previous successful build.");
                    } else {
                        println!("Creating placeholder model file for local development...");
                    }
                    
                    fs::write(&model_path, b"placeholder")?;
                    fs::write(&version_path, "unknown-offline")?;
                    
                    if env::var("CI").is_ok() {
                        println!("WARNING: Using placeholder model in CI. Tests may fail without real model.");
                    } else {
                        println!("WARNING: Using placeholder model file. The application may not work correctly.");
                        println!("To get a real model, ensure network access and run with FORCE_DOWNLOAD=1");
                    }
                }
            }
        }
    } else {
        println!("Using existing ONNX model");

        // If we have an existing model but no version file, try to get version
        if !version_path.exists() {
            match get_latest_release_tag() {
                Ok(latest_tag) => {
                    fs::write(&version_path, &latest_tag)?;
                }
                Err(_) => {
                    fs::write(&version_path, "unknown-offline")?;
                }
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

    // Try with default TLS configuration first
    let client = ureq::Agent::new();
    let result = client
        .get(api_url)
        .set("User-Agent", "beaker-build-script/0.1.0")
        .call();

    let response = match result {
        Ok(resp) => resp,
        Err(ureq::Error::Transport(transport_err)) => {
            eprintln!("Warning: GitHub API access failed due to network/TLS issues: {transport_err}");
            eprintln!("This is commonly caused by firewall restrictions or certificate validation issues.");
            eprintln!("Falling back to default version handling...");
            return Err("Network access unavailable".into());
        }
        Err(e) => return Err(e.into()),
    };

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
    let result = client
        .get(api_url)
        .set("User-Agent", "beaker-build-script/0.1.0")
        .call();

    let response = match result {
        Ok(resp) => resp,
        Err(ureq::Error::Transport(transport_err)) => {
            eprintln!("Error: GitHub API access failed due to network/TLS issues: {transport_err}");
            eprintln!("This is commonly caused by firewall restrictions or certificate validation issues.");
            eprintln!("Please check network connectivity and firewall settings.");
            eprintln!("Required URLs to allowlist:");
            eprintln!("  - api.github.com");
            eprintln!("  - github.com");
            eprintln!("  - objects.githubusercontent.com");
            return Err(format!("Network access failed: {transport_err}").into());
        }
        Err(e) => return Err(e.into()),
    };

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

            // Download the model with better error handling
            let response = match client
                .get(download_url)
                .set("User-Agent", "beaker-build-script/0.1.0")
                .call()
            {
                Ok(resp) => resp,
                Err(ureq::Error::Transport(transport_err)) => {
                    eprintln!("Error: Model download failed due to network/TLS issues: {transport_err}");
                    return Err(format!("Model download failed: {transport_err}").into());
                }
                Err(e) => return Err(e.into()),
            };

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
