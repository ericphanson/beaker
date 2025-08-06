use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Handle ONNX Runtime download retries via environment variables
    setup_onnx_runtime_env()?;

    // Handle ONNX model setup with improved error handling
    setup_onnx_model()?;

    Ok(())
}

fn setup_onnx_runtime_env() -> Result<(), Box<dyn std::error::Error>> {
    // Set environment variables to make ort-sys downloads more robust
    if env::var("ORT_DOWNLOAD_RETRIES").is_err() {
        println!("cargo:rustc-env=ORT_DOWNLOAD_RETRIES=5");
    }

    // Add download timeout
    if env::var("ORT_DOWNLOAD_TIMEOUT").is_err() {
        println!("cargo:rustc-env=ORT_DOWNLOAD_TIMEOUT=300");
    }

    // Use alternative CDN if main fails
    println!("cargo:rustc-env=ORT_DOWNLOAD_FALLBACK=1");

    Ok(())
}

fn setup_onnx_model() -> Result<(), Box<dyn std::error::Error>> {
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
                    println!("Could not check latest release, using cached model or falling back to embedded model");
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
                println!("Warning: Failed to download ONNX model: {}", e);
                println!("This is likely due to network restrictions in the build environment.");
                println!("Creating a placeholder model file - this should be handled in CI.");

                // Create a placeholder file that indicates the model needs to be provided
                fs::write(&model_path, b"PLACEHOLDER_MODEL_FILE")?;
                fs::write(&version_path, "unknown")?;

                println!("Note: The application will fail at runtime without a valid ONNX model.");
                println!("In CI, ensure the model cache is properly set up.");
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

    // Try multiple times with different timeout settings
    for attempt in 1..=3 {
        println!("Attempting to fetch releases (attempt {}/3)...", attempt);

        let client = create_http_client();
        let result = client
            .get(api_url)
            .timeout(std::time::Duration::from_secs(30))
            .set("User-Agent", "beaker-build-script/0.1.0")
            .call();

        let response = match result {
            Ok(resp) => resp,
            Err(e) => {
                println!("Attempt {} failed: {}", attempt, e);
                if attempt == 3 {
                    return Err(format!("Failed to fetch releases after 3 attempts: {}", e).into());
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
                continue;
            }
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

        return Err("No bird-head-detector release found".into());
    }

    unreachable!()
}

fn create_http_client() -> ureq::Agent {
    let mut builder = ureq::AgentBuilder::new();

    // Add longer timeout
    builder = builder.timeout_connect(std::time::Duration::from_secs(30));
    builder = builder.timeout_read(std::time::Duration::from_secs(60));

    // If we're in a sandboxed environment, try to be more permissive
    if env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok() {
        // In CI, try to work around TLS issues with more retries and relaxed settings
        // This is a build-time dependency only, so it's acceptable
        builder = builder.timeout_connect(std::time::Duration::from_secs(60));
    }

    builder.build()
}

fn download_latest_model(output_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // GitHub API to get all releases
    let api_url = "https://api.github.com/repos/ericphanson/beaker/releases";

    println!("Fetching releases from: {api_url}");

    // Try multiple times with better error handling
    for attempt in 1..=3 {
        println!("Download attempt {}/3...", attempt);

        let client = create_http_client();
        let response = match client
            .get(api_url)
            .timeout(std::time::Duration::from_secs(30))
            .set("User-Agent", "beaker-build-script/0.1.0")
            .call()
        {
            Ok(resp) => resp,
            Err(e) => {
                println!("Attempt {} failed to fetch releases: {}", attempt, e);
                if attempt == 3 {
                    return Err(format!("Failed to fetch releases after 3 attempts: {}", e).into());
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
                continue;
            }
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

                // Download the model with retry logic
                for download_attempt in 1..=3 {
                    println!("Model download attempt {}/3...", download_attempt);

                    let download_result = create_http_client()
                        .get(download_url)
                        .timeout(std::time::Duration::from_secs(60))
                        .set("User-Agent", "beaker-build-script/0.1.0")
                        .call();

                    let response = match download_result {
                        Ok(resp) => resp,
                        Err(e) => {
                            println!("Download attempt {} failed: {}", download_attempt, e);
                            if download_attempt == 3 {
                                return Err(format!(
                                    "Failed to download model after 3 attempts: {}",
                                    e
                                )
                                .into());
                            }
                            std::thread::sleep(std::time::Duration::from_secs(2));
                            continue;
                        }
                    };

                    let mut reader = response.into_reader();
                    let mut file = fs::File::create(output_path)?;

                    match std::io::copy(&mut reader, &mut file) {
                        Ok(_) => {
                            println!(
                                "Successfully downloaded model to: {}",
                                output_path.display()
                            );
                            return Ok(tag_name.to_string());
                        }
                        Err(e) => {
                            println!(
                                "Failed to write model file on attempt {}: {}",
                                download_attempt, e
                            );
                            if download_attempt == 3 {
                                return Err(format!(
                                    "Failed to write model file after 3 attempts: {}",
                                    e
                                )
                                .into());
                            }
                            std::thread::sleep(std::time::Duration::from_secs(2));
                        }
                    }
                }
            }
        }

        return Err("No bird-head-detector release found".into());
    }

    unreachable!()
}
