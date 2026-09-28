use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn paq2piq_score(image: &Path, output_dir: &Path) -> f64 {
    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "quality",
            image.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--metadata",
            "--device",
            "cpu",
            "--threads",
            "2",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");
    assert!(
        output.status.success(),
        "Quality command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stem = image.file_stem().unwrap().to_str().unwrap();
    let toml_path = output_dir.join(format!("{stem}.beaker.toml"));
    let metadata: toml::Table =
        toml::from_str(&std::fs::read_to_string(&toml_path).unwrap()).unwrap();
    metadata["quality"]["global_paq2piq_score"]
        .as_float()
        .unwrap()
}

/// The global PaQ-2-PiQ score must fall when the image is blurred
#[test]
fn test_paq2piq_score_falls_with_blur() {
    let temp_dir = TempDir::new().unwrap();

    let sharp = temp_dir.path().join("sharp.jpg");
    std::fs::copy("../example.jpg", &sharp).unwrap();
    let blurred = temp_dir.path().join("blurred.jpg");
    image::imageops::blur(&image::open(&sharp).unwrap().to_rgb8(), 8.0)
        .save(&blurred)
        .unwrap();

    let sharp_score = paq2piq_score(&sharp, temp_dir.path());
    let blurred_score = paq2piq_score(&blurred, temp_dir.path());
    assert!(
        sharp_score - blurred_score > 5.0,
        "Expected blur to lower the PaQ-2-PiQ score by more than 5: sharp {sharp_score}, blurred {blurred_score}"
    );
}
