use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Returns (global_paq2piq_score, global_blur_score, global_quality_score)
fn quality_scores(image: &Path, output_dir: &Path) -> (f64, f64, f64) {
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
    let score = |key: &str| metadata["quality"][key].as_float().unwrap();
    (
        score("global_paq2piq_score"),
        score("global_blur_score"),
        score("global_quality_score"),
    )
}

/// Blurring an image must lower the PaQ-2-PiQ score and raise the blur score
#[test]
fn test_quality_scores_respond_to_blur() {
    let temp_dir = TempDir::new().unwrap();

    let sharp = temp_dir.path().join("sharp.jpg");
    std::fs::copy("../example.jpg", &sharp).unwrap();
    let blurred = temp_dir.path().join("blurred.jpg");
    image::imageops::blur(&image::open(&sharp).unwrap().to_rgb8(), 8.0)
        .save(&blurred)
        .unwrap();

    let (sharp_paq, sharp_blur, sharp_quality) = quality_scores(&sharp, temp_dir.path());
    let (blurred_paq, blurred_blur, blurred_quality) = quality_scores(&blurred, temp_dir.path());
    assert!(
        sharp_paq - blurred_paq > 5.0,
        "Expected blur to lower the PaQ-2-PiQ score by more than 5: sharp {sharp_paq}, blurred {blurred_paq}"
    );
    assert!(
        (0.0..50.0).contains(&sharp_blur) && blurred_blur > 80.0 && blurred_blur <= 100.0,
        "Expected a low blur score for the sharp image and a high one for the blurred copy: sharp {sharp_blur}, blurred {blurred_blur}"
    );

    // Quality score = PaQ-2-PiQ weighted down by the blur score (default alpha 0.7)
    for (paq, blur, quality) in [
        (sharp_paq, sharp_blur, sharp_quality),
        (blurred_paq, blurred_blur, blurred_quality),
    ] {
        let expected = paq * (1.0 - 0.7 * blur / 100.0);
        assert!(
            (quality - expected).abs() < 1e-3,
            "Expected quality score {expected}, got {quality}"
        );
    }
}
