use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};
use tempfile::TempDir;
use toml::Value;

fn setup_test_files(temp_dir: &TempDir) -> (PathBuf, PathBuf) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let example_jpg = repo_root.join("example.jpg");
    let example_2_birds = repo_root.join("example-2-birds.jpg");

    assert!(
        example_jpg.exists(),
        "Test file should exist: {}",
        example_jpg.display()
    );
    assert!(
        example_2_birds.exists(),
        "Test file should exist: {}",
        example_2_birds.display()
    );

    let temp_example = temp_dir.path().join("example.jpg");
    let temp_2_birds = temp_dir.path().join("example-2-birds.jpg");
    fs::copy(&example_jpg, &temp_example).expect("Failed to copy example.jpg");
    fs::copy(&example_2_birds, &temp_2_birds).expect("Failed to copy example-2-birds.jpg");

    (temp_example, temp_2_birds)
}

fn run_beaker_command(args: &[&str]) -> i32 {
    let beaker_binary = env!("CARGO_BIN_EXE_beaker");
    let output = Command::new(beaker_binary)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    let exit_code = output.status.code().unwrap_or(-1);
    if exit_code != 0 {
        eprintln!("=== BEAKER COMMAND FAILED ===");
        eprintln!("Command: beaker {}", args.join(" "));
        eprintln!("Exit code: {exit_code}");
        eprintln!("=== STDOUT ===");
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("=== STDERR ===");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        eprintln!("=== END BEAKER COMMAND OUTPUT ===");
    }
    exit_code
}

fn parse_output_metadata(extra_args: &[&str]) -> Value {
    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let (example_jpg, _) = setup_test_files(&temp_dir);

    let mut args = vec!["detect", example_jpg.to_str().unwrap()];
    args.extend_from_slice(extra_args);
    args.extend_from_slice(&[
        "--metadata",
        "--output-dir",
        temp_dir.path().to_str().unwrap(),
    ]);

    let exit_code = run_beaker_command(&args);
    assert_eq!(exit_code, 0, "beaker detect command should succeed");

    let metadata_path = temp_dir.path().join("example.beaker.toml");
    let metadata_raw =
        fs::read_to_string(&metadata_path).expect("Failed to read generated metadata");
    toml::from_str::<Value>(&metadata_raw).expect("Failed to parse generated metadata TOML")
}

fn detect_detections(metadata: &Value) -> &Vec<Value> {
    metadata
        .get("detect")
        .and_then(|v| v.get("detections"))
        .and_then(Value::as_array)
        .expect("Expected [detect.detections] to be present")
}

fn read_number(table: &toml::value::Table, key: &str) -> f64 {
    table
        .get(key)
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
        .unwrap_or_else(|| panic!("Expected numeric field '{key}'"))
}

#[test]
fn test_refinement_adds_quality_refined_with_crop_scores() {
    let metadata = parse_output_metadata(&["--refine-detection-quality"]);
    let detections = detect_detections(&metadata);
    assert!(!detections.is_empty(), "Expected at least one detection");

    let mut refined_count = 0usize;

    for detection in detections {
        let detection_table = detection
            .as_table()
            .expect("Each detection should be a TOML table");

        assert!(
            detection_table.contains_key("quality"),
            "All detections should retain baseline quality"
        );

        let Some(refined) = detection_table.get("quality_refined") else {
            continue;
        };

        refined_count += 1;

        let class_name = detection_table
            .get("class_name")
            .and_then(Value::as_str)
            .expect("Detection should contain class_name");
        assert!(
            class_name == "bird" || class_name == "head",
            "Only bird/head detections should be refined, got {class_name}"
        );

        let width = read_number(detection_table, "x2") - read_number(detection_table, "x1");
        let height = read_number(detection_table, "y2") - read_number(detection_table, "y1");
        assert!(
            width >= 64.0 && height >= 64.0,
            "Refined detections should satisfy min size 64x64, got {width:.1}x{height:.1}"
        );

        let refined_table = refined
            .as_table()
            .expect("quality_refined should be a TOML table");
        for field in [
            "crop_paq2piq_global",
            "crop_blur_score",
            "crop_final_quality_score",
        ] {
            assert!(
                refined_table
                    .get(field)
                    .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
                    .is_some(),
                "quality_refined should include numeric field '{field}'"
            );
        }
    }

    assert!(
        refined_count > 0,
        "Expected at least one detection with quality_refined when refinement is enabled"
    );
}

#[test]
fn test_refinement_off_keeps_quality_refined_absent() {
    let metadata = parse_output_metadata(&[]);
    let detections = detect_detections(&metadata);
    assert!(!detections.is_empty(), "Expected at least one detection");

    for detection in detections {
        let detection_table = detection
            .as_table()
            .expect("Each detection should be a TOML table");
        assert!(
            detection_table.contains_key("quality"),
            "Baseline quality should remain available"
        );
        assert!(
            !detection_table.contains_key("quality_refined"),
            "quality_refined should be absent unless refinement is explicitly enabled"
        );
    }
}
