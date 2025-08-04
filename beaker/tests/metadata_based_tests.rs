use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tempfile::TempDir;

// Import the metadata structures
use beaker::shared_metadata::BeakerMetadata;

// Import performance tracking
mod test_performance_tracker;
use std::sync::Once;
use test_performance_tracker::{initialize_performance_tracker, record_test_performance};

static INIT: Once = Once::new();

/// Initialize performance tracker with test scenarios
fn ensure_performance_tracker_initialized() {
    INIT.call_once(|| {
        let scenarios = get_test_scenarios();
        let test_names: Vec<String> = scenarios.iter().map(|s| s.name.to_string()).collect();
        initialize_performance_tracker(test_names);
    });
}

/// Test scenario definition for systematic testing
#[derive(Debug)]
struct TestScenario {
    name: &'static str,
    tool: &'static str, // "head", "cutout", or "both"
    args: Vec<&'static str>,
    expected_files: Vec<&'static str>,
    metadata_checks: Vec<MetadataCheck>,
}

/// Metadata validation checks
#[derive(Debug)]
enum MetadataCheck {
    /// Verify device was used for a specific tool
    DeviceUsed(&'static str, &'static str), // tool, device
    /// Verify number of files processed
    FilesProcessed(&'static str, usize), // tool, count
    /// Verify configuration value
    ConfigValue(&'static str, &'static str, Value), // tool, field_path, expected_value
    /// Verify timing is within bounds
    TimingBound(&'static str, &'static str, f64, f64), // tool, field, min_ms, max_ms
    /// Verify output file was created
    OutputCreated(&'static str), // filename
    /// Verify execution provider
    ExecutionProvider(&'static str, &'static str), // tool, provider
    /// Verify exit code
    ExitCode(&'static str, i32), // tool, expected_code
    /// Verify beaker version is present
    BeakerVersion(&'static str), // tool
    /// Verify core results field exists
    CoreResultsField(&'static str, &'static str), // tool, field_name
}

/// Copy test files to temp directory and return their paths
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

/// Run a beaker command and return exit code
fn run_beaker_command(args: &[&str]) -> i32 {
    let mut full_args = vec!["run", "--"];
    full_args.extend_from_slice(args);

    let output = Command::new("cargo")
        .args(&full_args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    output.status.code().unwrap_or(-1)
}

/// Parse metadata from TOML file
fn parse_metadata(path: &Path) -> BeakerMetadata {
    assert!(
        path.exists(),
        "Metadata file should exist: {}",
        path.display()
    );
    let content = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read metadata file: {}", path.display()));
    toml::from_str(&content)
        .unwrap_or_else(|_| panic!("Failed to parse metadata TOML: {}", path.display()))
}

/// Get nested value from JSON using dot notation (e.g., "config.confidence")
fn get_nested_value<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = value;

    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }

    Some(current)
}

/// Validate a metadata check against parsed metadata
fn validate_metadata_check(metadata: &BeakerMetadata, check: &MetadataCheck, test_name: &str) {
    match check {
        MetadataCheck::DeviceUsed(tool, expected_device) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );
            let device = system.unwrap().device_selected.as_ref().unwrap_or_else(|| {
                panic!("Device selected should be set for {tool} in test {test_name}")
            });
            assert_eq!(
                device, expected_device,
                "Device should be {expected_device} for {tool} in test {test_name}, got {device}"
            );
        }

        MetadataCheck::FilesProcessed(tool, expected_count) => {
            let input = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.input.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.input.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                input.is_some(),
                "Input info should exist for {tool} in test {test_name}"
            );
            let successful = input.unwrap().successful_files.unwrap_or_else(|| {
                panic!("Successful files should be set for {tool} in test {test_name}")
            });
            assert_eq!(successful, *expected_count,
                "Should have processed {expected_count} files for {tool} in test {test_name}, got {successful}");
        }

        MetadataCheck::ConfigValue(tool, field_path, expected_value) => {
            let config = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.config.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.config.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                config.is_some(),
                "Config should exist for {tool} in test {test_name}"
            );

            // Handle special case for device which is stored in system section
            if *field_path == "device" {
                let system = match *tool {
                    "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                    "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                    _ => panic!("Unknown tool: {tool}"),
                };
                assert!(
                    system.is_some(),
                    "System should exist for device check in {tool} test {test_name}"
                );
                let device_requested =
                    system
                        .unwrap()
                        .device_requested
                        .as_ref()
                        .unwrap_or_else(|| {
                            panic!("Device requested should exist for {tool} in test {test_name}")
                        });
                assert_eq!(&json!(device_requested), expected_value,
                    "Device should be {expected_value:?} for {tool} in test {test_name}, got {device_requested}");
            } else {
                let actual_value =
                    get_nested_value(config.unwrap(), field_path).unwrap_or_else(|| {
                        panic!(
                            "Config field {field_path} should exist for {tool} in test {test_name}"
                        )
                    });

                // Handle floating point comparison tolerance for numbers
                match (actual_value, expected_value) {
                    (Value::Number(actual), Value::Number(expected)) => {
                        let actual_f64 = actual.as_f64().unwrap_or(0.0);
                        let expected_f64 = expected.as_f64().unwrap_or(0.0);
                        let tolerance = 1e-6;
                        assert!((actual_f64 - expected_f64).abs() < tolerance,
                            "Config {field_path} should be approximately {expected_f64} for {tool} in test {test_name}, got {actual_f64} (diff: {})",
                            (actual_f64 - expected_f64).abs());
                    }
                    _ => {
                        assert_eq!(actual_value, expected_value,
                            "Config {field_path} should be {expected_value:?} for {tool} in test {test_name}, got {actual_value:?}");
                    }
                }
            }
        }

        MetadataCheck::TimingBound(tool, field, min_ms, max_ms) => {
            let timing_value = match *tool {
                "head" => {
                    let head_sections = metadata.head.as_ref().unwrap_or_else(|| {
                        panic!("Head sections should exist for test {test_name}")
                    });
                    match *field {
                        "execution.total_processing_time_ms" => head_sections
                            .execution
                            .as_ref()
                            .and_then(|e| e.total_processing_time_ms),
                        "system.model_load_time_ms" => head_sections
                            .system
                            .as_ref()
                            .and_then(|s| s.model_load_time_ms),
                        _ => panic!("Unknown timing field: {field}"),
                    }
                }
                "cutout" => {
                    let cutout_sections = metadata.cutout.as_ref().unwrap_or_else(|| {
                        panic!("Cutout sections should exist for test {test_name}")
                    });
                    match *field {
                        "execution.total_processing_time_ms" => cutout_sections
                            .execution
                            .as_ref()
                            .and_then(|e| e.total_processing_time_ms),
                        "system.model_load_time_ms" => cutout_sections
                            .system
                            .as_ref()
                            .and_then(|s| s.model_load_time_ms),
                        _ => panic!("Unknown timing field: {field}"),
                    }
                }
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                timing_value.is_some(),
                "Timing field {field} should exist for {tool} in test {test_name}"
            );
            let actual_time = timing_value.unwrap();
            assert!(actual_time >= *min_ms && actual_time <= *max_ms,
                "Timing {field} should be between {min_ms}ms and {max_ms}ms for {tool} in test {test_name}, got {actual_time}ms");
        }

        MetadataCheck::OutputCreated(_filename) => {
            // This should be checked at the file system level by the test runner
            // We'll implement this in run_and_validate_scenario
        }

        MetadataCheck::ExecutionProvider(tool, expected_provider) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );
            let provider = system
                .unwrap()
                .execution_provider_used
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("Execution provider should be set for {tool} in test {test_name}")
                });
            assert_eq!(provider, expected_provider,
                "Execution provider should be {expected_provider} for {tool} in test {test_name}, got {provider}");
        }

        MetadataCheck::ExitCode(tool, expected_code) => {
            let execution = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.execution.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.execution.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                execution.is_some(),
                "Execution info should exist for {tool} in test {test_name}"
            );
            let exit_code = execution.unwrap().exit_code.unwrap_or_else(|| {
                panic!("Exit code should be set for {tool} in test {test_name}")
            });
            assert_eq!(exit_code, *expected_code,
                "Exit code should be {expected_code} for {tool} in test {test_name}, got {exit_code}");
        }

        MetadataCheck::BeakerVersion(tool) => {
            let execution = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.execution.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.execution.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                execution.is_some(),
                "Execution info should exist for {tool} in test {test_name}"
            );
            let version = execution
                .unwrap()
                .beaker_version
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("Beaker version should be set for {tool} in test {test_name}")
                });
            assert!(
                !version.is_empty(),
                "Beaker version should not be empty for {tool} in test {test_name}"
            );
        }

        MetadataCheck::CoreResultsField(tool, field_name) => {
            // Convert metadata to JSON for easier nested access
            let metadata_json = serde_json::to_value(metadata)
                .expect("Should be able to serialize metadata to JSON");

            let tool_section = metadata_json.get(tool).unwrap_or_else(|| {
                panic!("Tool section {tool} should exist in JSON for test {test_name}")
            });

            let field_value = get_nested_value(tool_section, field_name);
            assert!(
                field_value.is_some(),
                "Core field {field_name} should exist for {tool} in test {test_name}"
            );
        }
    }
}

/// Run and validate a test scenario
fn run_and_validate_scenario(scenario: TestScenario, temp_dir: &TempDir) {
    let start_time = Instant::now();

    // Setup test files in temp directory
    let (example_jpg, example_2_birds) = setup_test_files(temp_dir);

    // Handle special multi-tool case
    let exit_code = if scenario.tool == "both" {
        // Run head first
        let head_exit = run_beaker_command(&[
            "head",
            example_jpg.to_str().unwrap(),
            "--crop",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);

        // Then cutout
        let cutout_exit = run_beaker_command(&[
            "cutout",
            example_jpg.to_str().unwrap(),
            "--save-mask",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);

        // Both should succeed
        assert_eq!(
            head_exit, 0,
            "Head command should succeed in multi-tool test {}",
            scenario.name
        );
        assert_eq!(
            cutout_exit, 0,
            "Cutout command should succeed in multi-tool test {}",
            scenario.name
        );
        0
    } else {
        // Single tool execution - replace file placeholders with actual paths
        let mut full_args = vec![scenario.tool];
        for arg in &scenario.args {
            if *arg == "../example.jpg" {
                full_args.push(example_jpg.to_str().unwrap());
            } else if *arg == "../example-2-birds.jpg" {
                full_args.push(example_2_birds.to_str().unwrap());
            } else {
                full_args.push(arg);
            }
        }
        full_args.extend_from_slice(&["--output-dir", temp_dir.path().to_str().unwrap()]);
        run_beaker_command(&full_args)
    };

    let test_duration = start_time.elapsed();

    // Track performance metrics
    record_test_performance(scenario.name, scenario.tool, test_duration);

    // Verify command succeeded
    assert_eq!(
        exit_code, 0,
        "Command should succeed for test: {}",
        scenario.name
    );

    // Check expected files exist
    for expected_file in &scenario.expected_files {
        let file_path = temp_dir.path().join(expected_file);
        assert!(
            file_path.exists(),
            "Expected file {} should exist for test {}",
            expected_file,
            scenario.name
        );

        // For output files, check they have content
        if !expected_file.ends_with(".toml") {
            let metadata = fs::metadata(&file_path).expect("Failed to get file metadata");
            assert!(
                metadata.len() > 0,
                "Output file {} should not be empty for test {}",
                expected_file,
                scenario.name
            );
        }
    }

    // Parse and validate metadata
    let metadata_path = if scenario
        .expected_files
        .contains(&"example-2-birds.beaker.toml")
    {
        temp_dir.path().join("example-2-birds.beaker.toml")
    } else {
        temp_dir.path().join("example.beaker.toml")
    };

    assert!(
        metadata_path.exists(),
        "Metadata file should exist for test: {}",
        scenario.name
    );

    let metadata = parse_metadata(&metadata_path);

    // Validate all metadata checks
    for check in &scenario.metadata_checks {
        // Handle OutputCreated check at file system level
        if let MetadataCheck::OutputCreated(filename) = check {
            let output_path = temp_dir.path().join(filename);
            assert!(
                output_path.exists(),
                "Output file {} should exist for test {}",
                filename,
                scenario.name
            );
        } else {
            validate_metadata_check(&metadata, check, scenario.name);
        }
    }
}

/// Define comprehensive test scenarios based on the metadata testing strategy
fn get_test_scenarios() -> Vec<TestScenario> {
    vec![
        // Head Detection Tests - Comprehensive (fast model enables this)
        TestScenario {
            name: "head_detection_cpu_single_image",
            tool: "head",
            args: vec!["../example.jpg", "--device", "cpu", "--confidence", "0.25"],
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::DeviceUsed("head", "cpu"),
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::ConfigValue("head", "confidence", json!(0.25)),
                MetadataCheck::ConfigValue("head", "device", json!("cpu")),
                MetadataCheck::TimingBound(
                    "head",
                    "execution.total_processing_time_ms",
                    10.0,
                    5000.0,
                ),
                MetadataCheck::ExecutionProvider("head", "CPUExecutionProvider"),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::BeakerVersion("head"),
                MetadataCheck::CoreResultsField("head", "model_version"),
            ],
        },
        TestScenario {
            name: "head_detection_auto_device",
            tool: "head",
            args: vec!["../example.jpg", "--device", "auto", "--confidence", "0.5"],
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::ConfigValue("head", "confidence", json!(0.5)),
                MetadataCheck::ConfigValue("head", "device", json!("auto")),
                MetadataCheck::TimingBound(
                    "head",
                    "execution.total_processing_time_ms",
                    10.0,
                    5000.0,
                ),
                MetadataCheck::TimingBound("head", "system.model_load_time_ms", 1.0, 1000.0),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::CoreResultsField("head", "model_version"),
            ],
        },
        TestScenario {
            name: "head_detection_with_crops_and_bbox",
            tool: "head",
            args: vec![
                "../example.jpg",
                "--confidence",
                "0.5",
                "--crop",
                "--bounding-box",
            ],
            expected_files: vec![
                "example.beaker.toml",
                "example.jpg",
                "example_bounding-box.jpg",
            ],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("head", "confidence", json!(0.5)),
                MetadataCheck::ConfigValue("head", "crop", json!(true)),
                MetadataCheck::ConfigValue("head", "bounding_box", json!(true)),
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::OutputCreated("example.jpg"),
                MetadataCheck::OutputCreated("example_bounding-box.jpg"),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::CoreResultsField("head", "detections"),
            ],
        },
        TestScenario {
            name: "head_detection_high_confidence",
            tool: "head",
            args: vec!["../example.jpg", "--confidence", "0.9"],
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("head", "confidence", json!(0.9)),
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::CoreResultsField("head", "model_version"),
                // Note: May or may not have detections depending on the image
            ],
        },
        TestScenario {
            name: "head_detection_two_birds",
            tool: "head",
            args: vec!["../example-2-birds.jpg", "--confidence", "0.3"],
            expected_files: vec!["example-2-birds.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("head", "confidence", json!(0.3)),
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::CoreResultsField("head", "detections"),
                MetadataCheck::TimingBound(
                    "head",
                    "execution.total_processing_time_ms",
                    10.0,
                    5000.0,
                ),
            ],
        },
        TestScenario {
            name: "head_detection_batch_processing",
            tool: "head",
            args: vec![
                "../example.jpg",
                "../example-2-birds.jpg",
                "--confidence",
                "0.25",
            ],
            expected_files: vec!["example.beaker.toml", "example-2-birds.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("head", "confidence", json!(0.25)),
                MetadataCheck::TimingBound(
                    "head",
                    "execution.total_processing_time_ms",
                    10.0,
                    5000.0,
                ),
                MetadataCheck::ExitCode("head", 0),
            ],
        },
        // Cutout Processing Tests - Selective (slow model requires optimization)
        TestScenario {
            name: "cutout_basic_processing",
            tool: "cutout",
            args: vec!["../example.jpg"],
            expected_files: vec!["example.beaker.toml", "example.png"],
            metadata_checks: vec![
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::ConfigValue("cutout", "alpha_matting", json!(false)),
                MetadataCheck::ConfigValue("cutout", "save_mask", json!(false)),
                MetadataCheck::OutputCreated("example.png"),
                MetadataCheck::TimingBound(
                    "cutout",
                    "execution.total_processing_time_ms",
                    1000.0,
                    15000.0,
                ),
                MetadataCheck::ExitCode("cutout", 0),
                MetadataCheck::CoreResultsField("cutout", "model_version"),
            ],
        },
        TestScenario {
            name: "cutout_with_alpha_matting_and_mask",
            tool: "cutout",
            args: vec!["../example.jpg", "--alpha-matting", "--save-mask"],
            expected_files: vec!["example.beaker.toml", "example.png", "example_mask.png"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("cutout", "alpha_matting", json!(true)),
                MetadataCheck::ConfigValue("cutout", "save_mask", json!(true)),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::OutputCreated("example.png"),
                MetadataCheck::OutputCreated("example_mask.png"),
                MetadataCheck::TimingBound(
                    "cutout",
                    "execution.total_processing_time_ms",
                    1000.0,
                    15000.0,
                ),
                MetadataCheck::ExitCode("cutout", 0),
            ],
        },
        // Multi-Tool Integration Tests - Essential workflows
        TestScenario {
            name: "multi_tool_sequential_processing",
            tool: "both", // Special case handled in run_and_validate_scenario
            args: vec![], // Handled specially
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::ConfigValue("head", "crop", json!(true)),
                MetadataCheck::ConfigValue("cutout", "save_mask", json!(true)),
                MetadataCheck::ExitCode("head", 0),
                MetadataCheck::ExitCode("cutout", 0),
                MetadataCheck::TimingBound(
                    "head",
                    "execution.total_processing_time_ms",
                    10.0,
                    5000.0,
                ),
                MetadataCheck::TimingBound(
                    "cutout",
                    "execution.total_processing_time_ms",
                    1000.0,
                    15000.0,
                ),
            ],
        },
    ]
}

/// Individual test functions for better isolation and parallel execution
macro_rules! generate_metadata_tests {
    ($($scenario_name:expr,)*) => {
        // Validation function that runs before any test
        fn validate_test_scenarios_completeness() {
            static mut VALIDATED: bool = false;
            static VALIDATION_LOCK: std::sync::Once = std::sync::Once::new();

            VALIDATION_LOCK.call_once(|| {
                let expected_scenarios: Vec<&str> = get_test_scenarios()
                    .iter()
                    .map(|s| s.name)
                    .collect();

                let macro_scenarios = vec![$($scenario_name,)*];

                // Check that all expected scenarios are in the macro
                for expected in &expected_scenarios {
                    if !macro_scenarios.contains(expected) {
                        panic!(
                            "Test scenario '{}' is defined in get_test_scenarios() but missing from generate_metadata_tests! macro",
                            expected
                        );
                    }
                }

                // Check that all macro scenarios exist in the expected list
                for macro_scenario in &macro_scenarios {
                    if !expected_scenarios.contains(macro_scenario) {
                        panic!(
                            "Test scenario '{}' is in generate_metadata_tests! macro but missing from get_test_scenarios()",
                            macro_scenario
                        );
                    }
                }

                // Check counts match
                if expected_scenarios.len() != macro_scenarios.len() {
                    panic!(
                        "Mismatch between number of scenarios in get_test_scenarios() ({}) and generate_metadata_tests! macro ({})",
                        expected_scenarios.len(),
                        macro_scenarios.len()
                    );
                }

                eprintln!("✅ All {} test scenarios are correctly included in the macro", expected_scenarios.len());
                unsafe { VALIDATED = true; }
            });
        }

        $(
            paste::paste! {
                #[test]
                fn [<test_ $scenario_name>]() {
                    // Validate scenarios before running any test
                    validate_test_scenarios_completeness();

                    // Initialize performance tracker once
                    ensure_performance_tracker_initialized();

                    let temp_dir = TempDir::new().expect("Failed to create temp directory");
                    let scenarios = get_test_scenarios();
                    let scenario = scenarios
                        .into_iter()
                        .find(|s| s.name == $scenario_name)
                        .unwrap_or_else(|| panic!("Should find test scenario: {}", $scenario_name));

                    // Run test scenario with automatic performance tracking
                    let result = std::panic::catch_unwind(|| {
                        run_and_validate_scenario(scenario, &temp_dir);
                    });

                    if let Err(panic_info) = result {
                        std::panic::resume_unwind(panic_info);
                    }
                }
            }
        )*
    };
}

generate_metadata_tests! {
    "head_detection_cpu_single_image",
    "head_detection_auto_device",
    "head_detection_with_crops_and_bbox",
    "head_detection_high_confidence",
    "head_detection_two_birds",
    "head_detection_batch_processing",
    "cutout_basic_processing",
    "cutout_with_alpha_matting_and_mask",
    "multi_tool_sequential_processing",
}
