use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// Import the metadata structures
use beaker::shared_metadata::BeakerMetadata;

/// Test performance tracking
#[derive(Debug)]
struct TestPerformanceTracker {
    head_invocations: u32,
    cutout_invocations: u32,
    total_test_time: Duration,
    slowest_tests: Vec<(String, Duration)>,
}

impl TestPerformanceTracker {
    fn new() -> Self {
        Self {
            head_invocations: 0,
            cutout_invocations: 0,
            total_test_time: Duration::ZERO,
            slowest_tests: Vec::new(),
        }
    }

    fn record_test(&mut self, test_name: &str, tool: &str, duration: Duration) {
        // Track per-tool invocations
        match tool {
            "head" => self.head_invocations += 1,
            "cutout" => self.cutout_invocations += 1,
            "both" => {
                self.head_invocations += 1;
                self.cutout_invocations += 1;
            }
            _ => {}
        }

        // Add to total time
        self.total_test_time += duration;

        // Track slowest tests (keep top 5)
        self.slowest_tests.push((test_name.to_string(), duration));
        self.slowest_tests.sort_by(|a, b| b.1.cmp(&a.1));
        if self.slowest_tests.len() > 5 {
            self.slowest_tests.truncate(5);
        }
    }

    fn print_summary(&self) {
        println!("\n📊 Test Performance Summary:");
        println!(
            "  Total test time: {:.2}s",
            self.total_test_time.as_secs_f64()
        );
        println!("  Head model invocations: {}", self.head_invocations);
        println!("  Cutout model invocations: {}", self.cutout_invocations);

        if !self.slowest_tests.is_empty() {
            println!("  Slowest tests:");
            for (name, duration) in &self.slowest_tests {
                println!("    - {}: {:.2}s", name, duration.as_secs_f64());
            }
        }

        // Performance warnings
        if self.total_test_time.as_secs() > 60 {
            println!("  ⚠️  Total test time exceeded 60s target");
        }
        if self.cutout_invocations > 10 {
            println!(
                "  ⚠️  High cutout model usage ({} invocations) - consider optimization",
                self.cutout_invocations
            );
        }
    }
}

// Global performance tracker
static PERFORMANCE_TRACKER: Mutex<TestPerformanceTracker> = Mutex::new(TestPerformanceTracker {
    head_invocations: 0,
    cutout_invocations: 0,
    total_test_time: Duration::ZERO,
    slowest_tests: Vec::new(),
});

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
                assert_eq!(actual_value, expected_value,
                    "Config {field_path} should be {expected_value:?} for {tool} in test {test_name}, got {actual_value:?}");
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

/// Record test performance metrics
fn record_test_performance(test_name: &str, _tool: &str, duration: Duration) {
    // Warn if any single test takes > 5 seconds
    if duration.as_secs() > 5 {
        eprintln!(
            "⚠️  Slow test: {} took {:.2}s",
            test_name,
            duration.as_secs_f64()
        );
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
    let metadata_path = temp_dir.path().join("example.beaker.toml");
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
            expected_files: vec!["example.beaker.toml", "example_cutout.png"],
            metadata_checks: vec![
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::ConfigValue("cutout", "alpha_matting", json!(false)),
                MetadataCheck::ConfigValue("cutout", "save_mask", json!(false)),
                MetadataCheck::OutputCreated("example_cutout.png"),
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
            expected_files: vec![
                "example.beaker.toml",
                "example_cutout.png",
                "example_mask.png",
            ],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("cutout", "alpha_matting", json!(true)),
                MetadataCheck::ConfigValue("cutout", "save_mask", json!(true)),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::OutputCreated("example_cutout.png"),
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

/// Main test runner that executes all scenarios
#[test]
fn test_comprehensive_metadata_validation() {
    let scenarios = get_test_scenarios();
    let start_time = Instant::now();

    println!(
        "🧪 Running {} metadata-based test scenarios...",
        scenarios.len()
    );

    for scenario in scenarios {
        println!("  Running: {}", scenario.name);
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        run_and_validate_scenario(scenario, &temp_dir);
    }

    let total_duration = start_time.elapsed();

    // Report basic performance metrics
    println!("\n📊 Test Performance Summary:");
    println!("  Total test time: {:.2}s", total_duration.as_secs_f64());

    // Fail if total test suite exceeds 60 seconds (allowing some buffer)
    assert!(
        total_duration.as_secs() < 60,
        "Test suite took {:.2}s, should be under 60s for fast feedback",
        total_duration.as_secs_f64()
    );

    println!("✅ All metadata validation tests passed!");
}

/// Individual test functions for better isolation and parallel execution
#[test]
fn test_head_detection_cpu_device() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let scenarios = get_test_scenarios();
    let scenario = scenarios
        .into_iter()
        .find(|s| s.name == "head_detection_cpu_single_image")
        .expect("Should find CPU test scenario");
    run_and_validate_scenario(scenario, &temp_dir);
}

#[test]
fn test_head_detection_auto_device() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let scenarios = get_test_scenarios();
    let scenario = scenarios
        .into_iter()
        .find(|s| s.name == "head_detection_auto_device")
        .expect("Should find auto device test scenario");
    run_and_validate_scenario(scenario, &temp_dir);
}

#[test]
fn test_head_detection_with_output_files() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let scenarios = get_test_scenarios();
    let scenario = scenarios
        .into_iter()
        .find(|s| s.name == "head_detection_with_crops_and_bbox")
        .expect("Should find crops and bbox test scenario");
    run_and_validate_scenario(scenario, &temp_dir);
}

#[test]
fn test_cutout_basic() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let scenarios = get_test_scenarios();
    let scenario = scenarios
        .into_iter()
        .find(|s| s.name == "cutout_basic_processing")
        .expect("Should find basic cutout test scenario");
    run_and_validate_scenario(scenario, &temp_dir);
}

#[test]
fn test_multi_tool_workflow() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let scenarios = get_test_scenarios();
    let scenario = scenarios
        .into_iter()
        .find(|s| s.name == "multi_tool_sequential_processing")
        .expect("Should find multi-tool test scenario");
    run_and_validate_scenario(scenario, &temp_dir);
}
