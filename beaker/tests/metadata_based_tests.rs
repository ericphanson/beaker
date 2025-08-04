// Import the test framework
mod metadata_test_framework;
use metadata_test_framework::*;

// Import performance tracking
mod test_performance_tracker;

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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.25)),
                MetadataCheck::ConfigValue("head", "device", toml::Value::from("cpu")),
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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.5)),
                MetadataCheck::ConfigValue("head", "device", toml::Value::from("auto")),
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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.5)),
                MetadataCheck::ConfigValue("head", "crop", toml::Value::from(true)),
                MetadataCheck::ConfigValue("head", "bounding_box", toml::Value::from(true)),
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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.9)),
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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.3)),
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
                MetadataCheck::ConfigValue("head", "confidence", toml::Value::from(0.25)),
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
                MetadataCheck::ConfigValue("cutout", "alpha_matting", toml::Value::from(false)),
                MetadataCheck::ConfigValue("cutout", "save_mask", toml::Value::from(false)),
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
                MetadataCheck::ConfigValue("cutout", "alpha_matting", toml::Value::from(true)),
                MetadataCheck::ConfigValue("cutout", "save_mask", toml::Value::from(true)),
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
                MetadataCheck::ConfigValue("head", "crop", toml::Value::from(true)),
                MetadataCheck::ConfigValue("cutout", "save_mask", toml::Value::from(true)),
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

// Generate the actual tests using the framework macro
generate_metadata_tests! {
    get_test_scenarios,
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
