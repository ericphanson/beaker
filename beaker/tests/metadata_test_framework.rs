use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tempfile::TempDir;
use toml::Value;

// Import the metadata structures
use beaker::shared_metadata::BeakerMetadata;

/// Test scenario definition for systematic testing
#[derive(Debug)]
pub struct TestScenario {
    pub name: &'static str,
    pub tool: &'static str, // "detect", "cutout", or "both"
    pub args: Vec<&'static str>,
    pub expected_files: Vec<&'static str>,
    pub metadata_checks: Vec<MetadataCheck>,
    pub env_vars: Vec<(&'static str, &'static str)>, // Environment variables to set
}

/// Metadata validation checks
#[derive(Debug)]
pub enum MetadataCheck {
    /// Verify device was used for a specific tool
    DeviceUsed(&'static str, &'static str), // tool, device
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
    /// Verify file I/O timing is present and valid
    IoTimingExists(&'static str), // tool
    /// Verify environment variable is present
    EnvVarPresent(&'static str, &'static str), // tool, env_var_name
    /// Verify environment variable has specific value
    EnvVarValue(&'static str, &'static str, &'static str), // tool, env_var_name, expected_value
    /// Verify mask encoding is present (cutout only)
    MaskEncodingPresent,
    /// Verify ASCII preview is present and contains expected characters
    AsciiPreviewValid,

    // Cache Statistics Checks
    /// Verify ONNX cache statistics are present (count and size)
    OnnxCacheStatsPresent(&'static str), // tool
    /// Verify ONNX cache statistics are absent (for models that don't access cache)
    OnnxCacheStatsAbsent(&'static str), // tool
    /// Verify download cache hit/miss field is present
    DownloadCacheHitPresent(&'static str), // tool
    /// Verify download cache hit/miss field is absent (for embedded models)
    DownloadCacheHitAbsent(&'static str), // tool
    /// Verify download timing is present (for downloaded models)
    DownloadTimingPresent(&'static str), // tool
    /// Verify download timing is absent (for embedded/cached models)
    DownloadTimingAbsent(&'static str), // tool
    /// Verify CoreML cache statistics are present (when CoreML device is used)
    CoremlCacheStatsPresent(&'static str), // tool
    /// Verify CoreML cache statistics are absent (when CoreML device is not used)
    CoremlCacheStatsAbsent(&'static str), // tool
}

/// Copy test files to temp directory and return their paths
pub fn setup_test_files(temp_dir: &TempDir) -> (PathBuf, PathBuf) {
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
pub fn run_beaker_command(args: &[&str]) -> i32 {
    run_beaker_command_with_env(args, &[])
}

/// Run a beaker command with environment variables and return exit code
pub fn run_beaker_command_with_env(args: &[&str], env_vars: &[(&str, &str)]) -> i32 {
    use std::sync::Once;

    static BUILD_ONCE: Once = Once::new();

    // Build the binary once at the start of testing
    BUILD_ONCE.call_once(|| {
        let build_output = Command::new("cargo")
            .args(["build"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to build beaker");

        if !build_output.status.success() {
            panic!(
                "Failed to build beaker: {}",
                String::from_utf8_lossy(&build_output.stderr)
            );
        }
    });

    // Run the built binary directly
    let beaker_binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/beaker");

    let output = Command::new(&beaker_binary)
        .args(args)
        .envs(env_vars.iter().map(|(k, v)| (*k, *v)))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute beaker command");

    let exit_code = output.status.code().unwrap_or(-1);

    // Print stdout and stderr for debugging when command fails
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

/// Parse metadata from TOML file
pub fn parse_metadata(path: &Path) -> BeakerMetadata {
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

/// Get nested value from TOML using dot notation (e.g., "config.confidence")
pub fn get_nested_value<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = value;

    for part in parts {
        match current {
            Value::Table(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }

    Some(current)
}

/// Validate a metadata check against parsed metadata
pub fn validate_metadata_check(metadata: &BeakerMetadata, check: &MetadataCheck, test_name: &str) {
    match check {
        MetadataCheck::DeviceUsed(tool, expected_device) => {
            let system = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.system.as_ref()),
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

        MetadataCheck::ConfigValue(tool, field_path, expected_value) => {
            let config = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.config.as_ref()),
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
                    "detect" => metadata.detect.as_ref().and_then(|d| d.system.as_ref()),
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
                assert_eq!(device_requested, expected_value.as_str().unwrap(),
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
                    (Value::Float(actual), Value::Float(expected)) => {
                        let tolerance = 1e-6;
                        assert!((actual - expected).abs() < tolerance,
                            "Config {field_path} should be approximately {expected} for {tool} in test {test_name}, got {actual} (diff: {})",
                            (actual - expected).abs());
                    }
                    (Value::Integer(actual), Value::Integer(expected)) => {
                        assert_eq!(actual, expected,
                            "Config {field_path} should be {expected} for {tool} in test {test_name}, got {actual}");
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
                "detect" => {
                    let detect_sections = metadata.detect.as_ref().unwrap_or_else(|| {
                        panic!("Detect sections should exist for test {test_name}")
                    });
                    match *field {
                        "execution.model_processing_time_ms" => detect_sections
                            .execution
                            .as_ref()
                            .and_then(|e| e.model_processing_time_ms),
                        "system.model_load_time_ms" => detect_sections
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
                        "execution.model_processing_time_ms" => cutout_sections
                            .execution
                            .as_ref()
                            .and_then(|e| e.model_processing_time_ms),
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
                "Timing {field} should be between {min_ms} ms and {max_ms} ms for {tool} in test {test_name}, got {actual_time} ms");
        }

        MetadataCheck::OutputCreated(_filename) => {
            // This should be checked at the file system level by the test runner
            // We'll implement this in run_and_validate_scenario
        }

        MetadataCheck::ExecutionProvider(tool, expected_provider) => {
            let system = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );
            let providers = &system.unwrap().execution_providers;

            // Check that expected_provider name is contained in any of the provider strings
            let provider_found = providers
                .iter()
                .any(|provider| provider.contains(expected_provider));

            assert!(provider_found,
        "Execution provider should contain {expected_provider} for {tool} in test {test_name}, got {providers:?}",);
        }

        MetadataCheck::ExitCode(tool, expected_code) => {
            let execution = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.execution.as_ref()),
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
                "detect" => metadata.detect.as_ref().and_then(|d| d.execution.as_ref()),
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
            // Convert metadata to TOML for easier nested access
            let metadata_toml = toml::Value::try_from(metadata)
                .expect("Should be able to serialize metadata to TOML");

            let tool_section = match metadata_toml {
                Value::Table(ref table) => table.get(&tool.to_string()).unwrap_or_else(|| {
                    panic!("Tool section {tool} should exist in TOML for test {test_name}")
                }),
                _ => panic!("Metadata should be a TOML table"),
            };

            let field_value = get_nested_value(tool_section, field_name);
            assert!(
                field_value.is_some(),
                "Core field {field_name} should exist for {tool} in test {test_name}"
            );
        }

        MetadataCheck::IoTimingExists(tool) => {
            let execution = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.execution.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.execution.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                execution.is_some(),
                "Execution context should exist for {tool} in test {test_name}"
            );

            let io_timing = execution.unwrap().file_io.as_ref();
            assert!(
                io_timing.is_some(),
                "File I/O timing should exist for {tool} in test {test_name}"
            );

            let timing = io_timing.unwrap();
            // At least one timing field should be present and non-zero (we do read and write operations)
            let has_read_timing = timing.read_time_ms.map_or(false, |t| t > 0.0);
            let has_write_timing = timing.write_time_ms.map_or(false, |t| t > 0.0);

            assert!(
                has_read_timing || has_write_timing,
                "At least one I/O timing value should be present and positive for {tool} in test {test_name}"
            );
        }

        MetadataCheck::EnvVarPresent(tool, env_var_name) => {
            let execution = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.execution.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.execution.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                execution.is_some(),
                "Execution info should exist for {tool} in test {test_name}"
            );

            let env_vars = execution.unwrap().beaker_env_vars.as_ref();
            assert!(
                env_vars.is_some(),
                "Environment variables should be present for {tool} in test {test_name}"
            );

            let env_map = env_vars.unwrap();
            assert!(
                env_map.contains_key(*env_var_name),
                "Environment variable {env_var_name} should be present for {tool} in test {test_name}"
            );
        }

        MetadataCheck::EnvVarValue(tool, env_var_name, expected_value) => {
            let execution = match *tool {
                "detect" => metadata.detect.as_ref().and_then(|d| d.execution.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.execution.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                execution.is_some(),
                "Execution info should exist for {tool} in test {test_name}"
            );

            let env_vars = execution.unwrap().beaker_env_vars.as_ref();
            assert!(
                env_vars.is_some(),
                "Environment variables should be present for {tool} in test {test_name}"
            );

            let env_map = env_vars.unwrap();
            let actual_value = env_map.get(*env_var_name).unwrap_or_else(|| {
                panic!("Environment variable {env_var_name} should be present for {tool} in test {test_name}")
            });

            assert_eq!(
                actual_value, expected_value,
                "Environment variable {env_var_name} should have value {expected_value} for {tool} in test {test_name}, got {actual_value}"
            );
        }

        MetadataCheck::MaskEncodingPresent => {
            assert!(
                metadata.cutout.is_some(),
                "Cutout metadata should be present for mask encoding check in test {test_name}"
            );

            let cutout = metadata.cutout.as_ref().unwrap();
            assert!(
                cutout.mask.is_some(),
                "Mask data should be present in cutout metadata for test {test_name}"
            );

            let mask = cutout.mask.as_ref().unwrap();
            assert!(
                !mask.data.is_empty(),
                "Mask data should not be empty for test {test_name}"
            );
            assert_eq!(
                mask.format, "rle-binary-v1 | gzip | base64",
                "Mask format should be correct for test {test_name}"
            );
        }

        MetadataCheck::AsciiPreviewValid => {
            assert!(
                metadata.cutout.is_some(),
                "Cutout metadata should be present for ASCII preview check in test {test_name}"
            );

            let cutout = metadata.cutout.as_ref().unwrap();
            assert!(
                cutout.mask.is_some(),
                "Mask data should be present for ASCII preview check in test {test_name}"
            );

            let mask = cutout.mask.as_ref().unwrap();
            assert!(
                mask.preview.is_some(),
                "ASCII preview should be present for test {test_name}"
            );

            let preview = mask.preview.as_ref().unwrap();
            assert_eq!(
                preview.format, "ascii",
                "Preview format should be ascii for test {test_name}"
            );
            assert!(
                preview.width > 0 && preview.height > 0,
                "Preview dimensions should be positive for test {test_name}"
            );
            assert_eq!(
                preview.rows.len(),
                preview.height as usize,
                "Preview rows count should match height for test {test_name}"
            );

            // Check that preview contains expected characters (# and .)
            let all_chars: String = preview.rows.join("");
            assert!(
                all_chars.contains('#') && all_chars.contains('.'),
                "ASCII preview should contain both '#' and '.' characters for test {test_name}"
            );

            // Check that each row has the correct width
            for (i, row) in preview.rows.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    preview.width as usize,
                    "Preview row {i} should have width {} for test {test_name}, got {}",
                    preview.width,
                    row.len()
                );
            }
        }

        // Cache Statistics Checks
        MetadataCheck::OnnxCacheStatsPresent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );

            let system = system.unwrap();
            assert!(
                system.onnx_cache.is_some(),
                "onnx_cache should be present for {tool} in test {test_name}"
            );
            let onnx_cache = system.onnx_cache.as_ref().unwrap();
            assert!(
                onnx_cache.cached_models_count.is_some(),
                "onnx_cache.cached_models_count should be present for {tool} in test {test_name}"
            );
            assert!(
                onnx_cache.cached_models_size_mb.is_some(),
                "onnx_cache.cached_models_size_mb should be present for {tool} in test {test_name}"
            );
        }

        MetadataCheck::OnnxCacheStatsAbsent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            if let Some(system) = system {
                assert!(
                    system.onnx_cache.is_none(),
                    "onnx_cache should be absent for {tool} in test {test_name}"
                );
            }
        }

        MetadataCheck::DownloadCacheHitPresent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );

            let system = system.unwrap();
            assert!(
                system.onnx_cache.is_some(),
                "onnx_cache should be present for {tool} in test {test_name}"
            );
            let onnx_cache = system.onnx_cache.as_ref().unwrap();
            assert!(
                onnx_cache.model_cache_hit.is_some(),
                "onnx_cache.model_cache_hit should be present for {tool} in test {test_name}"
            );
        }

        MetadataCheck::DownloadCacheHitAbsent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            if let Some(system) = system {
                if let Some(onnx_cache) = &system.onnx_cache {
                    assert!(
                        onnx_cache.model_cache_hit.is_none(),
                        "onnx_cache.model_cache_hit should be absent for {tool} in test {test_name}"
                    );
                }
                // If onnx_cache itself is None, that's also considered absent
            }
        }

        MetadataCheck::DownloadTimingPresent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );

            let system = system.unwrap();
            assert!(
                system.onnx_cache.is_some(),
                "onnx_cache should be present for {tool} in test {test_name}"
            );
            let onnx_cache = system.onnx_cache.as_ref().unwrap();
            assert!(
                onnx_cache.download_time_ms.is_some(),
                "onnx_cache.download_time_ms should be present for {tool} in test {test_name}"
            );
        }

        MetadataCheck::DownloadTimingAbsent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            if let Some(system) = system {
                if let Some(onnx_cache) = &system.onnx_cache {
                    assert!(
                        onnx_cache.download_time_ms.is_none(),
                        "onnx_cache.download_time_ms should be absent for {tool} in test {test_name}"
                    );
                }
                // If onnx_cache itself is None, that's also considered absent
            }
        }

        MetadataCheck::CoremlCacheStatsPresent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            assert!(
                system.is_some(),
                "System info should exist for {tool} in test {test_name}"
            );

            let system = system.unwrap();
            assert!(
                system.coreml_cache.is_some(),
                "coreml_cache should be present for {tool} in test {test_name}"
            );
            let coreml_cache = system.coreml_cache.as_ref().unwrap();
            assert!(
                coreml_cache.cache_hit.is_some(),
                "coreml_cache.cache_hit should be present for {tool} in test {test_name}"
            );
            assert!(
                coreml_cache.cache_count.is_some(),
                "coreml_cache.cache_count should be present for {tool} in test {test_name}"
            );
            assert!(
                coreml_cache.cache_size_mb.is_some(),
                "coreml_cache.cache_size_mb should be present for {tool} in test {test_name}"
            );
        }

        MetadataCheck::CoremlCacheStatsAbsent(tool) => {
            let system = match *tool {
                "head" => metadata.head.as_ref().and_then(|h| h.system.as_ref()),
                "cutout" => metadata.cutout.as_ref().and_then(|c| c.system.as_ref()),
                _ => panic!("Unknown tool: {tool}"),
            };

            if let Some(system) = system {
                assert!(
                    system.coreml_cache.is_none(),
                    "coreml_cache should be absent for {tool} in test {test_name}"
                );
            }
        }
    }
}

/// Run and validate a test scenario
pub fn run_and_validate_scenario<F>(scenario: TestScenario, temp_dir: &TempDir, record_perf: F)
where
    F: Fn(&str, &str, std::time::Duration),
{
    let start_time = Instant::now();

    // Setup test files in temp directory
    let (example_jpg, example_2_birds) = setup_test_files(temp_dir);

    // Handle special cases
    let exit_code = if scenario.tool == "both" {
        // Run detect first
        let detect_exit = run_beaker_command(&[
            "detect",
            example_jpg.to_str().unwrap(),
            "--crop=head",
            "--metadata",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);

        // Then cutout
        let cutout_exit = run_beaker_command(&[
            "cutout",
            example_jpg.to_str().unwrap(),
            "--save-mask",
            "--metadata",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);

        // Both should succeed
        assert_eq!(
            detect_exit, 0,
            "Detect command should succeed in multi-tool test {}",
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
        full_args.extend_from_slice(&[
            "--metadata",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ]);
        run_beaker_command_with_env(&full_args, &scenario.env_vars)
    };

    let test_duration = start_time.elapsed();

    // Track performance metrics
    record_perf(scenario.name, scenario.tool, test_duration);

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

    // Parse and validate metadata (skip for version command)
    if scenario.tool != "version" {
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
    } else {
        // For version command, no output files should be created
        // Version commands don't generate metadata files or output images
    }
}

/// Macro to generate metadata tests with validation
#[macro_export]
macro_rules! generate_metadata_tests {
    ($get_scenarios_fn:ident, $($scenario_name:expr,)*) => {
        // Import performance tracking for use in tests
        use $crate::test_performance_tracker::{initialize_performance_tracker, record_test_performance};
        use std::sync::Once;

        static INIT: Once = Once::new();

        /// Initialize performance tracker with test scenarios
        fn ensure_performance_tracker_initialized(test_names: Vec<String>) {
            INIT.call_once(|| {
                initialize_performance_tracker(test_names);
            });
        }

        // Validation function that runs before any test
        fn validate_test_scenarios_completeness() {
            static mut VALIDATED: bool = false;
            static VALIDATION_LOCK: std::sync::Once = std::sync::Once::new();

            VALIDATION_LOCK.call_once(|| {
                let expected_scenarios: Vec<&str> = $get_scenarios_fn()
                    .iter()
                    .map(|s| s.name)
                    .collect();

                let macro_scenarios = vec![$($scenario_name,)*];

                // Check that all expected scenarios are in the macro
                for expected in &expected_scenarios {
                    if !macro_scenarios.contains(expected) {
                        panic!(
                            "Test scenario '{}' is defined in {}() but missing from generate_metadata_tests! macro",
                            expected, stringify!($get_scenarios_fn)
                        );
                    }
                }

                // Check that all macro scenarios exist in the expected list
                for macro_scenario in &macro_scenarios {
                    if !expected_scenarios.contains(macro_scenario) {
                        panic!(
                            "Test scenario '{}' is in generate_metadata_tests! macro but missing from {}()",
                            macro_scenario, stringify!($get_scenarios_fn)
                        );
                    }
                }

                // Check counts match
                if expected_scenarios.len() != macro_scenarios.len() {
                    panic!(
                        "Mismatch between number of scenarios in {}() ({}) and generate_metadata_tests! macro ({})",
                        stringify!($get_scenarios_fn), expected_scenarios.len(), macro_scenarios.len()
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
                    use $crate::metadata_test_framework::*;

                    // Validate scenarios before running any test
                    validate_test_scenarios_completeness();

                    // Initialize performance tracker once
                    let test_names: Vec<String> = $get_scenarios_fn().iter().map(|s| s.name.to_string()).collect();
                    ensure_performance_tracker_initialized(test_names);

                    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp directory");
                    let scenarios = $get_scenarios_fn();
                    let scenario = scenarios
                        .into_iter()
                        .find(|s| s.name == $scenario_name)
                        .unwrap_or_else(|| panic!("Should find test scenario: {}", $scenario_name));

                    // Run test scenario with automatic performance tracking
                    let result = std::panic::catch_unwind(|| {
                        run_and_validate_scenario(scenario, &temp_dir, record_test_performance);
                    });

                    if let Err(panic_info) = result {
                        std::panic::resume_unwind(panic_info);
                    }
                }
            }
        )*
    };
}
