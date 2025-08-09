/// Integration test specifically for cache statistics functionality
/// This test verifies that cache statistics appear in metadata based on model access patterns
mod metadata_test_framework;

use metadata_test_framework::*;
use tempfile::TempDir;

#[test]
fn test_cache_stats_integration() {
    // Set up test environment
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let (_example_jpg, _example_2_birds) = setup_test_files(&temp_dir);

    println!("Testing cache statistics with embedded model (detect detection)...");

    // Test 1: Embedded model (detect detection) - should NOT have download cache stats since cache is not used
    // Allow CoreML cache stats to appear naturally based on device setting
    let embedded_model_scenario = TestScenario {
        name: "cache_stats_embedded_model",
        tool: "detect",
        args: vec!["../example.jpg", "--confidence", "0.5"],
        expected_files: vec!["example.beaker.toml"],
        metadata_checks: vec![
            MetadataCheck::OnnxCacheStatsAbsent("detect"), // No download cache stats for embedded models
            MetadataCheck::DownloadCacheHitAbsent("detect"), // No cache hit/miss for embedded models
            MetadataCheck::DownloadTimingAbsent("detect"), // No download timing for embedded models
                                                           // Removed CoreML cache check - allow it to appear naturally based on device
        ],
        env_vars: vec![],
    };

    run_and_validate_scenario(embedded_model_scenario, &temp_dir, |_, _, _| {});

    println!("✅ Embedded model cache stats test passed");

    println!("Testing cache statistics with downloaded model (cutout)...");

    // Test 2: Downloaded model (cutout) - should have download cache stats
    // Allow CoreML cache stats to appear naturally based on device setting
    let downloaded_model_scenario = TestScenario {
        name: "cache_stats_downloaded_model",
        tool: "cutout",
        args: vec!["../example.jpg"],
        expected_files: vec!["example.beaker.toml", "example_cutout.png"],
        metadata_checks: vec![
            MetadataCheck::OnnxCacheStatsPresent("cutout"), // Download cache stats should be present
            MetadataCheck::DownloadCacheHitPresent("cutout"), // Cache hit/miss should be present for downloaded models
                                                              // Removed CoreML cache check - allow it to appear naturally based on device
        ],
        env_vars: vec![],
    };

    run_and_validate_scenario(downloaded_model_scenario, &temp_dir, |_, _, _| {});

    println!("✅ Downloaded model cache stats test passed");

    println!("Testing explicit CPU device with no CoreML cache...");

    // Test 3: Explicit CPU device test - enforce no CoreML cache stats
    let cpu_device_scenario = TestScenario {
        name: "cache_stats_cpu_device",
        tool: "detect",
        args: vec!["../example.jpg", "--confidence", "0.5", "--device", "cpu"],
        expected_files: vec!["example.beaker.toml"],
        metadata_checks: vec![
            MetadataCheck::OnnxCacheStatsAbsent("detect"), // No download cache stats for embedded models
            MetadataCheck::DownloadCacheHitAbsent("detect"), // No cache hit/miss for embedded models
            MetadataCheck::DownloadTimingAbsent("detect"), // No download timing for embedded models
            MetadataCheck::CoremlCacheStatsAbsent("detect"), // No CoreML stats when explicitly using CPU
        ],
        env_vars: vec![],
    };

    run_and_validate_scenario(cpu_device_scenario, &temp_dir, |_, _, _| {});

    println!("✅ CPU device cache stats test passed");

    println!("✅ All cache statistics integration tests passed!");
}
