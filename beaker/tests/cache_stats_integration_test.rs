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

    println!("Testing cache statistics with embedded model (head detection)...");

    // Test 1: Embedded model (head detection) - should have general cache stats but no download stats
    let embedded_model_scenario = TestScenario {
        name: "cache_stats_embedded_model",
        tool: "head",
        args: vec!["../example.jpg", "--confidence", "0.5"],
        expected_files: vec!["example.beaker.toml"],
        metadata_checks: vec![
            MetadataCheck::OnnxCacheStatsPresent("head"), // General cache stats should be present
            MetadataCheck::DownloadCacheHitAbsent("head"), // No cache hit/miss for embedded models
            MetadataCheck::DownloadTimingAbsent("head"),  // No download timing for embedded models
            MetadataCheck::CoremlCacheStatsAbsent("head"), // No CoreML stats when using CPU
        ],
        env_vars: vec![],
    };

    run_and_validate_scenario(embedded_model_scenario, &temp_dir, |_, _, _| {});

    println!("✅ Embedded model cache stats test passed");

    println!("Testing cache statistics with downloaded model (cutout)...");

    // Test 2: Downloaded model (cutout) - should have all applicable cache stats
    let downloaded_model_scenario = TestScenario {
        name: "cache_stats_downloaded_model",
        tool: "cutout",
        args: vec!["../example.jpg"],
        expected_files: vec!["example.beaker.toml", "example_cutout.png"],
        metadata_checks: vec![
            MetadataCheck::OnnxCacheStatsPresent("cutout"), // General cache stats should be present
            MetadataCheck::DownloadCacheHitPresent("cutout"), // Cache hit/miss should be present for downloaded models
            MetadataCheck::CoremlCacheStatsAbsent("cutout"),  // No CoreML stats when using CPU
        ],
        env_vars: vec![],
    };

    run_and_validate_scenario(downloaded_model_scenario, &temp_dir, |_, _, _| {});

    println!("✅ Downloaded model cache stats test passed");

    println!("✅ All cache statistics integration tests passed!");
}
