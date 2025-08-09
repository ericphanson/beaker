# Parallel Process Stress Test Plan

## Overview

This document outlines a comprehensive testing framework for validating the robustness of beaker's ONNX and CoreML caches under concurrent access with simulated network failures. The framework will ensure cache controls work correctly when multiple beaker processes execute simultaneously in noisy network environments.

## Background

Beaker uses two primary caching mechanisms:
1. **ONNX Model Cache**: Downloads and caches ONNX models with MD5 verification and lock-file based concurrency protection
2. **CoreML Cache**: On Apple Silicon, ONNX Runtime compiles ONNX models to CoreML format and caches the result

Both caches must handle concurrent access gracefully without corruption, race conditions, or deadlocks.

## Goals

- **Reliability**: Ensure caches are robust to concurrent access patterns
- **Failure Resilience**: Validate graceful handling of network failures, corruption, and timeouts
- **Performance**: Verify cache performance under stress without blocking legitimate operations
- **Fast Tests**: Keep stress tests fast enough for frequent CI execution
- **Deterministic**: No flaky tests - only test conditions that are guaranteed to be true

## Framework Architecture

### 1. Crate Analysis and Selection

#### HTTP Mocking Options Evaluated

**httpmock = "0.7.0"** (RECOMMENDED)
- ✅ Simpler API focused on HTTP mocking
- ✅ No async complexity - works with beaker's blocking model
- ✅ MIT/Apache-2.0 license
- ✅ Good for simulating network failures and responses
- ✅ Thread-safe for concurrent test execution
- ✅ Lighter weight than wiremock

**wiremock = "0.6.4"** (Alternative)
- ⚠️ Adds async complexity with tokio dependency
- ⚠️ More heavyweight for simple HTTP mocking needs
- ✅ More features but potentially overkill

**Custom HTTP Server**
- ❌ Significant development overhead
- ❌ Need to implement failure injection from scratch
- ❌ Testing the test framework instead of the application

#### Stress Testing Utilities

**cargo-stress = "0.2.0"**
- ✅ Specifically designed for catching non-deterministic failures
- ✅ Could complement our framework for test reliability
- ⚠️ Focused on existing test discovery, not custom stress scenarios

#### Process Management

**std::process::Command** (Current approach)
- ✅ Already used in existing test framework
- ✅ Platform independent
- ✅ Full control over environment and arguments
- ✅ No additional dependencies

### 2. Mock HTTP Server with Failure Injection

**Selected Crate**: `httpmock = "0.7.0"`

**Failure Injection Capabilities**:
- **Network Timeouts**: Configurable delay before response
- **Connection Failures**: Simulate network connectivity issues
- **Partial Downloads**: Return incomplete responses
- **Corrupted Data**: Serve models with incorrect checksums
- **HTTP Errors**: Return 4xx/5xx status codes intermittently
- **Flaky Responses**: Randomly succeed/fail to simulate unreliable networks

### 3. Test Setup and Model Preparation

**Pre-test Model Acquisition**:
```rust
// Download real models once during test setup to a reliable cache
// This ensures we have good copies to serve from our mock server
fn setup_test_models(test_cache: &Path) -> TestModelStore {
    // Download current production models to test_cache/reliable/
    // - cutout model (larger, good for network stress)
    // - alternative head model from issue #22
    // - validate checksums and store metadata
}
```

**Mock Server Initialization**:
```rust
fn start_mock_server(models: &TestModelStore) -> MockServer {
    // Start httpmock server on random port
    // Configure routes to serve test models with controllable failures
    // Return server handle with failure injection controls
}
```

### 4. Concurrency Test Scenarios

#### Scenario 1: Concurrent Same-Model Downloads
**Purpose**: Test ONNX cache lock contention
```rust
fn test_concurrent_same_model_download() {
    // Launch 10 beaker processes simultaneously requesting same model
    // Verify only one downloads, others wait for completion
    // Ensure all processes end up with valid cached model
    // Check lock file cleanup
}
```

#### Scenario 2: Concurrent Different-Model Downloads
**Purpose**: Test cache isolation and parallel downloads
```rust
fn test_concurrent_different_model_downloads() {
    // Launch processes requesting different models simultaneously
    // Verify parallel downloads work without interference
    // Check each model caches correctly with proper checksums
}
```

#### Scenario 3: Network Failure During Download
**Purpose**: Test failure recovery and cache consistency
```rust
fn test_network_failure_recovery() {
    // Start downloads, inject failures partway through
    // Verify partial downloads are cleaned up
    // Ensure retry logic works correctly
    // Check lock files don't become stale
}
```

#### Scenario 4: Corrupted Download Handling
**Purpose**: Test checksum validation under concurrent access
```rust
fn test_corrupted_download_handling() {
    // Serve model with wrong checksum to some processes
    // Verify corrupted downloads are rejected
    // Ensure cache doesn't get poisoned
    // Check concurrent processes handle failures independently
}
```

#### Scenario 5: Cache Directory Stress
**Purpose**: Test filesystem operations under concurrency
```rust
fn test_cache_directory_stress() {
    // Multiple processes creating cache directories
    // Concurrent file creation and deletion
    // Test permission and ownership handling
    // Verify cache cleanup operations
}
```

#### Scenario 6: CoreML Cache Concurrency (macOS only)
**Purpose**: Test CoreML compilation cache under load
```rust
#[cfg(target_os = "macos")]
fn test_coreml_cache_concurrency() {
    // Multiple processes compiling same ONNX model to CoreML
    // Verify CoreML cache directory isolation
    // Check concurrent compilation handling
    // Note: Limited by ONNX Runtime API capabilities
}
```

### 5. Failure Injection Strategies

**Gradual Failure Introduction**:
```rust
struct FailureController {
    failure_rate: Arc<AtomicU32>,        // 0-100% failure rate
    timeout_ms: Arc<AtomicU64>,          // Response delay
    corruption_rate: Arc<AtomicU32>,     // % of responses corrupted
}

impl FailureController {
    // Start with 0% failures, gradually increase during test
    // Allows observing system behavior under increasing stress
    // Can identify failure threshold where system breaks down
}
```

**Targeted Failure Patterns**:
- **Early Failures**: Fail immediately on connection
- **Mid-Download Failures**: Start transfer, then disconnect
- **Checksum Corruption**: Modify content to invalidate MD5
- **Slow Networks**: Introduce realistic network delays
- **Intermittent Connectivity**: Random success/failure patterns

### 6. Process Management and Orchestration

**Concurrent Process Execution**:
```rust
struct StressTestOrchestrator {
    max_concurrent_processes: usize,
    test_duration_seconds: u64,
    process_spawn_interval_ms: u64,
}

impl StressTestOrchestrator {
    fn run_stress_test(&self, scenario: TestScenario) -> StressTestResults {
        // Spawn beaker processes with controlled timing using threads
        // Monitor process lifecycle and results
        // Collect exit codes, timing, and error messages
        // Ensure proper cleanup of all processes
    }
}
```

**Process Isolation**:
- Each process gets its own temporary cache directory
- Environment variables control model URLs to point to mock server
- Separate output directories to avoid conflicts
- Independent metadata file generation

### 7. Validation and Metrics Collection

**Cache State Validation**:
```rust
struct CacheValidator {
    fn validate_cache_consistency(&self, cache_dir: &Path) -> ValidationResult {
        // Check no partial/corrupted files remain
        // Verify all cached models have correct checksums
        // Ensure proper file permissions and ownership
        // Check cache directory structure integrity
    }

    fn validate_concurrent_safety(&self, cache_dirs: &[Path]) -> ValidationResult {
        // Compare cache states across process directories
        // Verify identical models have identical cache entries
        // Check for race condition artifacts
        // Validate lock file cleanup
    }
}
```

**Performance Metrics**:
- Download completion rates under various failure scenarios
- Cache hit/miss ratios during concurrent access
- Lock contention duration and frequency
- Process completion times and failure modes
- Network bandwidth utilization efficiency

**Metadata Analysis**:
- Leverage issue #35 cache metadata when available
- Track model load times under concurrent access
- Monitor cache hit/miss patterns
- Analyze failure recovery timing

### 8. Test Implementation Strategy

#### Phase 1: Foundation Infrastructure
1. Add httpmock dependency to `[dev-dependencies]`
2. Create `tests/stress/` module structure
3. Implement basic mock server with model serving
4. Build process orchestration framework
5. Add cache validation utilities

#### Phase 2: Basic Concurrency Tests
1. Implement concurrent same-model download test
2. Add concurrent different-model download test
3. Create cache validation for these scenarios
4. Ensure tests run reliably in CI environment

#### Phase 3: Failure Injection
1. Add network failure simulation capabilities
2. Implement corrupted download testing
3. Create progressive failure rate testing
4. Add timeout and slow network simulation

#### Phase 4: Advanced Scenarios
1. Cache directory stress testing
2. CoreML cache testing (macOS)
3. Long-running stability tests
4. Resource exhaustion scenarios

#### Phase 5: Integration and CI
1. Integrate with existing test framework
2. Add CI job for stress testing
3. Performance regression detection
4. Documentation and maintenance guides

### 9. Implementation Example

Here's a concrete example of how the stress test framework would look:

```rust
// tests/stress/mod.rs
use httpmock::{MockServer, Mock, When, Then};
use std::process::Command;
use std::time::Duration;
use std::thread;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[test]
fn test_concurrent_model_download_stress() {
    // Setup: Download real model for serving
    let test_models = setup_test_models();

    // Start mock server with failure injection
    let mock_server = MockServer::start();
    let failure_controller = Arc::new(FailureController::new());

    // Configure mock responses
    let failure_controller_clone = Arc::clone(&failure_controller);
    Mock::new()
        .expect_request(When::path("/cutout-model.onnx"))
        .return_response_with(move |_| {
            if failure_controller_clone.should_fail() {
                Then::new().status(500) // Simulate server error
            } else if failure_controller_clone.should_timeout() {
                Then::new()
                    .status(200)
                    .delay(Duration::from_secs(30)) // Simulate slow network
                    .body(test_models.cutout_model_bytes.clone())
            } else {
                Then::new()
                    .status(200)
                    .body(test_models.cutout_model_bytes.clone())
            }
        })
        .create_on(&mock_server);

    // Environment setup for beaker processes
    let temp_cache = TempDir::new().unwrap();
    let mock_url = format!("{}/cutout-model.onnx", mock_server.base_url());

    // Launch concurrent beaker processes using threads
    let handles: Vec<_> = (0..10).map(|i| {
        let cache_dir = temp_cache.path().join(format!("cache_{}", i));
        let output_dir = temp_cache.path().join(format!("output_{}", i));
        let mock_url = mock_url.clone();

        thread::spawn(move || {
            let exit_code = Command::new("./target/debug/beaker")
                .args(&[
                    "cutout",
                    "../example.jpg",
                    "--metadata",
                    "--output-dir", output_dir.to_str().unwrap()
                ])
                .env("CUTOUT_MODEL_URL", mock_url)
                .env("ONNX_MODEL_CACHE_DIR", cache_dir.to_str().unwrap())
                .status()
                .unwrap()
                .code()
                .unwrap_or(-1);

            (i, exit_code, cache_dir, output_dir)
        })
    }).collect();

    // Introduce failures during execution using a separate thread
    let failure_controller_for_injection = Arc::clone(&failure_controller);
    let _failure_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        failure_controller_for_injection.set_failure_rate(20); // 20% failure rate

        thread::sleep(Duration::from_secs(2));
        failure_controller_for_injection.set_timeout_rate(30); // 30% timeout rate
    });

    // Collect results
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    // Validation
    let success_count = results.iter()
        .filter(|r| r.1 == 0)
        .count();

    // At least some processes should succeed despite failures
    assert!(success_count >= 5, "Expected at least 5 successes, got {}", success_count);

    // Validate cache consistency across successful processes
    for result in &results {
        let (process_id, exit_code, cache_dir, output_dir) = result;
        if *exit_code == 0 {
            // Check cache has valid model
            let cached_model = cache_dir.join("cutout-model.onnx");
            assert!(cached_model.exists(), "Process {} should have cached model", process_id);

            // Verify checksum
            let checksum = calculate_md5(&cached_model).unwrap();
            assert_eq!(checksum, test_models.cutout_model_checksum);

            // Check metadata was generated
            let metadata_file = output_dir.join("example.beaker.toml");
            assert!(metadata_file.exists(), "Process {} should have metadata", process_id);
        }
    }

    // Verify no partial downloads remain
    for result in &results {
        let (_, _, cache_dir, _) = result;
        let lock_file = cache_dir.join("cutout-model.onnx.lock");
        assert!(!lock_file.exists(), "No lock files should remain");

        // Check for partial/temporary files
        let cache_entries: Vec<_> = std::fs::read_dir(cache_dir)
            .unwrap()
            .collect();
        for entry in cache_entries {
            let entry = entry.unwrap();
            let filename = entry.file_name();
            assert!(!filename.to_string_lossy().contains(".tmp"),
                   "No temporary files should remain: {:?}", filename);
        }
    }
}

struct FailureController {
    failure_rate: Arc<AtomicU32>,
    timeout_rate: Arc<AtomicU32>,
}

impl FailureController {
    fn new() -> Self {
        Self {
            failure_rate: Arc::new(AtomicU32::new(0)),
            timeout_rate: Arc::new(AtomicU32::new(0)),
        }
    }

    fn should_fail(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0..100) < self.failure_rate.load(Ordering::Relaxed)
    }

    fn should_timeout(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0..100) < self.timeout_rate.load(Ordering::Relaxed)
    }

    fn set_failure_rate(&self, rate: u32) {
        self.failure_rate.store(rate, Ordering::Relaxed);
    }

    fn set_timeout_rate(&self, rate: u32) {
        self.timeout_rate.store(rate, Ordering::Relaxed);
    }
}
```

### 10. Required Dependencies

The following dependencies should be added to `[dev-dependencies]` in `Cargo.toml`:

```toml
[dev-dependencies]
# Existing dependencies
tempfile = "3.8"
paste = "1.0"

# New dependencies for stress testing
httpmock = "0.7.0"              # HTTP mocking with failure injection
rand = "0.8"                    # For probabilistic failure injection
serde_json = "1.0"              # For test result serialization
```

These dependencies are:
- **Minimal**: Only 3 additional crates for comprehensive stress testing
- **Well-maintained**: All are industry standard with active development
- **Zero runtime impact**: Only in dev-dependencies, no production overhead
- **Compatible**: Work well with beaker's existing blocking/synchronous model
- **No async complexity**: Avoids tokio/futures dependencies that would complicate the codebase

### 11. Test Configuration and Controls

**Environment Variables**:
```rust
// Control test execution behavior
BEAKER_STRESS_TEST_DURATION=60        // Test duration in seconds
BEAKER_STRESS_MAX_PROCESSES=20        // Max concurrent processes
BEAKER_STRESS_FAILURE_RATE=10         // Initial failure rate %
BEAKER_STRESS_ENABLE_COREML=true      // Test CoreML on macOS
BEAKER_STRESS_CACHE_SIZE_LIMIT=1GB    // Cache size constraints
```

**Test Selection**:
```rust
// Granular test control for development
cargo test stress::concurrency::same_model
cargo test stress::failures::network_timeout
cargo test stress::coreml --features=macos-only
cargo test stress --release -- --test-threads=1  // For deterministic results
```

### 12. Success Criteria

**Functional Requirements**:
- ✅ No cache corruption under any concurrent access pattern
- ✅ Lock files properly cleaned up after process termination
- ✅ Checksum validation never passes for corrupted data
- ✅ Failed downloads don't leave partial files in cache
- ✅ Concurrent processes don't deadlock or hang indefinitely

**Performance Requirements**:
- ✅ Cache hit performance doesn't degrade with concurrency
- ✅ Lock contention doesn't exceed 5 seconds under normal load
- ✅ Download failures don't prevent cache access for other models
- ✅ Memory usage remains bounded during stress testing
- ✅ Network failure recovery completes within reasonable time

**Reliability Requirements**:
- ✅ Tests complete successfully in under 5 minutes
- ✅ No flaky test behavior across multiple runs
- ✅ Deterministic results for given failure injection patterns
- ✅ Clean process termination without resource leaks
- ✅ Comprehensive error reporting for debugging failures

### 13. Implementation Considerations

**Minimizing Development Impact**:
- Stress tests live entirely in `tests/stress/` directory
- No changes to production code paths for test hooks
- Optional dependency on httpmock only in dev-dependencies
- Can be disabled for faster development builds
- Uses simple thread-based concurrency matching beaker's blocking model

**Instrumentation Strategy**:
- Leverage existing metadata output system
- Add cache-specific metrics when issue #35 is implemented
- Use debug logging for detailed concurrency analysis
- Export test results in machine-readable format

**CI Integration**:
- Run basic concurrency tests on every PR
- Extended stress tests on nightly builds
- Platform-specific tests (CoreML on macOS runners)
- Performance regression alerts for significant degradation

**Maintenance and Evolution**:
- Framework designed to accommodate new models (issue #22)
- Extensible failure injection for new failure modes
- Modular test scenarios for incremental development
- Documentation for adding new concurrency test cases

## Related Issues Integration

This stress testing framework is designed to work alongside and validate upcoming features:

### Issue #22 - CLI API for Model Access
- **Integration Point**: Test framework will validate concurrent access when users specify custom model URLs/paths via CLI
- **Stress Scenarios**: Multiple processes using different models specified via CLI arguments
- **Validation**: Ensure cache isolation between different model URLs and local paths

### Issue #35 - Cache Metadata Enhancement
- **Integration Point**: Leverage cache hit/miss metadata for stress test validation
- **Metrics Collection**: Use cache timing and hit rate data to validate performance under load
- **Instrumentation**: Additional metadata will help identify cache contention and performance bottlenecks

The framework is designed to evolve with these features, providing immediate testing capabilities as they are implemented.

## Conclusion

This stress testing framework will provide comprehensive validation of beaker's caching mechanisms under realistic concurrent usage patterns. By using established testing libraries and focusing on deterministic, fast-running tests, it will enhance confidence in cache robustness without impeding development velocity.

The framework is designed to evolve alongside beaker's caching implementation, supporting future enhancements like CLI model selection (#22) and cache metadata improvements (#35), while maintaining fast test execution and reliable results for continuous integration.
