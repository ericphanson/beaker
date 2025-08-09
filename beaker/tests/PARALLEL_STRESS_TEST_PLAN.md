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
- **Failure Resilience**: Validate graceful handling of network failures, corruption, and connection issues
- **Deterministic**: Event-based testing with eventual invariants - no timing dependencies
- **Fast Tests**: Virtualized time and immediate failure injection for rapid CI execution
- **No Flakiness**: Logic-based validation that always converges to correct state

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
- **Connection Failures**: Simulate network connectivity issues (immediate)
- **Partial Downloads**: Return incomplete responses at specific byte counts
- **Corrupted Data**: Serve models with incorrect checksums (deterministic corruption)
- **HTTP Errors**: Return 4xx/5xx status codes at predetermined points
- **Deterministic Patterns**: Predefined failure sequences for reproducible testing

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

### 5. Deterministic Failure Injection

**Event-Based Failure Controller**:
```rust
struct FailureController {
    failure_sequence: Vec<FailureEvent>,
    current_request: AtomicUsize,
}

#[derive(Clone)]
enum FailureEvent {
    Success,
    ConnectionRefused,
    PartialResponse(usize), // Fail after N bytes
    CorruptedChecksum,
    Http500Error,
}

impl FailureController {
    fn new(pattern: Vec<FailureEvent>) -> Self {
        Self {
            failure_sequence: pattern,
            current_request: AtomicUsize::new(0),
        }
    }
    
    fn next_response(&self) -> FailureEvent {
        let index = self.current_request.fetch_add(1, Ordering::SeqCst);
        self.failure_sequence[index % self.failure_sequence.len()].clone()
    }
}
```

**Predefined Failure Patterns**:
- **Progressive Degradation**: `[Success, Success, PartialResponse(1024), Success, ConnectionRefused]`
- **Checksum Corruption**: `[Success, CorruptedChecksum, Success, Success]`
- **Server Instability**: `[Http500Error, Http500Error, Success, Success]`
- **Partial Download Recovery**: `[PartialResponse(512), PartialResponse(1024), Success]`

### 6. Process Management and Orchestration

**Event-Based Process Coordination**:
```rust
use std::sync::{Arc, Barrier, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};

struct StressTestOrchestrator {
    max_concurrent_processes: usize,
    failure_patterns: Vec<Vec<FailureEvent>>,
}

#[derive(Debug)]
struct ProcessResult {
    process_id: usize,
    exit_code: i32,
    cache_state: CacheValidationResult,
    error_output: String,
}

impl StressTestOrchestrator {
    fn run_stress_test(&self, scenario: TestScenario) -> StressTestResults {
        let start_barrier = Arc::new(Barrier::new(self.max_concurrent_processes + 1));
        let (result_tx, result_rx): (Sender<ProcessResult>, Receiver<ProcessResult>) = channel();
        
        // Launch processes with deterministic failure patterns
        let handles: Vec<_> = (0..self.max_concurrent_processes).map(|i| {
            let barrier = Arc::clone(&start_barrier);
            let tx = result_tx.clone();
            let failure_pattern = self.failure_patterns[i % self.failure_patterns.len()].clone();
            
            std::thread::spawn(move || {
                // Wait for all processes to be ready
                barrier.wait();
                
                // Execute beaker with deterministic environment
                let result = execute_beaker_process(i, &failure_pattern);
                tx.send(result).unwrap();
            })
        }).collect();
        
        // Start all processes simultaneously
        start_barrier.wait();
        
        // Collect results as they complete (no timeouts)
        let mut results = Vec::new();
        for _ in 0..self.max_concurrent_processes {
            results.push(result_rx.recv().unwrap());
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        StressTestResults::new(results)
    }
}
```

**Process Isolation**:
- Each process gets its own temporary cache directory
- Environment variables control model URLs to point to mock server with specific failure pattern
- Separate output directories to avoid conflicts
- Independent metadata file generation
- No timing dependencies - processes coordinate through barriers and channels

### 7. Validation and Metrics Collection

**Cache State Validation**:
```rust
struct CacheValidator;

impl CacheValidator {
    fn validate_cache_consistency(&self, cache_dir: &Path) -> ValidationResult {
        // Check no partial/corrupted files remain
        // Verify all cached models have correct checksums
        // Ensure proper file permissions and ownership
        // Check cache directory structure integrity
        ValidationResult::from_invariants(&[
            self.no_partial_files_remain(cache_dir),
            self.all_cached_models_valid_checksum(cache_dir),
            self.proper_permissions(cache_dir),
            self.directory_structure_intact(cache_dir),
        ])
    }

    fn validate_concurrent_safety(&self, cache_dirs: &[Path]) -> ValidationResult {
        // Compare cache states across process directories
        // Verify identical models have identical cache entries
        // Check for race condition artifacts
        // Validate lock file cleanup
        ValidationResult::from_invariants(&[
            self.identical_models_have_same_cache(cache_dirs),
            self.no_race_condition_artifacts(cache_dirs),
            self.all_lock_files_cleaned_up(cache_dirs),
        ])
    }
    
    // Eventual invariants - these must eventually be true
    fn no_partial_files_remain(&self, cache_dir: &Path) -> bool {
        !cache_dir.read_dir().unwrap()
            .any(|entry| entry.unwrap().file_name().to_string_lossy().contains(".tmp"))
    }
    
    fn all_lock_files_cleaned_up(&self, cache_dirs: &[Path]) -> bool {
        cache_dirs.iter().all(|dir| {
            !dir.read_dir().unwrap()
                .any(|entry| entry.unwrap().file_name().to_string_lossy().ends_with(".lock"))
        })
    }
}
```

**Logical Metrics Collection**:
- Process completion states (success/failure with specific error types)
- Cache consistency invariants (eventual properties that must hold)
- Lock file lifecycle completeness (acquire -> release pattern validation)
- Model integrity verification (checksum consistency across processes)
- Error recovery patterns (how failures propagate and resolve)

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
use std::thread;
use std::sync::{Arc, Barrier, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

#[test]
fn test_concurrent_model_download_stress() {
    // Setup: Download real model for serving
    let test_models = setup_test_models();

    // Create deterministic failure pattern
    let failure_pattern = vec![
        FailureEvent::Success,           // Process 0: succeeds
        FailureEvent::ConnectionRefused, // Process 1: fails initially
        FailureEvent::PartialResponse(1024), // Process 2: partial download
        FailureEvent::Success,           // Process 3: succeeds
        FailureEvent::CorruptedChecksum, // Process 4: gets corrupted data
    ];

    // Start mock server with deterministic responses
    let mock_server = MockServer::start();
    let failure_controller = Arc::new(FailureController::new(failure_pattern));

    // Configure mock responses - no delays, immediate responses
    let failure_controller_clone = Arc::clone(&failure_controller);
    Mock::new()
        .expect_request(When::path("/cutout-model.onnx"))
        .return_response_with(move |_| {
            match failure_controller_clone.next_response() {
                FailureEvent::Success => {
                    Then::new()
                        .status(200)
                        .body(test_models.cutout_model_bytes.clone())
                }
                FailureEvent::ConnectionRefused => {
                    Then::new().status(0) // Connection refused
                }
                FailureEvent::PartialResponse(bytes) => {
                    Then::new()
                        .status(200)
                        .body(test_models.cutout_model_bytes[..bytes].to_vec())
                }
                FailureEvent::CorruptedChecksum => {
                    let mut corrupted = test_models.cutout_model_bytes.clone();
                    corrupted[0] = !corrupted[0]; // Flip first byte
                    Then::new()
                        .status(200)
                        .body(corrupted)
                }
                FailureEvent::Http500Error => {
                    Then::new().status(500)
                }
            }
        })
        .create_on(&mock_server);

    // Environment setup for beaker processes
    let temp_cache = TempDir::new().unwrap();
    let mock_url = format!("{}/cutout-model.onnx", mock_server.base_url());

    // Synchronization for simultaneous start
    let process_count = 5;
    let start_barrier = Arc::new(Barrier::new(process_count + 1));
    let (result_tx, result_rx): (Sender<ProcessResult>, Receiver<ProcessResult>) = channel();

    // Launch concurrent beaker processes
    let handles: Vec<_> = (0..process_count).map(|i| {
        let barrier = Arc::clone(&start_barrier);
        let cache_dir = temp_cache.path().join(format!("cache_{}", i));
        let output_dir = temp_cache.path().join(format!("output_{}", i));
        let mock_url = mock_url.clone();
        let tx = result_tx.clone();

        thread::spawn(move || {
            // Wait for all processes to be ready
            barrier.wait();

            // Execute beaker process
            let result = Command::new("./target/debug/beaker")
                .args(&[
                    "cutout",
                    "../example.jpg",
                    "--metadata",
                    "--output-dir", output_dir.to_str().unwrap()
                ])
                .env("CUTOUT_MODEL_URL", mock_url)
                .env("ONNX_MODEL_CACHE_DIR", cache_dir.to_str().unwrap())
                .output()
                .unwrap();

            let exit_code = result.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&result.stderr).to_string();

            tx.send(ProcessResult {
                process_id: i,
                exit_code,
                cache_dir: cache_dir.clone(),
                output_dir: output_dir.clone(),
                error_output: stderr,
            }).unwrap();
        })
    }).collect();

    // Start all processes simultaneously
    start_barrier.wait();

    // Collect results as they complete (no timeouts)
    let mut results = Vec::new();
    for _ in 0..process_count {
        results.push(result_rx.recv().unwrap());
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Validation: Eventual invariants that must hold
    
    // At least some processes should succeed despite deterministic failures
    let success_count = results.iter()
        .filter(|r| r.exit_code == 0)
        .count();
    assert!(success_count >= 2, "Expected at least 2 successes, got {}", success_count);

    // All successful processes must have consistent cache state
    for result in &results {
        if result.exit_code == 0 {
            // Check cache has valid model
            let cached_model = result.cache_dir.join("cutout-model.onnx");
            assert!(cached_model.exists(), "Process {} should have cached model", result.process_id);

            // Verify checksum
            let checksum = calculate_md5(&cached_model).unwrap();
            assert_eq!(checksum, test_models.cutout_model_checksum);

            // Check metadata was generated
            let metadata_file = result.output_dir.join("example.beaker.toml");
            assert!(metadata_file.exists(), "Process {} should have metadata", result.process_id);
        }
    }

    // Eventual invariant: No partial downloads or lock files remain
    for result in &results {
        let lock_file = result.cache_dir.join("cutout-model.onnx.lock");
        assert!(!lock_file.exists(), "No lock files should remain for process {}", result.process_id);

        // Check for partial/temporary files
        if result.cache_dir.exists() {
            let cache_entries: Vec<_> = std::fs::read_dir(&result.cache_dir)
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

    // Validate error patterns match expected failures
    let connection_failures = results.iter()
        .filter(|r| r.exit_code != 0 && r.error_output.contains("connection"))
        .count();
    let checksum_failures = results.iter()
        .filter(|r| r.exit_code != 0 && r.error_output.contains("checksum"))
        .count();
    
    // Expect specific failure types based on our deterministic pattern
    assert!(connection_failures >= 1, "Expected at least 1 connection failure");
    assert!(checksum_failures >= 1, "Expected at least 1 checksum failure");
}

#[derive(Debug)]
struct ProcessResult {
    process_id: usize,
    exit_code: i32,
    cache_dir: PathBuf,
    output_dir: PathBuf,
    error_output: String,
}

#[derive(Clone, Debug)]
enum FailureEvent {
    Success,
    ConnectionRefused,
    PartialResponse(usize),
    CorruptedChecksum,
    Http500Error,
}

struct FailureController {
    failure_sequence: Vec<FailureEvent>,
    current_request: AtomicUsize,
}

impl FailureController {
    fn new(pattern: Vec<FailureEvent>) -> Self {
        Self {
            failure_sequence: pattern,
            current_request: AtomicUsize::new(0),
        }
    }

    fn next_response(&self) -> FailureEvent {
        let index = self.current_request.fetch_add(1, Ordering::SeqCst);
        self.failure_sequence[index % self.failure_sequence.len()].clone()
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
httpmock = "0.7.0"              # HTTP mocking with deterministic failure injection
serde_json = "1.0"              # For test result serialization
```

These dependencies are:
- **Minimal**: Only 2 additional crates for comprehensive stress testing (removed rand dependency)
- **Well-maintained**: All are industry standard with active development
- **Zero runtime impact**: Only in dev-dependencies, no production overhead
- **Compatible**: Work well with beaker's existing blocking/synchronous model
- **No timing dependencies**: Deterministic behavior without sleep or delays
- **No async complexity**: Avoids tokio/futures dependencies that would complicate the codebase

### 11. Test Configuration and Controls

**Environment Variables**:
```rust
// Control test execution behavior
BEAKER_STRESS_MAX_PROCESSES=20        // Max concurrent processes
BEAKER_STRESS_FAILURE_PATTERN=deterministic  // Use predefined failure sequences
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

**Logical Invariants**:
- ✅ Cache consistency across all successful processes (eventual property)
- ✅ Deterministic failure handling matches expected patterns
- ✅ All processes eventually complete (no infinite hanging)
- ✅ Error recovery follows expected paths for each failure type
- ✅ Resource cleanup completes for all process termination scenarios

**Reliability Requirements**:
- ✅ Tests complete rapidly with immediate failure injection (under 30 seconds)
- ✅ No flaky test behavior - deterministic failure patterns
- ✅ Reproducible results for given failure injection sequences
- ✅ Clean process termination without resource leaks
- ✅ Comprehensive error reporting for debugging failures

### 13. Implementation Considerations

**Minimizing Development Impact**:
- Stress tests live entirely in `tests/stress/` directory
- No changes to production code paths for test hooks
- Optional dependency on httpmock only in dev-dependencies
- Can be disabled for faster development builds
- Uses simple thread-based concurrency matching beaker's blocking model

**Deterministic Testing Strategy**:
- Event-based coordination with barriers and channels instead of timing
- Predefined failure sequences ensure reproducible test outcomes
- Logical invariants validation rather than time-based assertions
- Immediate failure injection eliminates flaky timing dependencies

**CI Integration**:
- Run basic concurrency tests on every PR (fast execution under 30 seconds)
- Extended deterministic stress patterns on nightly builds
- Platform-specific tests (CoreML on macOS runners)
- Logical invariant validation for reliable regression detection

**Maintenance and Evolution**:
- Framework designed to accommodate new models (issue #22)
- Extensible failure patterns for new failure modes
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

This stress testing framework will provide comprehensive validation of beaker's caching mechanisms under realistic concurrent usage patterns. By using deterministic failure injection and event-based coordination, it eliminates timing dependencies while ensuring fast execution and reliable results.

The framework focuses on logical invariants and eventual properties that must hold, making tests both robust and maintainable. With immediate failure injection and barrier-based process coordination, tests complete rapidly without the flakiness inherent in time-based testing approaches.

The framework is designed to evolve alongside beaker's caching implementation, supporting future enhancements like CLI model selection (#22) and cache metadata improvements (#35), while maintaining deterministic test execution and comprehensive failure mode coverage for continuous integration.
