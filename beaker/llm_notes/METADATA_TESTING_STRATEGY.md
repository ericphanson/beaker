# Metadata-Based Testing Strategy

## Overview

Currently, our integration tests rely heavily on parsing log output to verify behavior, which makes them brittle when logging formats change. This document proposes a shift to **metadata-based testing**, where we emit structured data to metadata files and verify behavior by parsing these files instead of logs.

## Current Testing Pain Points

### Log-Dependent Tests
- Tests break when log messages change format (e.g., "Using device: cpu" → "🖥️ Using CPU execution provider")
- Verbose failure messages require parsing stderr output
- Race conditions in parallel test execution affect log consistency
- CoreML vs CPU device selection changes break expectations

### Specific Test Failures
1. **Device Selection Tests**: Expect specific logging patterns for device selection
2. **Batch Processing Tests**: Look for "Processing X images" in logs
3. **Multi-source Tests**: Search for "inputs:" or "Found X image" patterns
4. **Execution Provider Tests**: Parse logs to verify CoreML/CPU usage

## Proposed Solution: Metadata-Based Testing

### Core Concept
Instead of parsing logs, emit **structured execution metadata** to the `.beaker.toml` files and verify behavior by reading these files.

### Metadata Schema Extensions

#### Design Principles

**1. Per-Invocation Scoping**: All metadata (execution, system, input, errors) is scoped per tool invocation. Latest invocation overwrites previous metadata for that tool.

**2. Direct Serialization**: Serialize existing internal state objects to TOML rather than maintaining separate metadata wrappers. Enhance existing structures where needed instead of duplicating.

**3. Config Consolidation**: Move all configuration to `[head.config]` and `[cutout.config]` sections by serializing actual config objects.

#### 1. Head Detection Sections

```toml
[head]
# Core results (existing pattern preserved)
model_version = "bird-head-detector-v1.0.0"
processing_time_ms = 1425.3
timestamp = "2025-08-04T12:34:56.789Z"
bounding_box_path = "example_bounding-box.jpg"
detections = [
    { x1 = 100.0, y1 = 50.0, x2 = 200.0, y2 = 150.0, confidence = 0.95, crop_path = "example_crop_1.jpg" }
]

# All configuration via direct serialization of HeadDetectionConfig
[head.config]
confidence = 0.25
iou_threshold = 0.45
crop = true
bounding_box = true
device = "auto"
sources = ["../example.jpg"]
output_dir = "/tmp/test_output"
skip_metadata = false
strict = true

# Execution context for this head invocation
[head.execution]
timestamp = "2025-08-04T12:34:56.789Z"
beaker_version = "0.1.1"
command_line = ["head", "../example.jpg", "--confidence", "0.25"]
exit_code = 0
total_processing_time_ms = 1425.3

# System info for this head invocation
[head.system]
device_requested = "auto"
device_selected = "coreml"
device_selection_reason = "Auto-selected CoreML (available)"
execution_provider_used = "CoreMLExecutionProvider"
model_source = "embedded"
model_size_bytes = 12183794
model_load_time_ms = 234.5

# Input processing for this head invocation
[head.input]
sources = ["../example.jpg"]
source_types = ["file"]
total_files_found = 1
successful_files = 1
failed_files = 0
strict_mode = true
output_files = ["example_crop_1.jpg", "example_bounding-box.jpg"]

# Errors for this head invocation (if any)
[head.errors]
error_count = 0
errors = []
```

#### 2. Cutout Processing Sections

```toml
[cutout]
# Core results (existing pattern preserved)
processing_time_ms = 6100.0
timestamp = "2025-08-04T12:35:30.123Z"
mask_path = "example_mask.png"

# All configuration via direct serialization of CutoutConfig
[cutout.config]
alpha_matting = true
alpha_matting_foreground_threshold = 200
alpha_matting_background_threshold = 50
alpha_matting_erode_size = 10
background_color = [255, 0, 0, 255]
save_mask = true
device = "cpu"
sources = ["../example.jpg"]
output_dir = "/tmp/test_output"
skip_metadata = false
strict = false

# Execution context for this cutout invocation
[cutout.execution]
timestamp = "2025-08-04T12:35:30.123Z"
beaker_version = "0.1.1"
command_line = ["cutout", "../example.jpg", "--alpha-matting", "--device", "cpu"]
exit_code = 0
total_processing_time_ms = 6100.0

# System info for this cutout invocation
[cutout.system]
device_requested = "cpu"
device_selected = "cpu"
device_selection_reason = "User explicitly chose cpu"
execution_provider_used = "CPUExecutionProvider"
model_source = "downloaded"
model_path = "/Users/eph/Library/Caches/beaker/models/isnet-general-use.onnx"
model_size_bytes = 176000000
model_load_time_ms = 834.2

# Input processing for this cutout invocation
[cutout.input]
sources = ["../example.jpg"]
source_types = ["file"]
total_files_found = 1
successful_files = 1
failed_files = 0
strict_mode = false
output_files = ["example_cutout.png", "example_mask.png"]

# Errors for this cutout invocation (if any)
[cutout.errors]
error_count = 0
errors = []
```

## Test Transformation Analysis

### 1. Device Selection Tests

**Current Approach:**
```rust
assert!(stderr.contains("Using CPU execution provider"));
```

**Metadata Approach:**
```rust
let metadata = parse_toml(&toml_path)?;
assert_eq!(metadata.head.system.execution_provider_used, "CPUExecutionProvider");
assert_eq!(metadata.head.system.device_selected, "cpu");
```

### 2. Batch Processing Tests

**Current Approach:**
```rust
assert!(stderr.contains("Found 3") || stderr.contains("3 image"));
```

**Metadata Approach:**
```rust
let metadata = parse_toml(&toml_path)?;
assert_eq!(metadata.head.input.total_files_found, 3);
assert_eq!(metadata.head.input.successful_files, 3);
```

### 3. Multi-Source Tests

**Current Approach:**
```rust
assert!(stderr.contains("inputs:") || stderr.contains("Found"));
```

**Metadata Approach:**
```rust
let metadata = parse_toml(&toml_path)?;
assert_eq!(metadata.head.input.sources.len(), 2);
assert_eq!(metadata.head.input.source_types, vec!["file", "directory"]);
```

### 4. Configuration Tests

**Current Approach:**
```rust
assert!(stderr.contains("alpha matting"));
```

**Metadata Approach:**
```rust
let metadata = parse_toml(&toml_path)?;
assert_eq!(metadata.cutout.config.alpha_matting, true);
assert_eq!(metadata.cutout.config.alpha_matting_foreground_threshold, 200);
```

### 5. Performance Tests

**Current Approach:**
```rust
// No current way to test performance metrics
```

**Metadata Approach:**
```rust
let metadata = parse_toml(&toml_path)?;
assert!(metadata.head.execution.total_processing_time_ms > 0.0);
assert!(metadata.head.system.model_load_time_ms > 0.0);
```

### 6. Command-Line Verification Tests

**New Capability:**
```rust
let metadata = parse_toml(&toml_path)?;
assert_eq!(metadata.head.execution.command_line, vec!["head", "../example.jpg", "--confidence", "0.25"]);
assert_eq!(metadata.head.config.confidence, 0.25);
assert_eq!(metadata.head.config.device, "auto");
```

### 7. Multi-Tool Processing Tests

**New Capability:**
```rust
// Run head detection first
run_command(&["head", "../example.jpg", "--crop"]);
let metadata = parse_toml(&toml_path)?;
assert!(metadata.head.is_some());
assert_eq!(metadata.head.unwrap().input.successful_files, 1);

// Run cutout on same file - overwrites cutout sections only
run_command(&["cutout", "../example.jpg", "--save-mask"]);
let metadata = parse_toml(&toml_path)?;
assert!(metadata.head.is_some()); // Preserved from previous run
assert!(metadata.cutout.is_some()); // New from current run
assert!(metadata.cutout.unwrap().mask_path.is_some());
```

## Implementation Strategy

### Phase 1: Direct Serialization Infrastructure
1. **Config Serialization**: Modify `HeadDetectionConfig` and `CutoutConfig` to derive `Serialize` and emit to `[tool.config]` sections
2. **State Enhancement**: Add missing fields to existing internal state objects (execution context, system info, input stats)
3. **Serialization Points**: Identify where to capture and serialize state during processing pipeline
4. **Per-Tool Scoping**: Ensure each tool invocation only overwrites its own sections, preserving other tool data

### Phase 2: Enhanced State Collection
1. **Execution Context**: Capture command-line args, timestamps, exit codes in processing pipeline
2. **System Information**: Collect device selection, model loading metrics, execution provider details
3. **Input Processing**: Track source analysis, file discovery, success/failure statistics
4. **Error Collection**: Structured error capture with file-level granularity
5. **Timing Integration**: Add timing fields to existing result structures

### Phase 3: Serialization Integration
1. **Metadata Output**: Extend existing metadata save functions to include new sections
2. **State Passing**: Thread enhanced state through processing pipeline without duplication
3. **Section Management**: Implement per-tool section overwriting while preserving other tools
4. **Schema Validation**: Ensure serialized state matches expected TOML structure

### Phase 4: New Test Framework
1. **Metadata Parsing**: Build utilities to parse and validate tool-scoped metadata sections
2. **Scenario Framework**: Implement loop-based test scenarios with tool-specific validation
3. **Assertion Helpers**: Create type-safe assertions for common metadata checks
4. **Multi-Tool Tests**: Test scenarios involving sequential tool invocations on same files

## Opinionated Design Decisions

### 1. Latest Invocation Wins
**Decision**: Only the most recent invocation of each tool matters. Previous metadata for that tool is overwritten.

**Rationale**:
- Simplifies metadata management - no versioning or history tracking needed
- Reflects actual user workflow - latest results are what matter for testing
- Avoids metadata file bloat from repeated invocations
- Makes test assertions predictable - always testing current state

**Trade-off**: Lose historical execution information, but gain simplicity and predictability.

### 2. Direct State Serialization
**Decision**: Serialize existing internal state objects directly to TOML rather than maintaining separate metadata structures.

**Rationale**:
- **No Duplication**: Avoid maintaining parallel metadata structures that can drift from actual state
- **Single Source of Truth**: Internal state objects become the definitive source for both execution and testing
- **Automatic Sync**: Changes to internal state automatically reflect in metadata
- **Type Safety**: Leverage existing type definitions and validation

**Trade-off**: Internal state objects become part of the external API (TOML schema), but this enforces good design and makes testing more accurate.

### 3. Configuration Consolidation
**Decision**: Move all configuration to `[tool.config]` sections by serializing actual config objects.

**Rationale**:
- **Complete Visibility**: Tests can verify every configuration option was applied correctly
- **CLI Validation**: Verify command-line parsing produced expected internal state
- **Consistency**: Same configuration representation in code and tests
- **Extensibility**: New config options automatically become testable

**Trade-off**: Larger metadata files, but comprehensive testability of configuration state.

### 4. Tool-Scoped Everything
**Decision**: All metadata sections (execution, system, input, errors) are scoped per tool.

**Rationale**:
- **Clear Ownership**: Each tool owns its complete execution context
- **Independent Testing**: Can test each tool's behavior independently
- **Multi-Tool Workflows**: Support testing sequential tool usage on same files
- **Isolation**: Tool failures don't affect other tool's metadata

**Trade-off**: More metadata volume, but clearer test semantics and better multi-tool support.

### 5. Enhanced Internal State
**Decision**: Add fields to existing internal state objects rather than creating separate metadata structures.

**Rationale**:
- **Truth at Source**: State enhancement happens where execution occurs
- **No Translation Layer**: Direct serialization without conversion overhead
- **Better Code Design**: Forces us to track execution details we should already be tracking
- **Debugging Benefits**: Enhanced state helps with runtime debugging too

**Trade-off**: Slightly more complex internal state, but better observability and testing capabilities.

## New Testing Approach: Comprehensive from Scratch

Instead of converting existing brittle tests, we'll design comprehensive test coverage from the ground up using metadata validation.

### Test Speed Strategy

**Performance Characteristics:**
- **Head Model**: Very fast (milliseconds) - enables comprehensive testing
- **Cutout Model**: Very slow (seconds) - requires selective testing
- **Shared Architecture**: Both models use unified framework, so head testing covers most shared code paths

**Testing Strategy:**
- **Head Detection**: Test comprehensively with all option combinations (fast execution enables this)
- **Background Removal**: Test sparsely, focusing on cutout-specific features (alpha matting, mask output, background colors)
- **Performance Tracking**: Metadata includes per-test timing and model invocation counts
- **Speed Target**: Keep total test suite under 30 seconds

**Test Distribution:**
- ~70% head detection scenarios (comprehensive coverage, fast execution)
- ~20% cutout scenarios (targeted feature testing, slow execution)
- ~10% multi-tool integration scenarios (essential workflows)

### Test Categories to Cover

#### 1. Core Functionality Tests
- **Head Detection**: All confidence/IoU combinations, crop/bbox options, device matrix (comprehensive)
- **Background Removal**: Core alpha matting features, mask saving, background colors (selective)
- **Multi-Model**: Essential head→cutout processing workflows

#### 2. Device and Performance Tests
- **Device Selection**: Auto/CPU/CoreML across different batch sizes (head model focus)
- **Performance Regression**: Timing bounds for model loading and inference (both models)
- **Resource Usage**: Memory and disk usage validation (lightweight checks)
- **Cross-Platform**: Device availability and fallback behavior (head model focus)

#### 3. Input/Output Tests
- **Input Sources**: Files, directories, globs, mixed combinations (head model focus)
- **File Format**: JPEG, PNG, handling, format preservation (head model focus)
- **Batch Processing**: 1, 3, 10+ image scenarios (head model, single cutout test)
- **Error Handling**: Missing files, invalid formats, permission issues (shared framework)

#### 4. Configuration Tests
- **All Flag Combinations**: Systematic testing of feature flags (head comprehensive, cutout selective)
- **Output Patterns**: Naming conventions, directory structures (shared framework)
- **Metadata Completeness**: Verify all expected fields are populated (both models)
- **Performance Monitoring**: Track test execution times and model invocation counts

### Loop-Based Test Implementation

```rust
#[derive(Debug)]
struct TestScenario {
    name: &'static str,
    tool: &'static str, // "head" or "cutout"
    args: Vec<&'static str>,
    expected_files: Vec<&'static str>,
    metadata_checks: Vec<MetadataCheck>,
}

enum MetadataCheck {
    // Tool-scoped checks
    DeviceUsed(&'static str, &'static str), // tool, device
    FilesProcessed(&'static str, usize), // tool, count
    ConfigValue(&'static str, &'static str, serde_json::Value), // tool, field_path, expected_value
    TimingBound(&'static str, &'static str, f64, f64), // tool, field, min_ms, max_ms
    OutputCreated(&'static str, &'static str), // tool, filename
    ErrorCount(&'static str, usize), // tool, expected_errors
    CommandLine(&'static str, Vec<&'static str>), // tool, expected_args
}

fn test_scenario_matrix() {
    let scenarios = vec![
        TestScenario {
            name: "head_detection_cpu_single_image",
            tool: "head",
            args: vec!["../example.jpg", "--device", "cpu", "--confidence", "0.25"],
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::DeviceUsed("head", "cpu"),
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::ConfigValue("head", "config.confidence", json!(0.25)),
                MetadataCheck::ConfigValue("head", "config.device", json!("cpu")),
                MetadataCheck::TimingBound("head", "execution.total_processing_time_ms", 10.0, 500.0), // Head model: very fast
                MetadataCheck::CommandLine("head", vec!["head", "../example.jpg", "--device", "cpu", "--confidence", "0.25"]),
            ],
        },
        TestScenario {
            name: "head_detection_comprehensive_matrix",
            tool: "head",
            args: vec!["../example.jpg", "--confidence", "0.5", "--iou-threshold", "0.3", "--crop", "--bounding-box"],
            expected_files: vec!["example.beaker.toml", "example_crop_1.jpg", "example_bounding-box.jpg"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("head", "config.confidence", json!(0.5)),
                MetadataCheck::ConfigValue("head", "config.iou_threshold", json!(0.3)),
                MetadataCheck::ConfigValue("head", "config.crop", json!(true)),
                MetadataCheck::ConfigValue("head", "config.bounding_box", json!(true)),
                MetadataCheck::TimingBound("head", "execution.total_processing_time_ms", 10.0, 500.0),
                MetadataCheck::OutputCreated("head", "example_crop_1.jpg"),
                MetadataCheck::OutputCreated("head", "example_bounding-box.jpg"),
            ],
        },
        TestScenario {
            name: "cutout_with_alpha_matting",
            tool: "cutout",
            args: vec!["../example.jpg", "--alpha-matting", "--save-mask"],
            expected_files: vec!["example.beaker.toml", "example_cutout.png", "example_mask.png"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("cutout", "config.alpha_matting", json!(true)),
                MetadataCheck::ConfigValue("cutout", "config.save_mask", json!(true)),
                MetadataCheck::OutputCreated("cutout", "example_cutout.png"),
                MetadataCheck::OutputCreated("cutout", "example_mask.png"),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::TimingBound("cutout", "execution.total_processing_time_ms", 1000.0, 15000.0), // Cutout model: slow
            ],
        },
        TestScenario {
            name: "cutout_background_color_only", // Selective cutout testing - just core features
            tool: "cutout",
            args: vec!["../example.jpg", "--background-color", "255,0,0,255"],
            expected_files: vec!["example.beaker.toml", "example_cutout.png"],
            metadata_checks: vec![
                MetadataCheck::ConfigValue("cutout", "config.background_color", json!([255, 0, 0, 255])),
                MetadataCheck::ConfigValue("cutout", "config.alpha_matting", json!(false)),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::TimingBound("cutout", "execution.total_processing_time_ms", 1000.0, 15000.0),
            ],
        },
        TestScenario {
            name: "multi_tool_sequential",
            tool: "both", // Special case for sequential execution
            args: vec![], // Handled specially
            expected_files: vec!["example.beaker.toml"],
            metadata_checks: vec![
                MetadataCheck::FilesProcessed("head", 1),
                MetadataCheck::FilesProcessed("cutout", 1),
                MetadataCheck::ConfigValue("head", "config.crop", json!(true)),
                MetadataCheck::ConfigValue("cutout", "config.save_mask", json!(true)),
                MetadataCheck::TimingBound("head", "execution.total_processing_time_ms", 10.0, 500.0), // Fast head
                MetadataCheck::TimingBound("cutout", "execution.total_processing_time_ms", 1000.0, 15000.0), // Slow cutout
            ],
        },
        // Add more head detection scenarios here (comprehensive testing)
        // - Device matrix: auto/cpu/coreml
        // - Confidence/IoU combinations: 0.1-0.9 range
        // - Batch sizes: 1, 3, 10 images
        // - Output options: crop/bbox combinations
        // - Input types: file/directory/glob patterns
        // Total: ~15-20 head scenarios (fast execution enables comprehensive testing)

        // Add selective cutout scenarios here (targeted testing)
        // - Alpha matting core features
        // - Background color options
        // - Mask output validation
        // - Device selection (minimal)
        // Total: ~5-7 cutout scenarios (slow execution requires selectivity)

        // Add multi-tool integration scenarios
        // - Essential workflows only
        // - Focus on tool interaction, not comprehensive option testing
        // Total: ~3-5 integration scenarios
    ];

    for scenario in scenarios {
        run_and_validate_scenario(scenario);
    }
}

fn run_and_validate_scenario(scenario: TestScenario) {
    let temp_dir = TempDir::new().unwrap();
    let start_time = std::time::Instant::now();

    // Handle special multi-tool case
    if scenario.tool == "both" {
        run_command(&["head", "../example.jpg", "--crop", "--output-dir", temp_dir.path()]);
        run_command(&["cutout", "../example.jpg", "--save-mask", "--output-dir", temp_dir.path()]);
    } else {
        let mut full_args = vec![scenario.tool];
        full_args.extend_from_slice(&scenario.args);
        full_args.extend_from_slice(&["--output-dir", temp_dir.path().to_str().unwrap()]);
        run_command(&full_args);
    }

    let test_duration = start_time.elapsed();

    // Track performance metrics
    record_test_performance(&scenario.name, scenario.tool, test_duration);

    // Parse metadata and validate
    let metadata_path = temp_dir.path().join("example.beaker.toml");
    assert!(metadata_path.exists(), "Metadata file should exist for scenario: {}", scenario.name);

    let metadata_content = fs::read_to_string(&metadata_path).unwrap();
    let metadata: BeakerMetadata = toml::from_str(&metadata_content)
        .expect(&format!("Failed to parse metadata for scenario: {}\nContent: {}", scenario.name, metadata_content));

    // Validate all checks
    for check in scenario.metadata_checks {
        validate_metadata_check(&metadata, &check, &scenario.name);
    }
}

struct TestPerformanceTracker {
    head_invocations: u32,
    cutout_invocations: u32,
    total_test_time: Duration,
    slowest_tests: Vec<(String, Duration)>,
}

fn record_test_performance(test_name: &str, tool: &str, duration: Duration) {
    // Track per-test timing and model invocation counts
    // Warn if any single test takes > 5 seconds
    // Fail if total test suite exceeds 30 seconds
    // Report slowest tests for optimization opportunities

    if duration.as_secs() > 5 {
        eprintln!("⚠️  Slow test: {} took {:.2}s", test_name, duration.as_secs_f64());
    }

    match tool {
        "head" => increment_head_count(),
        "cutout" => increment_cutout_count(),
        "both" => {
            increment_head_count();
            increment_cutout_count();
        }
        _ => {}
    }
}
```

### Benefits of New Approach

#### Comprehensive Coverage
- **Systematic Testing**: Matrix of all feature combinations (balanced for speed)
- **Consistent Patterns**: Same validation logic across all tests
- **Easy Expansion**: Add new scenarios without rewriting infrastructure
- **Performance Tracking**: Built-in regression detection and test speed monitoring

#### Speed Optimization
- **Fast Model Focus**: Comprehensive testing on head detection (milliseconds)
- **Selective Slow Testing**: Targeted cutout model testing (seconds)
- **Performance Budgets**: 30-second total test suite target with per-test monitoring
- **Model Invocation Tracking**: Count and optimize expensive cutout model usage

#### Maintainability
- **Schema-Driven**: Tests automatically adapt to metadata schema changes
- **Single Source of Truth**: Metadata format drives test expectations
- **Reduced Duplication**: Shared validation logic across test scenarios
- **Clear Intent**: Each test scenario explicitly states what it validates
- **Speed Feedback**: Immediate warnings for slow tests, optimization guidance

## Pros and Cons Analysis

### Pros ✅

#### Robustness
- **Log Format Independence**: Tests won't break when log messages change
- **Structured Data**: Type-safe parsing instead of string matching
- **Comprehensive Coverage**: Can test aspects not visible in logs
- **Deterministic**: No race conditions or log interleaving issues

#### Maintainability
- **Clear Intent**: Tests explicitly check what they care about
- **Better Debugging**: Rich metadata dumps on test failures
- **Version Tracking**: Built-in versioning of test expectations
- **Self-Documenting**: Metadata serves as execution documentation

#### Enhanced Capabilities
- **Performance Testing**: Built-in timing and performance metrics
- **Configuration Validation**: Verify all settings were applied correctly
- **Cross-Platform Testing**: Device selection behavior verification
- **Error Analysis**: Structured error information for debugging

#### Developer Experience
- **Faster Test Development**: Clear metadata structure to build against
- **Rich Failure Information**: Immediate access to execution context
- **Regression Detection**: Automated detection of behavior changes
- **Documentation**: Metadata files serve as execution examples

### Cons ❌

#### Implementation Complexity
- **Schema Maintenance**: Need to keep metadata schema up-to-date
- **Collection Overhead**: Additional code to gather and emit metadata
- **Backwards Compatibility**: Managing schema evolution over time
- **Testing the Tests**: Need to verify metadata collection accuracy

#### Performance Impact
- **Metadata Generation**: Small overhead to collect and serialize data
- **File I/O**: Additional disk writes for metadata
- **Memory Usage**: Keeping execution context in memory during processing

#### Limitations
- **Runtime Behavior**: Can't test real-time log output or streaming behavior
- **User Experience**: Still need logs for user-facing communication validation
- **Debug Information**: May need logs for troubleshooting test issues
- **Schema Dependencies**: Tests depend on metadata file format (acceptable trade-off)

#### Migration Effort
- **New Test Framework**: Build comprehensive tests from scratch using metadata
- **Schema Design**: Upfront cost to design metadata extensions thoughtfully
- **Validation**: Need to ensure metadata accurately reflects execution
- **Tooling**: Build parsing and assertion utilities for metadata-driven tests
- **Legacy Coexistence**: Maintain existing tests during transition period

## Risk Assessment

### Low Risk
- **Metadata Collection**: Small, incremental changes to existing code
- **Schema Evolution**: Can be versioned and backward-compatible
- **Test Conversion**: Can be done incrementally, test by test

### Medium Risk
- **Performance Impact**: Need to measure overhead of metadata generation
- **Schema Complexity**: Risk of over-engineering metadata structure
- **Maintenance Burden**: Additional code surface area to maintain

### High Risk
- **False Positives**: Metadata might not reflect actual user experience
- **Test Coverage**: Risk of missing edge cases not captured in metadata
- **Development Timeline**: Large effort to build comprehensive new test framework
- **Schema Breaking Changes**: Acceptable trade-off for better testing infrastructure

## Recommendation

**Proceed with Phased Implementation**

The metadata-based testing strategy offers significant advantages in test robustness and maintainability. The structured approach addresses current pain points while enabling enhanced testing capabilities.

**Recommended Approach:**
1. Design comprehensive metadata schema extensions while preserving existing `[head]` and `[cutout]` sections
2. Build new test framework from scratch with loop-based scenario testing
3. Focus on systematic coverage of all feature combinations and edge cases
4. Maintain existing verbose logging for real-time user communication
5. Allow breaking schema changes to achieve optimal testing infrastructure
6. Keep existing tests running during development of new framework

**Success Criteria:**
- Tests pass consistently regardless of log format changes
- Comprehensive coverage of head detection features (fast model enables this)
- Selective but sufficient coverage of cutout features (slow model requires optimization)
- Total test suite execution under 30 seconds
- Built-in performance regression detection and timing validation
- Clear, actionable test failures with rich debugging information
- Systematic test expansion as new features are added
- Enhanced confidence in cross-platform and edge-case behavior
- Performance tracking and optimization guidance for slow tests

This strategy transforms testing from fragile log-parsing to a robust, systematic validation framework that provides comprehensive coverage while being maintainable and extensible.
