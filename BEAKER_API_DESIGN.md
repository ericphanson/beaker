# Beaker Unified Model API Design

## Executive Summary

This document outlines a unified API design for the beaker toolkit that eliminates code duplication between models while maintaining flexibility for model-specific behaviors. The design establishes clear contracts for model implementation while providing substantial shared infrastructure.

## 1. Unified API Overview

### Core Architecture

The unified API consists of four main components working together to provide a complete model processing framework:

```rust
// Core trait that all models implement
pub trait ModelProcessor {
    type Config: ModelConfig;
    type Result: ModelResult;

    // Required implementations (model-specific)
    fn create_session(config: &Self::Config) -> Result<Session>;
    fn preprocess_image(image: &DynamicImage, config: &Self::Config) -> Result<ModelInput>;
    fn run_inference(session: &mut Session, input: ModelInput) -> Result<ModelOutput>;
    fn postprocess_output(output: ModelOutput, config: &Self::Config) -> Result<Self::Result>;
    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()>;

    // Provided implementations (free)
    fn process_single_image(session: &mut Session, image_path: &Path, config: &Self::Config) -> Result<Self::Result>;
}

// Generic batch processing function (free for all models)
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize>;
```

### 1.1 Base Model Framework

**What Models Must Implement (5 methods):**
- `create_session()` - Model-specific ONNX session creation with appropriate providers
- `preprocess_image()` - Convert input image to model-specific tensor format
- `run_inference()` - Execute model inference (usually just session.run())
- `postprocess_output()` - Convert raw model output to meaningful results
- `handle_outputs()` - Save files and generate outputs based on model configuration

**What Models Get For Free:**
- `process_single_image()` - Complete single-image processing pipeline
- `run_model_processing()` - Full batch processing with error handling, timing, and logging
- Session management and device selection
- Input validation and file collection
- Error handling with graceful degradation

### 1.2 Configuration System

**CLI Layer (main.rs):**
```rust
#[derive(Parser)]
pub struct BaseCliArgs {
    pub sources: Vec<String>,
    pub device: String,
    pub output_dir: Option<String>,
    pub no_metadata: bool,
    pub verbose: bool,
    pub permissive: bool,
}

#[derive(Parser)]
pub struct HeadCommand {
    #[command(flatten)]
    pub base: BaseCliArgs,
    pub confidence: f32,
    pub crop: bool,
    // ... model-specific CLI options
}
```

**Internal Config Layer (src/config.rs):**
```rust
pub struct BaseModelConfig {
    pub sources: Vec<String>,
    pub device: String,
    pub output_dir: Option<String>,
    pub skip_metadata: bool,
    pub verbose: bool,
    pub strict: bool,
}

pub struct HeadDetectionConfig {
    pub base: BaseModelConfig,
    pub confidence: f32,
    pub crop: bool,
    // ... model-specific options
}

// Automatic conversion
impl From<HeadCommand> for HeadDetectionConfig { ... }
```

**Benefits:**
- Clean separation between CLI concerns and business logic
- Type-safe conversion with field name normalization
- Easy testing without CLI parsing overhead
- Pipeline can create configs programmatically

### 1.3 Output Management

**Unified Output Generation:**
```rust
pub struct OutputManager<'a> {
    config: &'a dyn ModelConfig,
    input_path: &'a Path,
}

impl<'a> OutputManager<'a> {
    // Core path generation
    pub fn generate_output_path(&self, suffix: Option<&str>, extension: &str) -> Result<PathBuf>;

    // For multiple similar outputs
    pub fn generate_numbered_output(&self, base_suffix: &str, index: usize, total: usize, extension: &str) -> Result<PathBuf>;

    // Metadata utilities
    pub fn make_relative_to_metadata(&self, path: &Path) -> Result<String>;
    pub fn save_metadata<T: Serialize>(&self, data: T, section_name: &str) -> Result<()>;
}
```

**Usage Examples:**
```rust
let output_manager = OutputManager::new(config, image_path)?;

// Single output: "image.png" or "image_cutout.png"
let main_output = output_manager.generate_output_path(Some("cutout"), "png")?;

// Multiple outputs: "image_crop-1.jpg", "image_crop-2.jpg"
for (i, detection) in detections.iter().enumerate() {
    let crop_path = output_manager.generate_numbered_output("crop", i + 1, detections.len(), "jpg")?;
}

// Auxiliary output: "image_mask.png"
let mask_path = output_manager.generate_output_path(Some("mask"), "png")?;
```

### 1.4 Inference Logging

**Centralized Logging:**
```rust
pub struct InferenceLogger<'a> {
    config: &'a dyn ModelConfig,
    model_name: &'static str,
}

impl<'a> InferenceLogger<'a> {
    pub fn log_start(&self, sources: &[String]);
    pub fn log_model_info(&self, model_info: &str, load_time: f64);
    pub fn log_device_selection(&self, device: &DeviceSelection);
    pub fn log_image_processing(&self, path: &Path, index: usize, total: usize);
    pub fn log_inference_result(&self, processing_time: f64, result_summary: &str);
    pub fn log_batch_summary(&self, total_results: usize, total_time: f64);
}
```

**Benefits:**
- Consistent logging format across all models
- Respects verbose flags automatically
- Standardized timing and progress reporting
- Model-specific context while using shared infrastructure

## 2. API Analysis

### 2.1 Pipeline Subcommand Validation

The proposed API handles complex pipeline orchestration elegantly:

**Pipeline Implementation:**
```rust
pub struct PipelineProcessor;

impl ModelProcessor for PipelineProcessor {
    type Config = PipelineConfig;
    type Result = PipelineResult;

    // Standard trait implementation...
    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()> {
        let session_manager = PipelineSessionManager::new();
        let mut current_input = image_path.to_path_buf();

        for step_name in &config.steps {
            match step_name.as_str() {
                "cutout" => {
                    let step_config = CutoutConfig::from_pipeline_config(config, &current_input);
                    let session = session_manager.get_or_create::<CutoutProcessor>(&step_config)?;
                    let result = CutoutProcessor::process_single_image(session, &current_input, &step_config)?;
                    current_input = result.output_path;
                }
                "head" => {
                    let step_config = HeadDetectionConfig::from_pipeline_config(config, &current_input);
                    let session = session_manager.get_or_create::<HeadProcessor>(&step_config)?;
                    HeadProcessor::process_single_image(session, &current_input, &step_config)?;
                }
                _ => return Err(anyhow!("Unknown pipeline step: {}", step_name)),
            }
        }
        Ok(())
    }
}
```

**Key Benefits:**
- **Reuses all existing infrastructure**: OutputManager, config conversion, logging
- **Session efficiency**: Models are loaded once and reused across images
- **Clean composition**: Each step uses the standard ModelProcessor interface
- **Type safety**: Compile-time verification of pipeline step compatibility

### 2.2 Batch Processing Extension

The API naturally extends to support batch inference optimization:

**Batch-Enabled Model:**
```rust
impl ModelProcessor for HeadProcessor {
    // ... standard methods ...

    // Opt-in batch support
    fn supports_batch_inference() -> bool { true }
    fn max_batch_size() -> Option<usize> { Some(8) }

    fn run_batch_inference(session: &mut Session, batch_input: BatchModelInput) -> Result<BatchModelOutput> {
        let batch_tensor = concatenate_tensors(batch_input.tensors)?;
        let outputs = session.run(ort::inputs!["images" => &batch_tensor])?;
        let individual_outputs = split_batch_output(outputs, batch_input.len())?;
        Ok(BatchModelOutput::from_vec(individual_outputs))
    }
}
```

**Enhanced Batch Processing:**
```rust
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;
    let mut session = P::create_session(&config)?;

    if P::supports_batch_inference() && image_files.len() > 1 {
        process_with_batching::<P>(&mut session, &image_files, &config)
    } else {
        process_individually::<P>(&mut session, &image_files, &config)
    }
}
```

**Key Benefits:**
- **Backward compatibility**: Non-batching models work unchanged
- **Opt-in optimization**: Models choose their own batch support strategy
- **Graceful fallback**: Batch failures automatically retry individually
- **Memory management**: Configurable batch sizes prevent resource exhaustion

### 2.3 API Flexibility Assessment

**Strengths:**
1. **Extensible**: New models require minimal implementation (5 methods)
2. **Composable**: Pipeline and batch processing build naturally on the base API
3. **Type-safe**: Compile-time verification of model compatibility
4. **Zero-cost abstractions**: No runtime overhead for unused features
5. **Testing-friendly**: Easy to mock and test individual components

**Potential Concerns:**
1. **Learning curve**: New contributors need to understand the trait system
2. **Indirection**: Some complexity moved from concrete implementations to generic framework
3. **Trait object limitations**: Some operations may require static dispatch

**Mitigation Strategies:**
1. **Comprehensive documentation**: Clear examples for each model type
2. **Template generators**: Scaffolding tools for new model implementation
3. **Gradual migration**: Existing models continue working during transition

## 3. Implementation Plan

### Phase 1: Foundation Layer
**Goal**: Establish core infrastructure without breaking existing functionality

**Tasks:**
1. **Create config layer** (`src/config.rs`):
   - Define `BaseCliArgs` and `BaseModelConfig` structs
   - Implement conversion traits
   - Add comprehensive tests

2. **Extract CLI structures** (in `main.rs`):
   - Move current config structs to CLI command structs
   - Update clap derives and help text
   - Maintain exact same CLI behavior

3. **Update existing models**:
   - Add `From<CliCommand>` implementations
   - Verify no functional changes
   - Update tests to use new config types

4. **Verbosity refactor**: use crate `clap-verbosity-flag` to handle verbosity flag. Then configure logger like
```
let cli = Cli::parse();
env_logger::Builder::new()
    .filter_level(cli.verbosity.log_level_filter())
    .init();
```

Then replace all of our custom verbosity/println stuff with an `env_logger` using the `log` crate. The only tricky bit is ORT using tracing, so we still want to control ORT's tracing level with our verbosity flag (e.g. `if log_enabled!(Level::Info) {`). We won't use tracing though, we will use the `log` crate. That means replacing `println!` with `info!` and similar. Here is example code:

```rust
// simplest log-verbosity support in a Rust CLI, with env_logger & clap‑verbosity‑flag

use clap::Parser;
use clap_verbosity_flag::Verbosity;
use env_logger;
use log::{error, warn, info, debug};

#[derive(Debug, Parser)]
struct Cli {
    /// -q / --quiet            ; -v / -vv / -vvv / -vvvv
    #[command(flatten)]
    verbosity: Verbosity,

    // ... other CLI options ...
}

fn main() {
    let cli = Cli::parse();

    // initialize the global env_logger at the verbosity-set level:
    env_logger::Builder::new()
        .filter_level(cli.verbosity.log_level_filter())
        .init();

    error!("critical failure");       // always shown
    warn!("something questionable");
    info!("informational step");
    debug!("detailed debug trace");
}
```

Note:

•	clap‑verbosity‑flag adds -q/--quiet and repeatable -v flags mapping to error‑only, warn, info, debug, trace levels (each extra -v increases verbosity)  ￼
•	.filter_level(cli.verbosity.log_level_filter()) wires that choice into env_logger so log macros automatically respect the user-specified level  ￼
•	If desired, RUST_LOG env vars still work for overriding or module-level filtering as supported by env_logger  ￼


**Validation**:

- All existing functionality works identically
- No warnings
- No duplicated structs or functions
- Reduction in lines of non-test code by `cargo warloc` (see success metrics below) in each model, some increase in shared code (but still net reduction overall)
- Better verbosity handling, supporting `-vvv` for example
- No need to thread `verbose` everywhere, instead using `log_enabled!` checks

### Phase 2: Output Management
**Goal**: Unify all path generation logic

**Tasks:**
1. **Create OutputManager** (`src/output_manager.rs`):
   - Extract path generation logic from both models
   - Implement numbered output support
   - Add metadata path utilities

2. **Refactor existing models**:
   - Replace inline path generation with OutputManager calls
   - Ensure identical output paths for existing use cases
   - Update tests

3. **Add comprehensive path generation tests**:
   - Test all suffix combinations
   - Test output directory vs same directory behavior
   - Test numbered output patterns

**Validation**: All output paths remain identical to current behavior

### Phase 3: Model Processor Framework
**Goal**: Implement the core ModelProcessor trait and generic processing

**Tasks:**
1. **Define ModelProcessor trait** (`src/model_processing.rs`):
   - Create trait with required and provided methods
   - Implement generic `run_model_processing` function
   - Add session management utilities

2. **Implement trait for existing models**:
   - Break down current processing into trait methods
   - Migrate HeadProcessor first, then CutoutProcessor
   - Ensure identical behavior

3. **Update main.rs**:
   - Replace direct model calls with generic processor calls
   - Simplify command handling logic

**Validation**: All models work through the generic interface

### Phase 4: Logging Unification
**Goal**: Centralize all logging and error handling

**Tasks:**
1. **Create InferenceLogger** (`src/logging.rs`):
   - Extract logging patterns from both models
   - Standardize timing and progress reporting
   - Implement model-specific context

2. **Update models to use centralized logging**:
   - Remove verbose_println! macros
   - Use InferenceLogger for all output
   - Maintain exact same log output

3. **Enhance error handling**:
   - Standardize error messages
   - Improve error context and debugging information

**Validation**: Log output remains identical with improved consistency

### Phase 5: Advanced Features
**Goal**: Implement pipeline and batch processing capabilities

**Tasks:**
1. **Implement pipeline support**:
   - Create PipelineProcessor and PipelineSessionManager
   - Add CLI command for pipeline
   - Test cutout→head pipeline

2. **Add batch processing framework**:
   - Extend ModelProcessor trait with batch methods
   - Implement batch processing in run_model_processing
   - Add batch support to head detection model

3. **Performance validation**:
   - Benchmark new vs old implementations
   - Verify no performance regressions
   - Optimize critical paths if needed

**Validation**: New features work correctly, no performance regressions

### Phase 6: Documentation and Cleanup
**Goal**: Finalize the implementation with comprehensive documentation

**Tasks:**
1. **Create model implementation guide**:
   - Step-by-step tutorial for adding new models
   - Code templates and examples
   - Best practices documentation

2. **Performance optimization**:
   - Profile critical paths
   - Optimize hot code sections
   - Add performance tests to CI

3. **Final cleanup**:
   - Remove any dead code from migration
   - Ensure consistent error messages
   - Update all documentation

**Validation**: Complete, documented, and optimized unified API

### Risk Mitigation

**Incremental Validation**: Each phase includes comprehensive testing to ensure no regressions
**Feature Flags**: Advanced features can be disabled if issues arise
**Rollback Plan**: Each phase can be reverted independently if needed
**Performance Monitoring**: Continuous benchmarking throughout implementation

### Success Metrics

#### Current Codebase Analysis (Lines of Code)

Counted by `cargo warloc --by-file`.

**Model-Specific Code**:
- **Head Detection**: 498 lines (`head_detection.rs`) + 51 lines (`yolo_preprocessing.rs`) + 95 lines (`yolo_postprocessing.rs`) = **644 lines**
- **Cutout Processing**: 262 lines (`cutout_processing.rs`) + 36 lines (`cutout_preprocessing.rs`) + 144 lines (`cutout_postprocessing.rs`) = **442 lines**
- **Total Model-Specific**: **1,086 lines**

**Shared Infrastructure**:
- **Session Management**: 185 lines (`onnx_session.rs`)
- **Input Processing**: 126 lines (`image_input.rs`)
- **Model Caching**: 96 lines (`model_cache.rs`)
- **Metadata Handling**: 47 lines (`shared_metadata.rs`)
- **CLI & Main**: 231 lines (`main.rs`)
- **Build System**: 150 lines (`build.rs`)
- **Library Root**: 7 lines (`lib.rs`)
- **Total Shared Infrastructure**: **842 lines**

**Test Code**:
- **Integration Tests**: 1,248 lines (`integration_tests.rs`)
- **Unit Tests**: 99 lines (`image_input.rs`) + 25 lines (`model_cache.rs`) + 27 lines (`shared_metadata.rs`) + 30 lines (`cutout_postprocessing.rs`) + 14 lines (`cutout_preprocessing.rs`) = **195 lines**
- **Total Test Code**: **1,443 lines**

**Current Totals**:
- **Production Code**: 1,928 lines (1,086 model-specific + 842 shared)
- **Test Code**: 1,443 lines
- **Grand Total**: 3,371 lines

#### Target Metrics After Unification

**Code Quality**:
- **40% reduction in model-specific code**: From 1,086 to ~650 lines (save ~436 lines)
- **30% increase in shared infrastructure**: From 842 to ~1,095 lines (add ~253 lines for abstractions)
- **Net 10% reduction in production code**: From 1,928 to ~1,745 lines (save ~183 lines)
- **70% reduction in effort to add new models**: New models need only ~5 trait methods vs full implementation
- **Zero functional regressions**: All existing behavior preserved

**Maintainability**:
- Centralized error handling and logging
- Consistent patterns across all models
- Comprehensive test coverage maintained at 1,443+ lines

**Performance**:
- No regressions in processing speed
- Improved batch processing efficiency
- Reduced memory usage through better session management
