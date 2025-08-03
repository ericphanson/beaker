# Beaker DRY Analysis: Opportunities for Code Simplification

## Executive Summary

The beaker codebase contains significant duplication between the `head` and `cutout` models, particularly in processing pipelines, configuration management, and output handling. This analysis identifies specific areas where abstraction can reduce redundancy and make adding new models substantially easier.

## Current Architecture Assessment

### What's Already Well-Abstracted ✅

1. **Image Input Processing** (`image_input.rs`)
   - Successfully unified glob pattern handling, directory traversal, and file validation
   - Clean separation of strict vs permissive modes
   - Good abstraction that both models use identically

2. **ONNX Session Management** (`onnx_session.rs`)
   - Unified device selection logic (auto/cpu/coreml)
   - Consistent session creation with proper provider selection
   - Shared verbose output trait for different config types

3. **Shared Metadata System** (`shared_metadata.rs`)
   - Common TOML-based metadata structure
   - Path generation utilities work for both models

## Major Duplication Areas Requiring Abstraction

### 1. Processing Pipeline Structure 🔄

**Current State**: Both models follow an identical high-level pattern but implement it separately:

```rust
// Pattern repeated in both head_detection.rs and cutout_processing.rs
pub fn run_MODEL_processing(config: ModelConfig) -> Result<usize> {
    // 1. Collect images using image_input
    // 2. Download/load model (different approaches)
    // 3. Determine device and create session
    // 4. Process each image sequentially
    // 5. Handle outputs and metadata
    // 6. Print summary statistics
}
```

**Opportunity**: Create a generic `ModelProcessor` trait and shared processing framework.

### 2. Configuration Management 📋

**Duplication**:
- Both `HeadDetectionConfig` and `CutoutConfig` have identical core fields:
  ```rust
  pub sources: Vec<String>,
  pub device: String,
  pub output_dir: Option<String>,
  pub skip_metadata: bool,
  pub verbose: bool,
  pub strict: bool,
  ```
- Both implement `VerboseOutput` identically
- Both have the same validation logic

**Current Architecture Challenge**:
The CLI structs (defined with `clap` derives) and internal config structs serve different purposes but contain overlapping data. This creates a question: should we unify them or keep them separate?

**Two Approaches Analyzed:**

**Approach A: Unified CLI/Config Structs**
```rust
// Single struct serves both CLI parsing and internal config
#[derive(Parser, Clone)]
pub struct HeadCommand {
    #[serde(flatten)]
    #[command(flatten)]
    pub base: BaseCliConfig,

    #[arg(short, long, default_value = "0.25")]
    pub confidence: f32,

    #[arg(long)]
    pub crop: bool,
    // ... other head-specific options
}

impl ModelConfig for HeadCommand {
    fn base(&self) -> &BaseModelConfig { &self.base.into() }
    fn model_name(&self) -> &'static str { "head" }
}

// Usage in main.rs
match cli.command {
    Commands::Head(head_config) => {
        run_model_processing::<HeadProcessor>(head_config)?;
    }
}
```

**Approach B: Separate CLI and Config Structs**
```rust
// CLI struct (what user provides)
#[derive(Parser)]
pub struct HeadCommand {
    #[command(flatten)]
    pub base: BaseCliArgs,

    #[arg(short, long, default_value = "0.25")]
    pub confidence: f32,

    #[arg(long)]
    pub crop: bool,
}

// Internal config struct (what models use)
#[derive(Clone)]
pub struct HeadDetectionConfig {
    pub base: BaseModelConfig,
    pub confidence: f32,
    pub crop: bool,
    // ... potentially additional derived/computed fields
}

impl From<HeadCommand> for HeadDetectionConfig {
    fn from(cmd: HeadCommand) -> Self {
        Self {
            base: cmd.base.into(),
            confidence: cmd.confidence,
            crop: cmd.crop,
        }
    }
}

// Usage in main.rs
match cli.command {
    Commands::Head(head_cmd) => {
        let config = HeadDetectionConfig::from(head_cmd);
        run_model_processing::<HeadProcessor>(config)?;
    }
}
```

**Recommendation: Approach B (Separate Structs)**

**Rationale:**
1. **Separation of Concerns**: CLI structs handle user interface concerns (help text, validation, defaults), while config structs handle business logic
2. **Evolution Flexibility**: Internal configs can add computed fields, caching, or validation without affecting CLI
3. **Testing**: Easier to construct test configs without CLI parsing machinery
4. **Pipeline Support**: Pipeline can create configs programmatically without CLI involvement
5. **Type Safety**: Can use different types internally (e.g., parsed `PathBuf` vs CLI `String`)

**Opportunity**: Extract a `BaseCliArgs` and `BaseModelConfig` with clean conversion between them.

### 3. Output Path Generation 📁

**Duplication**:
- `cutout_processing.rs` has `generate_output_path_with_suffix_control()` (35 lines)
- `head_detection.rs` has inline path generation logic scattered throughout `handle_image_outputs()` (100+ lines)
- Both handle the same patterns:
  - Single vs multiple outputs
  - Output directory vs same directory as input
  - Suffix handling for auxiliary files

**Opportunity**: Create a unified `OutputPathGenerator` that handles all path generation patterns.

### 4. Verbose Logging Patterns 📝

**Duplication**:
- Both use identical `verbose_println!` macros
- Same logging patterns for timing, progress, and results
- Duplicate summary statistics printing

**Opportunity**: Centralized logging utilities with consistent formatting.

### 5. Error Handling and Processing Loop 🔁

**Duplication**:
- Identical image processing loops in both `run_*` functions
- Same error handling: log error, continue with next image
- Same pattern for collecting results and timing

**Opportunity**: Generic processing loop that takes a processing closure.

## Proposed Abstractions

### 1. Base Model Processing Framework

```rust
// New file: src/model_processing.rs
pub trait ModelProcessor {
    type Config: ModelConfig;
    type Result: ModelResult;

    // Model lifecycle - each model implements this
    fn create_session(config: &Self::Config) -> Result<Session>;

    // Core processing pipeline - each model implements these
    fn preprocess_image(image: &DynamicImage, config: &Self::Config) -> Result<ModelInput>;
    fn run_inference(session: &mut Session, input: ModelInput) -> Result<ModelOutput>;
    fn postprocess_output(output: ModelOutput, config: &Self::Config) -> Result<Self::Result>;
    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()>;

    // Shared implementations - models get these for free
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config
    ) -> Result<Self::Result> {
        let image = image::open(image_path)?;
        let input = Self::preprocess_image(&image, config)?;
        let output = Self::run_inference(session, input)?;
        let result = Self::postprocess_output(output, config)?;
        Self::handle_outputs(&result, image_path, config)?;
        Ok(result)
    }
}

// Generic batch processing - works for any model
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    // 1. Collect images using image_input
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    // 2. Create session
    let mut session = P::create_session(&config)?;

    // 3. Process each image using the shared single-image method
    let mut total_results = 0;
    for image_path in &image_files {
        match P::process_single_image(&mut session, image_path, &config) {
            Ok(_result) => total_results += 1,
            Err(e) => {
                if config.base().verbose {
                    eprintln!("❌ Failed to process {}: {}", image_path.display(), e);
                }
                // Continue with other images
            }
        }
    }

    Ok(total_results)
}
```

### 2. Unified Configuration System

```rust
// Enhanced shared_metadata.rs or new config.rs

// CLI-level base arguments (what clap parses)
#[derive(Parser)]
pub struct BaseCliArgs {
    /// Path(s) to input images or directories
    #[arg(value_name = "IMAGES_OR_DIRS", required = true)]
    pub sources: Vec<String>,

    /// Device to use for inference (auto, cpu, coreml)
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Global output directory
    #[arg(long, global = true)]
    pub output_dir: Option<String>,

    /// Skip creating metadata output files
    #[arg(long, global = true)]
    pub no_metadata: bool,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Use permissive mode for input validation
    #[arg(long, global = true)]
    pub permissive: bool,
}

// Internal config (what models actually use)
pub struct BaseModelConfig {
    pub sources: Vec<String>,
    pub device: String,
    pub output_dir: Option<String>,
    pub skip_metadata: bool,
    pub verbose: bool,
    pub strict: bool,
}

impl From<BaseCliArgs> for BaseModelConfig {
    fn from(cli: BaseCliArgs) -> Self {
        Self {
            sources: cli.sources,
            device: cli.device,
            output_dir: cli.output_dir,
            skip_metadata: cli.no_metadata,
            verbose: cli.verbose,
            strict: !cli.permissive, // Note: inverted logic
        }
    }
}

pub trait ModelConfig: VerboseOutput {
    fn base(&self) -> &BaseModelConfig;
    fn model_name(&self) -> &'static str;
}

// CLI command structs
#[derive(Parser)]
pub struct HeadCommand {
    #[command(flatten)]
    pub base: BaseCliArgs,

    #[arg(short, long, default_value = "0.25")]
    pub confidence: f32,

    #[arg(long, default_value = "0.45")]
    pub iou_threshold: f32,

    #[arg(long)]
    pub crop: bool,

    #[arg(long)]
    pub bounding_box: bool,
}

#[derive(Parser)]
pub struct CutoutCommand {
    #[command(flatten)]
    pub base: BaseCliArgs,

    #[arg(long)]
    pub post_process: bool,

    #[arg(long)]
    pub alpha_matting: bool,

    // ... other cutout-specific fields
}

// Internal config structs
pub struct HeadDetectionConfig {
    pub base: BaseModelConfig,
    pub confidence: f32,
    pub iou_threshold: f32,
    pub crop: bool,
    pub bounding_box: bool,
}

pub struct CutoutConfig {
    pub base: BaseModelConfig,
    pub post_process_mask: bool,
    pub alpha_matting: bool,
    // ... other cutout-specific fields
}

impl From<HeadCommand> for HeadDetectionConfig {
    fn from(cmd: HeadCommand) -> Self {
        Self {
            base: cmd.base.into(),
            confidence: cmd.confidence,
            iou_threshold: cmd.iou_threshold,
            crop: cmd.crop,
            bounding_box: cmd.bounding_box,
        }
    }
}

impl From<CutoutCommand> for CutoutConfig {
    fn from(cmd: CutoutCommand) -> Self {
        Self {
            base: cmd.base.into(),
            post_process_mask: cmd.post_process,
            alpha_matting: cmd.alpha_matting,
            // ... other field mappings
        }
    }
}

impl ModelConfig for HeadDetectionConfig {
    fn base(&self) -> &BaseModelConfig { &self.base }
    fn model_name(&self) -> &'static str { "head" }
}

impl ModelConfig for CutoutConfig {
    fn base(&self) -> &BaseModelConfig { &self.base }
    fn model_name(&self) -> &'static str { "cutout" }
}
```

### 3. Unified Output Management

```rust
// New file: src/output_manager.rs
pub struct OutputManager<'a> {
    config: &'a dyn ModelConfig,
    input_path: &'a Path,
    metadata_path: Option<PathBuf>,
}

impl<'a> OutputManager<'a> {
    pub fn new(config: &'a dyn ModelConfig, input_path: &'a Path) -> Result<Self>;

    // Core path generation - models specify exactly what they want
    pub fn generate_output_path(&self, suffix: Option<&str>, extension: &str) -> Result<PathBuf>;

    // For multiple similar outputs (e.g., multiple head crops)
    pub fn generate_numbered_output(&self, base_suffix: &str, index: usize, total: usize, extension: &str) -> Result<PathBuf>;

    // Path utilities for metadata
    pub fn make_relative_to_metadata(&self, path: &Path) -> Result<String>;
    pub fn get_metadata_path(&self) -> Result<PathBuf>;

    // Metadata management
    pub fn save_metadata<T: Serialize>(&self, data: T, section_name: &str) -> Result<()>;
}
```

### 4. Unified Inference Logging

```rust
// Enhanced onnx_session.rs or new logging.rs
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

## Implementation Strategy

### Phase 1: Extract Base Configuration
1. Create `BaseModelConfig` struct
2. Update both existing configs to use it via composition
3. Update main.rs CLI parsing to use the new structure
4. Verify no functional changes

### Phase 2: Unify Output Path Generation
1. Extract all path generation logic into `OutputManager`
2. Replace inline path generation in both models
3. Add comprehensive tests for path generation patterns

### Phase 3: Create Generic Processing Framework
1. Define `ModelProcessor` trait
2. Implement trait for both existing models
3. Create generic `run_model_processing` function
4. Migrate both models to use the new framework

### Phase 4: Unify Logging and Error Handling
1. Extract logging utilities into `InferenceLogger`
2. Standardize error handling patterns
3. Consolidate timing and statistics reporting

## Benefits for Adding New Models

With these abstractions, adding a new model would require:

1. **Define model-specific configuration** (extends `BaseModelConfig`)
2. **Implement `ModelProcessor` trait** (~100 lines):
   - `load_model()` - model loading logic
   - `preprocess_image()` - input preprocessing
   - `run_inference()` - usually just pass-through
   - `postprocess_output()` - output processing
   - `handle_outputs()` - file saving and output generation
3. **Add CLI command** in main.rs (~20 lines)
4. **Model-specific modules** for preprocessing/postprocessing

**Current effort**: ~800-1000 lines of code per model
**With abstractions**: ~200-300 lines of code per model

## Code Reduction Estimate

- **cutout_processing.rs**: 344 lines → ~150 lines (56% reduction)
- **head_detection.rs**: 641 lines → ~200 lines (69% reduction)
- **Total new shared code**: ~300 lines
- **Net reduction**: ~635 lines (42% reduction)

## Risk Mitigation

1. **Backward Compatibility**: All abstractions designed to maintain exact same CLI behavior
2. **Incremental Migration**: Each phase can be implemented and tested independently
3. **Performance**: No performance impact - abstractions are zero-cost at runtime
4. **Complexity**: While adding some indirection, the patterns are common Rust idioms

## Conclusion

The current duplication creates maintenance burden and high barriers to adding new models. The proposed abstractions would:

- **Reduce codebase size by ~40%**
- **Cut new model implementation effort by 70%**
- **Improve maintainability through centralized logic**
- **Preserve all existing functionality and performance**

The abstractions follow common Rust patterns (traits, generics, composition) and would make the codebase more idiomatic while significantly reducing the complexity of adding new models.

## Stress Test: Pipeline Subcommand

To validate these abstractions, let's consider implementing a more complex use case: a `pipeline` subcommand that runs multiple models in sequence.

### Requirements
```bash
beaker pipeline --steps cutout,head --output-dir out_dir path/*.jpg
```

This should:
1. Run cutout processing on each image first
2. Run head detection on the cutout results
3. Handle shared configuration (device, output-dir, metadata)
4. Allow step-specific overrides

### Implementation Analysis

**CLI Configuration:**
```rust
pub struct PipelineConfig {
    #[serde(flatten)]
    pub base: BaseModelConfig,
    pub steps: Vec<String>,
    pub step_overrides: HashMap<String, serde_json::Value>, // For step-specific options
}

impl ModelConfig for PipelineConfig {
    fn base(&self) -> &BaseModelConfig { &self.base }
    fn model_name(&self) -> &'static str { "pipeline" }
}
```

**Pipeline Processor Implementation:**
```rust
pub struct PipelineProcessor;

impl ModelProcessor for PipelineProcessor {
    type Config = PipelineConfig;
    type Result = PipelineResult;

    fn load_model(config: &Self::Config) -> Result<ModelData> {
        // Pipeline doesn't load its own model - delegates to steps
        Ok(ModelData::None)
    }

    fn preprocess_image(image: &DynamicImage, config: &Self::Config) -> Result<ModelInput> {
        // No preprocessing at pipeline level
        Ok(ModelInput::PassThrough(image.clone()))
    }

    fn run_inference(session: &mut Session, input: ModelInput) -> Result<ModelOutput> {
        // Pipeline doesn't run inference directly
        Ok(ModelOutput::PassThrough)
    }

    fn postprocess_output(output: ModelOutput, config: &Self::Config) -> Result<Self::Result> {
        // No postprocessing at pipeline level
        Ok(PipelineResult::default())
    }

    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()> {
        // This is where the magic happens - run each step in sequence
        let mut current_input = image_path.to_path_buf();

        for step_name in &config.steps {
            match step_name.as_str() {
                "cutout" => {
                    let step_config = create_cutout_config_from_pipeline(config, &current_input)?;
                    let step_result = run_single_step::<CutoutProcessor>(step_config)?;
                    current_input = step_result.primary_output_path;
                }
                "head" => {
                    let step_config = create_head_config_from_pipeline(config, &current_input)?;
                    let _step_result = run_single_step::<HeadProcessor>(step_config)?;
                    // Head detection doesn't change the input for next step
                }
                _ => return Err(anyhow::anyhow!("Unknown pipeline step: {}", step_name)),
            }
        }

        Ok(())
    }
}

// Helper function to run a single step without the full batch processing
fn run_single_step<P: ModelProcessor>(config: P::Config) -> Result<P::Result> {
    // Load model, create session, process single image
    // This is like run_model_processing but for a single image
}
```

### Stress Test Results

**✅ What Works Well:**

1. **BaseModelConfig Reuse**: Pipeline can inherit all the common CLI options (device, output-dir, verbose, etc.) through the base config pattern.

2. **OutputManager Flexibility**: Each step can use OutputManager independently:
   ```rust
   // Cutout step creates its output
   let output_manager = OutputManager::new(&cutout_config, current_input)?;
   let cutout_path = output_manager.generate_output_path(Some("cutout"), "png")?;

   // Head step uses cutout output as input
   let head_output_manager = OutputManager::new(&head_config, &cutout_path)?;
   ```

3. **ModelProcessor Trait**: The trait is flexible enough to handle both regular models and meta-models like pipeline.

4. **InferenceLogger**: Can be used by each step independently for consistent logging.

**⚠️ Areas Needing Refinement:**

1. **Single Image Processing**: The current `run_model_processing` is designed for batch processing. We need a `process_single_image` variant that individual pipeline steps can use.

2. **Model Loading Efficiency**: Pipeline steps shouldn't reload models for each image. We need session management that persists across steps.

3. **Result Communication**: Steps need to communicate their outputs to the next step. The `ModelProcessor` trait might need a way to return output paths.

### Proposed Enhancements

**Enhanced ModelProcessor Trait:**
```rust
pub trait ModelProcessor {
    type Config: ModelConfig;
    type Result: ModelResult;

    // Model lifecycle
    fn create_session(config: &Self::Config) -> Result<Session>;

    // Core processing pipeline (used by both batch and single-image processing)
    fn preprocess_image(image: &DynamicImage, config: &Self::Config) -> Result<ModelInput>;
    fn run_inference(session: &mut Session, input: ModelInput) -> Result<ModelOutput>;
    fn postprocess_output(output: ModelOutput, config: &Self::Config) -> Result<Self::Result>;
    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()>;

    // Convenience method that combines the pipeline steps for single images
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config
    ) -> Result<Self::Result> {
        let image = image::open(image_path)?;
        let input = Self::preprocess_image(&image, config)?;
        let output = Self::run_inference(session, input)?;
        let result = Self::postprocess_output(output, config)?;
        Self::handle_outputs(&result, image_path, config)?;
        Ok(result)
    }
}
```

**Session Manager for Pipeline:**
```rust
pub struct PipelineSessionManager {
    sessions: HashMap<String, Session>,
}

impl PipelineSessionManager {
    pub fn get_or_create_session<P: ModelProcessor>(
        &mut self,
        step_name: &str,
        config: &P::Config
    ) -> Result<&mut Session> {
        if !self.sessions.contains_key(step_name) {
            let session = P::create_session(config)?;
            self.sessions.insert(step_name.to_string(), session);
        }
        Ok(self.sessions.get_mut(step_name).unwrap())
    }
}
```

### Conclusion

The proposed abstractions handle the pipeline use case well with minor enhancements:

1. **Core abstractions remain sound**: BaseModelConfig, OutputManager, and ModelConfig all work as designed
2. **Minor trait additions needed**: Single-image processing and session management methods
3. **New utility needed**: Session manager for persistent model loading across pipeline steps
4. **Pattern scales well**: Adding new pipeline steps becomes trivial once the framework exists

This stress test validates that the abstractions are flexible enough for complex use cases while maintaining the simplicity benefits for basic model implementation.

## Stress Test: Batch Inference Support

Another important use case to validate is support for models that can efficiently process multiple images in a single inference call through dynamic batch dimensions.

### Requirements

Some ONNX models support dynamic batch sizes where the input shape is `[batch_size, channels, height, width]` instead of `[1, channels, height, width]`. For these models, we could:

1. Preprocess all images individually
2. Concatenate preprocessed tensors into a single batch
3. Run one inference call on the entire batch
4. Split the batch output back into individual results
5. Postprocess each result individually

This can be significantly more efficient for GPU inference, especially with larger batch sizes.

### Implementation Analysis

**Enhanced ModelProcessor Trait:**
```rust
pub trait ModelProcessor {
    type Config: ModelConfig;
    type Result: ModelResult;

    // Existing methods...
    fn create_session(config: &Self::Config) -> Result<Session>;
    fn preprocess_image(image: &DynamicImage, config: &Self::Config) -> Result<ModelInput>;
    fn run_inference(session: &mut Session, input: ModelInput) -> Result<ModelOutput>;
    fn postprocess_output(output: ModelOutput, config: &Self::Config) -> Result<Self::Result>;
    fn handle_outputs(result: &Self::Result, image_path: &Path, config: &Self::Config) -> Result<()>;

    // Batch processing support - models can opt into this
    fn supports_batch_inference() -> bool { false }
    fn max_batch_size() -> Option<usize> { None }

    fn run_batch_inference(session: &mut Session, batch_input: BatchModelInput) -> Result<BatchModelOutput> {
        // Default implementation: process each item individually
        let mut outputs = Vec::new();
        for input in batch_input.into_iter() {
            let output = Self::run_inference(session, input)?;
            outputs.push(output);
        }
        Ok(BatchModelOutput::from_vec(outputs))
    }

    // Single image processing (unchanged)
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config
    ) -> Result<Self::Result> {
        let image = image::open(image_path)?;
        let input = Self::preprocess_image(&image, config)?;
        let output = Self::run_inference(session, input)?;
        let result = Self::postprocess_output(output, config)?;
        Self::handle_outputs(&result, image_path, config)?;
        Ok(result)
    }
}
```

**Enhanced Batch Processing Function:**
```rust
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    let mut session = P::create_session(&config)?;
    let mut total_results = 0;

    if P::supports_batch_inference() && image_files.len() > 1 {
        // Use batch inference
        total_results += process_with_batching::<P>(&mut session, &image_files, &config)?;
    } else {
        // Use single-image processing
        for image_path in &image_files {
            match P::process_single_image(&mut session, image_path, &config) {
                Ok(_) => total_results += 1,
                Err(e) => {
                    if config.base().verbose {
                        eprintln!("❌ Failed to process {}: {}", image_path.display(), e);
                    }
                }
            }
        }
    }

    Ok(total_results)
}

fn process_with_batching<P: ModelProcessor>(
    session: &mut Session,
    image_files: &[PathBuf],
    config: &P::Config,
) -> Result<usize> {
    let batch_size = P::max_batch_size().unwrap_or(8);
    let mut total_results = 0;

    for batch_chunk in image_files.chunks(batch_size) {
        // 1. Preprocess all images in the batch
        let mut batch_data = Vec::new();
        let mut valid_paths = Vec::new();

        for image_path in batch_chunk {
            match image::open(image_path) {
                Ok(image) => {
                    match P::preprocess_image(&image, config) {
                        Ok(input) => {
                            batch_data.push(input);
                            valid_paths.push(image_path);
                        }
                        Err(e) => {
                            if config.base().verbose {
                                eprintln!("❌ Preprocessing failed for {}: {}", image_path.display(), e);
                            }
                        }
                    }
                }
                Err(e) => {
                    if config.base().verbose {
                        eprintln!("❌ Failed to load {}: {}", image_path.display(), e);
                    }
                }
            }
        }

        if batch_data.is_empty() {
            continue;
        }

        // 2. Run batch inference
        let batch_input = BatchModelInput::from_vec(batch_data);
        match P::run_batch_inference(session, batch_input) {
            Ok(batch_output) => {
                // 3. Postprocess each result individually
                for (output, image_path) in batch_output.into_iter().zip(valid_paths.iter()) {
                    match P::postprocess_output(output, config) {
                        Ok(result) => {
                            if let Err(e) = P::handle_outputs(&result, image_path, config) {
                                if config.base().verbose {
                                    eprintln!("❌ Output handling failed for {}: {}", image_path.display(), e);
                                }
                            } else {
                                total_results += 1;
                            }
                        }
                        Err(e) => {
                            if config.base().verbose {
                                eprintln!("❌ Postprocessing failed for {}: {}", image_path.display(), e);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                if config.base().verbose {
                    eprintln!("❌ Batch inference failed: {}", e);
                }
                // Fallback: try processing each image individually
                for image_path in valid_paths {
                    match P::process_single_image(session, image_path, config) {
                        Ok(_) => total_results += 1,
                        Err(e) => {
                            if config.base().verbose {
                                eprintln!("❌ Fallback processing failed for {}: {}", image_path.display(), e);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(total_results)
}
```

**Example: Batch-Enabled Head Detection:**
```rust
impl ModelProcessor for HeadProcessor {
    // ... existing methods ...

    fn supports_batch_inference() -> bool {
        true // YOLO models typically support dynamic batch sizes
    }

    fn max_batch_size() -> Option<usize> {
        Some(8) // Reasonable batch size for head detection
    }

    fn run_batch_inference(session: &mut Session, batch_input: BatchModelInput) -> Result<BatchModelOutput> {
        // Concatenate all preprocessed tensors into a single batch tensor
        let batch_tensor = concatenate_tensors(batch_input.tensors)?;

        // Run single inference on the batch
        let outputs = session.run(ort::inputs!["images" => &batch_tensor])?;

        // Split batch output back into individual outputs
        let individual_outputs = split_batch_output(outputs, batch_input.len())?;

        Ok(BatchModelOutput::from_vec(individual_outputs))
    }
}
```

### Stress Test Results

**✅ What Works Well:**

1. **Backward Compatibility**: Models that don't support batching continue to work exactly as before by using the default `run_batch_inference` implementation.

2. **Opt-in Design**: Models can choose to support batching by overriding a few methods and returning `true` from `supports_batch_inference()`.

3. **Flexible Batch Sizes**: Models can specify their optimal batch size, and the framework handles chunking automatically.

4. **Graceful Fallback**: If batch inference fails, the system automatically falls back to single-image processing.

5. **Pipeline Compatibility**: Pipeline processing can still use single-image methods when needed, while batch processing gets efficiency gains for the main `run_model_processing` flow.

**⚠️ Design Considerations:**

1. **Memory Management**: Large batches could use significant GPU memory. The `max_batch_size()` method allows models to set reasonable limits.

2. **Error Handling**: If one image in a batch fails preprocessing, we still process the rest. If batch inference fails entirely, we fall back to individual processing.

3. **Output Alignment**: The framework must carefully align batch outputs with their corresponding input images for proper metadata generation.

### Implementation Complexity

Adding batch support requires:

1. **New Types**: `BatchModelInput` and `BatchModelOutput` wrapper types
2. **Tensor Operations**: Helper functions for concatenating and splitting tensors
3. **Enhanced Processing Loop**: Logic to decide between batch and single-image processing
4. **Model Updates**: Optional implementation of batch methods for models that support it

**Estimated Implementation Effort:**
- Core batching framework: ~150 lines
- Helper utilities: ~100 lines
- Model-specific batch implementations: ~50 lines each (optional)

### Conclusion

The proposed abstractions handle batch inference naturally:

1. **Models remain simple**: Non-batching models require no changes
2. **Efficiency gains available**: Models that benefit from batching can opt in easily
3. **Robust error handling**: Multiple fallback strategies ensure reliability
4. **Memory conscious**: Configurable batch sizes prevent resource exhaustion

This validates that the `ModelProcessor` trait is flexible enough to accommodate advanced optimization techniques while maintaining simplicity for basic use cases.
