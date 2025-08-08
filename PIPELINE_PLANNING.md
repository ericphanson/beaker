# Beaker Pipeline Subcommand: Design and Implementation Plan

## Executive Summary

This document outlines the design and implementation plan for the `beaker pipeline` subcommand, which will enable efficient sequential processing of multiple computer vision models on image datasets. The pipeline subcommand aims to provide both ergonomic convenience and significant performance improvements through optimized memory management and session reuse.

## Problem Statement

Currently, users must run separate beaker commands to chain operations:
```bash
beaker cutout *.jpg --output-dir cutout_results
beaker head cutout_results/*.png --crop --output-dir final_results
```

This approach has several limitations:
- **Inefficient I/O**: Intermediate results written to disk unnecessarily
- **Session overhead**: Each command creates new ONNX sessions (expensive)
- **Memory waste**: No optimization across model boundaries
- **User friction**: Complex multi-step workflows require scripting

The proposed pipeline subcommand addresses these issues:
```bash
beaker pipeline --steps cutout,head --output-dir results --crop *.jpg
```

## Current Architecture Analysis

### Strengths to Leverage

1. **Robust Model Framework**: The `ModelProcessor` trait provides a clean, extensible interface
2. **Flexible Configuration**: `BaseModelConfig` + model-specific configs work well
3. **Comprehensive Output Management**: `OutputManager` handles complex path logic elegantly
4. **Session Management**: Existing ONNX session creation and device selection
5. **Progress & Metadata**: Built-in progress bars and comprehensive metadata tracking
6. **Testing Infrastructure**: Solid test framework for validation

### Current Limitations

1. **File-Only Data Flow**: Models only support file I/O, no in-memory passing
2. **No Batch Optimization**: Each model processes images individually
3. **Fixed Input/Output**: Models expect specific input formats
4. **No Pipeline Orchestration**: No coordination between models

## Proposed Interface Design

### Command Syntax

```bash
beaker pipeline [OPTIONS] --steps <STEPS> <IMAGES_OR_DIRS>...
```

### Core Options

```
--steps <STEPS>
    Comma-separated list of processing steps (e.g., "cutout,head")

--output-dir <OUTPUT_DIR>
    Global output directory for final results

--save-intermediates
    Save outputs from intermediate steps (default: false)

--batch-size <SIZE>
    Override automatic batch size calculation

--memory-limit <MB>
    Memory usage limit for optimization (default: auto-detect)
```

### Step-Specific Overrides

Support syntax for step-specific options:
```bash
beaker pipeline --steps cutout,head --crop \
    cutout:--alpha-matting cutout:--post-process \
    head:--confidence=0.8 \
    *.jpg
```

### Global Options (Inherited)

All existing global options apply:
- `--device`: Inference device (shared across all steps)
- `--metadata`: Generate comprehensive pipeline metadata
- `--verbose`/`--quiet`: Logging control
- `--permissive`: Input validation mode

## Architecture Design

### 1. Pipeline Configuration System

```rust
#[derive(Parser, Debug, Clone)]
pub struct PipelineCommand {
    pub sources: Vec<String>,

    #[arg(long, value_delimiter = ',')]
    pub steps: Vec<String>,

    #[arg(long)]
    pub save_intermediates: bool,

    #[arg(long)]
    pub batch_size: Option<usize>,

    #[arg(long)]
    pub memory_limit: Option<usize>,

    // Step-specific overrides parsed from remaining args
    pub step_overrides: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub base: BaseModelConfig,
    pub steps: Vec<PipelineStep>,
    pub save_intermediates: bool,
    pub batch_size_strategy: BatchSizeStrategy,
    pub memory_limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub name: String,
    pub config: Box<dyn ModelConfig>,
    pub save_outputs: bool,
}
```

### 2. Pipeline Data Flow

```rust
/// Represents data flowing between pipeline stages
#[derive(Debug)]
pub struct PipelineData {
    pub image: DynamicImage,
    pub metadata: PipelineMetadata,
    pub source_path: PathBuf,
}

/// Tracks data transformations through pipeline
#[derive(Debug)]
pub struct PipelineMetadata {
    pub original_path: PathBuf,
    pub transformations: Vec<TransformationRecord>,
    pub current_stage: String,
}

/// Interface for pipeline-aware model processing
pub trait PipelineModelProcessor: ModelProcessor {
    /// Process data in-memory (primary method for pipelines)
    fn process_pipeline_data(
        session: &mut Session,
        data: PipelineData,
        config: &Self::Config,
    ) -> Result<PipelineData>;

    /// Check if this model can accept the output from previous model
    fn accepts_input_from(previous_model: &str) -> bool;

    /// Get memory requirements for batch size calculation
    fn memory_requirements(config: &Self::Config) -> MemoryRequirements;
}
```

### 3. Pipeline Orchestrator

```rust
pub struct PipelineProcessor {
    steps: Vec<PipelineStepProcessor>,
    config: PipelineConfig,
    batch_manager: BatchManager,
}

struct PipelineStepProcessor {
    name: String,
    session: Option<Session>,
    processor: Box<dyn PipelineModelProcessor>,
    config: Box<dyn ModelConfig>,
}

impl PipelineProcessor {
    /// Main entry point for pipeline processing
    pub fn process_pipeline(&mut self) -> Result<PipelineResult> {
        let batches = self.batch_manager.create_batches(&self.config.base.sources)?;

        for batch in batches {
            self.process_batch(batch)?;
        }

        Ok(PipelineResult::new(/* ... */))
    }

    /// Process a single batch through all steps
    fn process_batch(&mut self, batch: Vec<PathBuf>) -> Result<()> {
        // Load batch into memory
        let mut pipeline_data = self.load_batch(batch)?;

        // Process through each step
        for step in &mut self.steps {
            pipeline_data = step.process_batch(pipeline_data)?;

            // Save intermediates if requested
            if self.config.save_intermediates && !step.is_final() {
                self.save_intermediate_results(&pipeline_data, &step.name)?;
            }

            // Memory pressure check - close session if needed
            if self.batch_manager.should_release_memory()? {
                step.close_session();
            }
        }

        // Save final results
        self.save_final_results(pipeline_data)?;

        Ok(())
    }
}
```

### 4. Memory Management System

```rust
#[derive(Debug)]
pub struct BatchManager {
    memory_limit: usize,
    current_memory_usage: usize,
    batch_size_strategy: BatchSizeStrategy,
}

#[derive(Debug, Clone)]
pub enum BatchSizeStrategy {
    Fixed(usize),
    Adaptive { max_memory_mb: usize },
    PerStep { limits: HashMap<String, usize> },
}

impl BatchManager {
    /// Calculate optimal batch size based on memory constraints
    pub fn calculate_batch_size(
        &self,
        image_sizes: &[ImageDimensions],
        pipeline_steps: &[PipelineStep],
    ) -> Result<usize> {
        // Calculate memory requirements per image through pipeline
        let memory_per_image = self.estimate_memory_per_image(image_sizes, pipeline_steps)?;

        // Determine batch size that fits in memory limit
        let max_batch_size = self.memory_limit / memory_per_image;

        Ok(max_batch_size.max(1).min(100)) // Reasonable bounds
    }

    /// Check if we should release memory (close sessions)
    pub fn should_release_memory(&self) -> Result<bool> {
        Ok(self.current_memory_usage > self.memory_limit * 8 / 10) // 80% threshold
    }
}
```

### 5. Enhanced Output Management

```rust
pub struct PipelineOutputManager {
    base_manager: OutputManager,
    pipeline_config: PipelineConfig,
    step_outputs: HashMap<String, Vec<PathBuf>>,
}

impl PipelineOutputManager {
    /// Generate output paths for pipeline results
    pub fn generate_pipeline_paths(
        &self,
        source_path: &Path,
        step_name: &str,
        is_final: bool,
    ) -> Result<OutputPaths> {
        if is_final {
            // Final outputs go to main output directory
            self.base_manager.generate_main_output_path(step_name, "jpg")
        } else if self.pipeline_config.save_intermediates {
            // Intermediate outputs go to step-specific subdirectories
            self.generate_intermediate_paths(source_path, step_name)
        } else {
            // No output paths needed for intermediate steps
            Ok(OutputPaths::none())
        }
    }

    /// Handle complex pipeline metadata
    pub fn create_pipeline_metadata(
        &self,
        pipeline_results: &[PipelineStepResult],
    ) -> Result<BeakerMetadata> {
        // Aggregate metadata from all pipeline steps
        // Include timing, configuration, and output information
        // Support both step-specific and overall pipeline metadata
    }
}
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal**: Establish core pipeline infrastructure without breaking existing functionality.

**Tasks**:
1. **Add PipelineCommand to CLI** - Extend main.rs with new subcommand
2. **Create PipelineConfig system** - Configuration parsing and validation
3. **Implement basic PipelineProcessor** - File-based processing only
4. **Add pipeline tests** - Basic functionality validation

**Acceptance Criteria**:
- `beaker pipeline --steps cutout,head --output-dir results *.jpg` works
- Uses existing file-based processing (no optimization yet)
- Maintains compatibility with existing commands
- Basic error handling and validation

**Risk**: Low - purely additive changes

### Phase 2: Memory Optimization (Week 3-4)

**Goal**: Add in-memory data flow and batch processing.

**Tasks**:
1. **Extend ModelProcessor trait** - Add in-memory processing methods
2. **Implement PipelineData flow** - In-memory image passing
3. **Add BatchManager** - Memory-aware batch size calculation
4. **Update existing models** - Support both file and in-memory processing

**Acceptance Criteria**:
- Significant performance improvement for multi-step pipelines
- Memory usage stays within configured limits
- Batch processing works correctly
- All existing functionality preserved

**Risk**: Medium - requires careful memory management

### Phase 3: Advanced Features (Week 5-6)

**Goal**: Add step-specific overrides and advanced output management.

**Tasks**:
1. **Implement step-specific overrides** - Parse `cutout:--alpha-matting` syntax
2. **Enhanced output management** - Intermediate outputs, complex naming
3. **Comprehensive metadata** - Pipeline-aware metadata generation
4. **Performance tuning** - Optimize batch sizes and memory usage

**Acceptance Criteria**:
- Step-specific overrides work correctly
- --save-intermediates produces expected outputs
- Rich metadata includes pipeline timing and configuration
- Performance meets or exceeds manual pipeline execution

**Risk**: Medium - complex configuration parsing

### Phase 4: Polish & Documentation (Week 7)

**Goal**: Production readiness and comprehensive testing.

**Tasks**:
1. **Comprehensive testing** - Edge cases, error conditions, performance
2. **Documentation** - Help text, examples, troubleshooting
3. **Performance benchmarking** - Validate efficiency improvements
4. **Error handling** - Graceful failure modes and recovery

**Acceptance Criteria**:
- All tests pass, including edge cases
- Clear help text and error messages
- Documented performance characteristics
- Ready for production use

**Risk**: Low - refinement and validation

## Required API Changes

### 1. ModelProcessor Trait Extensions

```rust
pub trait ModelProcessor {
    // Existing methods preserved for backward compatibility
    fn process_single_image(...) -> Result<Self::Result>;

    // New methods for pipeline support
    fn process_pipeline_data(
        session: &mut Session,
        data: PipelineData,
        config: &Self::Config,
    ) -> Result<PipelineData> {
        // Default implementation: convert to file-based processing
        // Models can override for true in-memory processing
    }

    fn memory_requirements(config: &Self::Config) -> MemoryRequirements {
        // Default conservative estimate
        MemoryRequirements::conservative()
    }

    fn accepts_input_from(previous_step: &str) -> bool {
        // Default: accept from any step
        true
    }
}
```

### 2. Configuration System Extensions

```rust
// Extend existing configs to support pipeline usage
impl HeadDetectionConfig {
    /// Create config from pipeline step overrides
    pub fn from_pipeline_args(
        base: BaseModelConfig,
        overrides: &[String]
    ) -> Result<Self> {
        // Parse step-specific arguments like --confidence=0.8
    }
}
```

### 3. Output Manager Extensions

```rust
impl OutputManager {
    /// Pipeline-aware path generation
    pub fn generate_pipeline_output_path(
        &self,
        step_name: &str,
        is_intermediate: bool,
        sequence_number: Option<usize>,
    ) -> Result<PathBuf> {
        // Handle complex pipeline output naming
    }
}
```

## Alternatives Considered

### Alternative 1: Simple Sequential Processing

**Description**: Execute existing commands sequentially with shared temporary directory.

**Pros**:
- Minimal code changes required
- Leverages existing tested infrastructure
- Low implementation risk

**Cons**:
- No performance benefits over manual chaining
- Still has I/O overhead between steps
- Limited value proposition

**Verdict**: Rejected - insufficient value for the implementation effort

### Alternative 2: Full In-Memory Pipeline

**Description**: Complete in-memory processing with no intermediate file I/O.

**Pros**:
- Maximum performance optimization
- Minimal disk I/O
- Elegant data flow design

**Cons**:
- High implementation complexity
- Significant memory management challenges
- Difficult error recovery
- Risk of memory exhaustion

**Verdict**: Considered but too risky for initial implementation

### Alternative 3: Hybrid File-Memory Approach (Recommended)

**Description**: File-based processing with in-memory optimizations and batch processing.

**Pros**:
- Significant performance improvements
- Manageable implementation complexity
- Graceful degradation under memory pressure
- Builds on existing infrastructure

**Cons**:
- Still some I/O overhead
- More complex than simple sequential approach

**Verdict**: **Selected** - optimal balance of benefits and implementation risk

## Scoping Options

### Scope 1: Minimal Viable Pipeline (MVP)

**Features**:
- Basic `--steps` parsing for cutout,head combination
- File-based processing with existing infrastructure
- Simple output management
- No step-specific overrides

**Pros**: Low risk, quick implementation, immediate user value
**Cons**: Limited performance benefits, not extensible
**Timeline**: 2-3 weeks

### Scope 2: Optimized Pipeline (Recommended)

**Features**:
- In-memory data flow between steps
- Memory-aware batch processing
- Step-specific override syntax
- Comprehensive output management
- --save-intermediates support

**Pros**: Significant performance gains, extensible design, full feature set
**Cons**: Higher complexity, longer timeline
**Timeline**: 6-7 weeks

### Scope 3: Advanced Pipeline System

**Features**:
- Everything in Scope 2 plus:
- Dynamic pipeline composition
- Plugin architecture for models
- Advanced memory optimization
- Parallel processing within steps

**Pros**: Future-proof, maximum performance
**Cons**: High complexity, long timeline, over-engineering risk
**Timeline**: 10+ weeks

**Recommendation**: **Scope 2** provides the best balance of value and implementation risk.

## Corner Cases and Considerations

### 1. Memory Management

**Challenge**: Managing peak memory usage with large batches
**Solution**:
- Conservative batch size calculation
- Memory pressure monitoring
- Session lifecycle management
- Graceful degradation to smaller batches

### 2. Error Handling

**Challenge**: Handling failures in multi-step pipelines
**Solution**:
- Fail-fast approach for configuration errors
- Continue processing remaining images on individual failures
- Detailed error reporting with step context
- Partial results preservation

### 3. Input/Output Compatibility

**Challenge**: Ensuring output from one step is valid input for next
**Solution**:
- Define standard data exchange format (DynamicImage + metadata)
- Validation hooks in PipelineModelProcessor trait
- Clear error messages for incompatible step combinations

### 4. Configuration Complexity

**Challenge**: Step-specific overrides can create complex configurations
**Solution**:
- Careful parsing with validation
- Clear error messages for syntax errors
- Fallback to step defaults for undefined overrides
- Documentation with examples

### 5. Output Management Complexity

**Challenge**: Managing outputs from multi-step pipelines
**Solution**:
- Consistent naming conventions
- Clear separation of final vs intermediate outputs
- Metadata tracking for output provenance
- Optional cleanup of intermediate files

## Performance Expectations

### Baseline Performance (Current)

For 100 images through cutout → head pipeline:
- **Current approach**: ~2x session creation overhead + I/O overhead
- **File I/O**: Write 100 cutout images, read 100 for head detection
- **Memory**: Peak usage: single image + model weights

### Expected Performance (Pipeline)

- **Session overhead**: Eliminated (reuse sessions across batches)
- **File I/O**: Reduced by ~50% (no intermediate file writes)
- **Memory**: Controlled peak usage through batch management
- **Overall speedup**: 30-50% improvement expected

### Memory Usage Targets

- **Conservative mode**: 4GB peak memory usage
- **Aggressive mode**: 8GB peak memory usage
- **Batch sizes**: 10-50 images depending on resolution and memory limit

## Testing Strategy

### 1. Unit Tests

- Configuration parsing and validation
- Batch size calculation algorithms
- Memory management components
- Error handling edge cases

### 2. Integration Tests

- End-to-end pipeline execution
- Step-specific override processing
- Output file generation and naming
- Metadata accuracy

### 3. Performance Tests

- Memory usage validation
- Processing speed benchmarks
- Batch size optimization
- Session reuse efficiency

### 4. Compatibility Tests

- Existing command functionality preserved
- Model accuracy maintained through pipeline
- Output format consistency

## Extensibility Considerations

### Future Models

The pipeline architecture supports adding new models:

1. **Pose Detection**: Could add `pose` step for bird pose estimation
2. **Head Orientation**: Could add `orientation` step for head angle detection
3. **Multi-Detection**: Could support detecting multiple birds per image

### Pipeline Composition

Advanced pipeline features for future consideration:

1. **Conditional Steps**: Steps that run based on previous results
2. **Parallel Branches**: Multiple processing paths that merge
3. **Custom Pipelines**: User-defined pipeline configurations

### Performance Scaling

1. **Multi-GPU Support**: Distribute steps across multiple GPUs
2. **Distributed Processing**: Process different batches on different machines
3. **Model Quantization**: Optimize models for pipeline usage

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Memory management bugs | Medium | High | Extensive testing, conservative defaults |
| Performance degradation | Low | Medium | Benchmarking, fallback to file-based processing |
| Configuration complexity | Medium | Medium | Clear documentation, validation |
| Breaking existing functionality | Low | High | Comprehensive regression testing |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Underestimated complexity | Medium | Medium | Phased implementation, early prototyping |
| Scope creep | Medium | Medium | Clear scope definition, incremental delivery |
| Testing overhead | Low | Low | Parallel testing development |

## Success Criteria

### Functional Success

1. **Basic Pipeline**: `beaker pipeline --steps cutout,head *.jpg` works correctly
2. **Step Overrides**: `cutout:--alpha-matting head:--confidence=0.8` syntax works
3. **Output Management**: Final and intermediate outputs generated correctly
4. **Metadata**: Comprehensive pipeline metadata generated

### Performance Success

1. **Speed**: 30-50% improvement over manual pipeline execution
2. **Memory**: Peak usage stays within configured limits
3. **Reliability**: No memory leaks or crashes under normal usage

### User Experience Success

1. **Ergonomics**: Simpler than manual multi-step workflows
2. **Documentation**: Clear help text and examples
3. **Error Messages**: Helpful error messages for common mistakes

## Conclusion

The proposed pipeline subcommand represents a significant enhancement to beaker's capabilities, providing both user convenience and performance improvements. The hybrid file-memory approach offers an optimal balance of benefits and implementation complexity.

Key success factors:
1. **Phased implementation** reduces risk and enables early validation
2. **Memory management** ensures reliable operation on large datasets
3. **Extensible design** supports future models and features
4. **Comprehensive testing** validates correctness and performance

The implementation plan provides a clear path from basic functionality to an optimized, production-ready pipeline system that will significantly improve beaker's utility for computer vision workflows.
