# Beaker Pipeline Subcommand: Design and Implementation Plan

## Executive Summary

This document outlines the design and implementation plan for the `beaker pipeline` subcommand, which will enable sequential processing of multiple computer vision models. **The primary goals are:**

1. **User ergonomics**: Single command vs multiple manual steps, eliminating user friction in complex multi-step workflows
2. **Internal API cleanliness**: Clean separation of concerns with models agnostic to pipeline vs standalone usage
3. **Simplicity**: Start with file-based sequential processing, defer memory optimization to later phases

Based on real performance measurements, file I/O represents only 0.8-1.5% of total processing time, making ergonomic workflow improvements the primary value proposition rather than performance optimization.

**Current scope limitation:** The initial pipeline design targets sequential processing with consistent image counts (cutout → head detection). Complex scenarios with variable image counts, metadata dependencies, and many-to-one transformations may be better served by bash scripts until more advanced pipeline features are implemented.

Currently, users must run separate beaker commands to chain operations:
```bash
beaker cutout *.jpg --output-dir cutout_results
beaker head cutout_results/*.png --crop --output-dir final_results
```

The proposed pipeline subcommand addresses user friction and workflow complexity:
```bash
beaker pipeline --steps cutout,head --output-dir results --crop *.jpg
```

## Complex Pipeline Scenarios

### Collage Pipeline Example

Consider a hypothetical `collage` command that creates artistic composites from multiple oriented bird images. This represents a "many-to-one" transformation consuming both images and metadata:

**Pipeline stages:**
1. **Bird detection** → Detect birds in directory photos (0+ birds per photo)
2. **Image filtering** → Discard photos with zero birds detected
3. **Bird cropping** → Crop to detected birds (may yield more/fewer images than input)
4. **Background removal** → Cut out backgrounds from cropped birds
5. **Head detection** → Detect heads and beak-eye orientation on cropped images
6. **Collage generation** → Combine head crops + orientation metadata → single collage image

**Data flow complexity:**
- **Variable image counts** at each stage (many → fewer → many → one)
- **Metadata propagation** from detection through to final composition
- **Cross-step dependencies** where later steps need data from multiple earlier steps

### Pipeline vs Bash Script Comparison

#### Bash Script Approach
```bash
#!/bin/bash
set -e

# Stage 1: Detect birds, filter empties
mkdir -p stage1_birds
beaker head *.jpg --output-dir stage1_birds --confidence 0.3 --metadata
find stage1_birds -name "*.toml" -exec grep -L "detections = \[\]" {} \; | \
  xargs -I {} dirname {} | sort -u > valid_dirs.txt

# Stage 2: Crop to birds
mkdir -p stage2_crops
while read dir; do
  beaker head "$dir"/*.jpg --crop --output-dir stage2_crops
done < valid_dirs.txt

# Stage 3: Remove backgrounds
mkdir -p stage3_cutouts
beaker cutout stage2_crops/*.png --output-dir stage3_cutouts

# Stage 4: Head detection on crops
mkdir -p stage4_heads
beaker head stage3_cutouts/*.png --crop --metadata --output-dir stage4_heads

# Stage 5: Collage generation (hypothetical)
beaker collage stage4_heads/*.png --metadata-dir stage4_heads --output final_collage.jpg

# Cleanup
rm -rf stage1_birds stage2_crops stage3_cutouts stage4_heads valid_dirs.txt
```

#### Pipeline Command Approach
```bash
# Hypothetical advanced pipeline syntax
beaker pipeline \
  --steps "detect:--confidence=0.3,filter-empty,crop-detections,cutout,head-detect:--confidence=0.7,collage" \
  --output-dir results \
  --save-intermediates \
  *.jpg
```

#### Comparison Analysis

| Aspect | Bash Script | Pipeline Command |
|--------|-------------|------------------|
| **Clarity** | Explicit control flow, easy to debug | Concise but requires learning pipeline syntax |
| **Flexibility** | Full shell capabilities, custom logic | Limited to predefined pipeline operations |
| **Error handling** | Manual cleanup and error checking | Automatic cleanup and consistent error handling |
| **Reusability** | Requires script modification for different params | Command-line parameter overrides |
| **Intermediate inspection** | Easy access to all intermediate files | Requires `--save-intermediates` flag |
| **Metadata handling** | Manual parsing and coordination | Automatic metadata threading |

**Key insight:** For complex variable-count pipelines, bash scripts may be simpler and more maintainable than trying to encode all logic into a pipeline command.

### Impact of Inference Caching

Inference caching fundamentally changes the performance equation and design tradeoffs:

#### Without Caching (Current State)
- **Inference**: 99%+ of execution time
- **File I/O**: 0.8-1.5% of execution time
- **Pipeline value**: Primarily ergonomic (single command vs multiple steps)
- **Memory optimization**: Minimal performance benefit (1-2% improvement)

#### With Inference Caching
```bash
# First run: Full inference cost
beaker pipeline --steps detect,crop,cutout,head-detect *.jpg
# Subsequent runs with parameter tweaks: Major speedup
beaker pipeline --steps detect,crop,cutout,head-detect:--confidence=0.8 *.jpg  # Only re-runs head-detect
beaker collage results/head-detect/*.png --layout=grid                        # Only runs collage
```

**Cache hit scenarios:**
- **Parameter tuning**: Adjust confidence thresholds without re-running earlier steps
- **Pipeline iteration**: Add/modify final steps while preserving expensive intermediate results
- **Partial failures**: Resume from cached intermediate results after fixing errors

**Performance implications with caching:**
- **First run**: Same performance as sequential commands
- **Subsequent runs**: File I/O becomes significant portion of remaining work
- **Memory optimization value**: Increases substantially for cached scenarios
- **Pipeline value**: Shifts from purely ergonomic to performance-critical

**Design considerations:**
- **Cache invalidation**: Must detect when input changes invalidate cached results
- **Metadata tracking**: Cache must store sufficient metadata to validate reuse
- **Storage overhead**: Intermediate results require substantial disk space
- **Memory vs disk tradeoffs**: In-memory pipelines become much more attractive for cached workflows

**Implementation impact:** Caching makes the case for in-memory processing much stronger, as the performance bottleneck shifts from inference to data marshaling when intermediate results are cached.

## Implementation

### Phase 1: File-Based Foundation

**File-based sequential processing** is the recommended starting point. Each step reads files, processes them using existing `process_single_image` methods, and writes results to temporary directories managed by the pipeline processor.

**Scope:** Handles consistent image count scenarios (1:1 transformations) effectively. Complex scenarios with variable image counts or many-to-one transformations may require bash scripts or future pipeline enhancements.

**Key Components:**
- **PipelineProcessor**: Orchestrates sequential execution using existing file-based model processing ([Issue 2](#issue-2-pipelineprocessor-implementation-with-file-based-processing))
- **CLI Infrastructure**: Argument parsing and step configuration ([Issue 1](#issue-1-pipeline-cli-command-infrastructure))
- **Configuration-based coordination**: Use existing `output_dir` field to coordinate temporary directories ([Issue 3](#issue-3-configuration-based-temporary-directory-support))

**Benefits:** Immediate ergonomic improvements with minimal risk and complexity. Leverages existing robust infrastructure.

**Risks:** Must handle temporary directory cleanup and ensure proper error propagation between steps. Mitigation: Use existing `OutputManager` patterns and standard RAII cleanup.

### Phase 2: Advanced Features

**Enhanced workflow features** building on the file-based foundation:
- **Step-specific overrides**: Parse `cutout:--alpha-matting head:--confidence=0.8` syntax ([Issue 5](#issue-5-step-specific-override-syntax-implementation))
- **Intermediate output management**: `--save-intermediates` support ([Issue 6](#issue-6-intermediate-output-management))
- **Enhanced benchmarking**: Unify performance measurement in `benchmark.py` ([Issue 4](#issue-4-enhanced-performance-analysis-and-benchmarking))

**Corner Cases:** Invalid step parameter combinations require clear validation and error messages. Configuration complexity should be managed through existing config validation patterns.

### Phase 3: Memory Optimization (Future)

**In-memory processing infrastructure** for performance optimization when justified by evidence:
- **ModelProcessor trait extensions**: Add `process_image_data` for in-memory data flow ([Issue 7](#issue-7-in-memory-processing-infrastructure-future))
- **Memory-aware batch processing**: Monitor memory usage and adjust batch sizes dynamically

**Risks:** Memory management complexity and increased coupling between components. Implementation should maintain file-based fallback options.

**Required API Changes:**
- **Phase 1**: No breaking changes - use existing `process_single_image` and configuration modification
- **Phase 2**: Extend argument parsing for step-specific overrides, enhance `OutputManager` for intermediate handling
- **Phase 3**: Extend `ModelProcessor` trait with optional in-memory methods

## Appendix A: Performance Analysis

### Current Bottlenecks (Measured Data)

**Head Detection (example.jpg, CPU):**
- Model inference: 96.0ms (92.7%)
- File I/O: 7.6ms (7.3%)

**Cutout Processing (example.jpg, CPU):**
- Model inference: 1908.9ms (99.1%)
- File I/O: 17.4ms (0.9%)

**Pipeline Potential:** 0.8-1.5% improvement from eliminating intermediate writes

**Conclusion:** Model inference dominates execution time (99%+). The pipeline value proposition is ergonomic workflow improvement, not performance optimization.

## Appendix B: Issues

Orthogonal, well-scoped GitHub issues for parallel development:

### Issue 1: Pipeline CLI Command Infrastructure

**Title**: Add pipeline subcommand with skeleton implementation

**Description**: Create the CLI infrastructure for the pipeline subcommand with basic argument parsing and skeleton implementation that returns "not implemented" errors.

**Scope:**
- Add `PipelineCommand` struct to `src/config.rs` with step parsing (`--steps cutout,head`)
- Add pipeline subcommand to `main.rs` CLI structure
- Create `src/pipeline_processing.rs` with `PipelineProcessor` trait and skeleton implementation
- Add basic help text and argument validation

**Acceptance Criteria:**
- `beaker pipeline --help` shows proper help text
- `beaker pipeline --steps cutout,head input.jpg` returns "Pipeline processing not yet implemented" error
- All existing tests continue to pass

**Dependencies**: None | **Phase**: 1

---

### Issue 2: PipelineProcessor Implementation with File-Based Processing

**Title**: Implement PipelineProcessor with file-based sequential processing and metadata support

**Description**: Implement the actual pipeline processing logic using existing file-based model processing infrastructure. This provides the ergonomic benefits without memory optimization complexity. **Initial scope focuses on 1:1 image transformations (cutout → head detection); complex variable-count scenarios are deferred to future enhancements.**

**Scope:**
- Implement `PipelineProcessor::process_pipeline()` method
- Create temporary directory management for intermediate files
- Sequential execution of cutout followed by head detection using existing `process_single_image` methods
- Generate comprehensive pipeline metadata in TOML format
- Add metadata test validations

**Limitations**: Does not handle variable image counts, filtering steps, or many-to-one transformations. Complex scenarios should use bash scripts until advanced pipeline features are implemented.

**Acceptance Criteria:**
- `beaker pipeline --steps cutout,head --output-dir results *.jpg` works correctly
- Intermediate files are cleaned up automatically
- Pipeline metadata TOML includes timing data for each step
- All existing metadata tests pass plus new pipeline-specific validations

**Dependencies**: [Issue 1](#issue-1-pipeline-cli-command-infrastructure) | **Phase**: 1

---

### Issue 3: Configuration-Based Temporary Directory Support

**Title**: Enable pipeline temporary directory coordination via config modification

**Description**: Implement support for pipeline processors to coordinate temporary directory output by modifying the model configuration's `output_dir` field. This approach keeps models completely agnostic to pipeline vs standalone usage.

**Scope:**
- Document and validate that modifying `BaseModelConfig::output_dir` enables temporary directory output
- Ensure `OutputManager` properly handles pipeline-controlled temporary directories
- Add utility functions for pipeline processors to safely clone and modify configurations

**Acceptance Criteria:**
- Pipeline processor can clone user config and set `output_dir` to a controlled temporary directory
- Models automatically output to the configured temporary directory without any changes
- All model functionality preserved with no API changes required

**Dependencies**: None | **Phase**: 1

---

### Issue 4: Enhanced Performance Analysis and Benchmarking

**Title**: Update benchmark.py to include I/O timing analysis and unify performance measurement

**Description**: Enhance the existing `benchmark.py` script to include file I/O timing analysis and unify all performance measurement into a single comprehensive tool. This replaces the separate `file_io_assessment.py` script.

**Scope:**
- Update `benchmark.py` to parse metadata files and extract I/O timing information
- Add display of % time spent in I/O vs model loading vs inference
- Include pipeline vs sequential manual execution comparison capabilities
- Delete `file_io_assessment.py` to consolidate all benchmarking functionality

**Acceptance Criteria:**
- Script displays % time breakdown: I/O, model loading, inference
- `file_io_assessment.py` is removed from repository
- Script can benchmark both standalone and pipeline processing modes

**Dependencies**: [Issue 2](#issue-2-pipelineprocessor-implementation-with-file-based-processing) | **Phase**: 1

---

### Issue 5: Step-Specific Override Syntax Implementation

**Title**: Add step-specific override parsing and configuration

**Description**: Implement parsing and handling of step-specific overrides using the `cutout:--alpha-matting head:--confidence=0.8` syntax.

**Scope:**
- Extend `PipelineCommand` to parse step-specific override syntax
- Create configuration merging logic for step-specific parameters
- Add validation for step-specific parameter compatibility

**Acceptance Criteria:**
- `beaker pipeline --steps "cutout:--alpha-matting head:--confidence=0.8" input.jpg` works correctly
- Invalid step parameters produce clear error messages
- Configuration validation catches incompatible parameter combinations

**Dependencies**: [Issue 2](#issue-2-pipelineprocessor-implementation-with-file-based-processing) | **Phase**: 2

---

### Issue 6: Intermediate Output Management

**Title**: Add --save-intermediates support to pipeline command and advanced output management

**Description**: Implement comprehensive output management including intermediate output saving and complex naming schemes for pipeline results.

**Scope:**
- Add `--save-intermediates` flag to pipeline command
- Implement pipeline-aware output naming conventions
- Create output organization for complex multi-step pipelines

**Acceptance Criteria:**
- `--save-intermediates` preserves intermediate files in organized directory structure
- Output naming clearly indicates pipeline step provenance
- Metadata includes full pipeline configuration and step timing

**Dependencies**: [Issue 2](#issue-2-pipelineprocessor-implementation-with-file-based-processing) | **Phase**: 2

---

### Issue 7: In-Memory Processing Infrastructure (Future)

**Title**: Add in-memory data flow and memory-aware batch processing

**Description**: Implement the ModelProcessor trait extensions for in-memory processing and add memory-aware batch management for performance optimization.

**Scope:**
- Extend ModelProcessor trait with `process_image_data` for in-memory processing
- Add `BatchManager` for memory-aware batch size calculation
- Update existing models to support both file and in-memory processing modes

**Acceptance Criteria:**
- In-memory processing provides measurable performance improvement for multi-step pipelines
- Memory usage stays within configured limits
- Graceful fallback to file-based processing under memory pressure

**Dependencies**: [Issues 1-2](#issue-1-pipeline-cli-command-infrastructure) | **Phase**: 3

**Note**: This issue should be implemented only after Phase 1-2 issues prove the value of the file-based approach and establish solid foundations.
