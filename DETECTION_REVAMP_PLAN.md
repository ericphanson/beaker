# Detection Command Revamp: From Head to Multi-Class Detection

## Overview

This document outlines the detailed changes needed to revamp the current `beaker head` command into a more general `beaker detect` command that supports multi-class detection (bird, head, eyes, beak) with flexible cropping options.

## Current vs Proposed Design

### Current Design
```bash
beaker head image.jpg --crop --bounding-box --confidence 0.25
```

### Proposed Design
```bash
beaker detect image.jpg --crop=head --bounding-box --confidence 0.25
beaker detect image.jpg --crop=head,bird --confidence 0.25
beaker detect image.jpg --crop=all --confidence 0.25
```

## Design Comparison and Alternatives

### Alternative 1: Comma-separated crop classes (Proposed)
```bash
beaker detect --crop=head,bird,eyes
```
**Pros:**
- Very flexible and extensible
- Single parameter handles complex scenarios
- Follows common CLI patterns (Docker `--mount`, Kubernetes labels)
- Easy to parse and validate

**Cons:**
- Slightly more complex parsing logic
- Might be less discoverable than boolean flags

### Alternative 2: Multiple boolean flags
```bash
beaker detect --crop-head --crop-bird --crop-eyes --crop-beak
```
**Pros:**
- Very clear and discoverable
- Each flag is boolean (simple)
- Excellent tab completion

**Cons:**
- Many flags to maintain
- Less extensible for future classes
- Verbose for multiple classes

### Alternative 3: Repeated parameter
```bash
beaker detect --crop head --crop bird --crop eyes
```
**Pros:**
- Clean syntax
- Extensible
- Natural CLI pattern

**Cons:**
- More complex clap configuration
- Potentially confusing with global --crop flag precedence

### Recommended Design: Alternative 1

Alternative 1 is the best choice because:
1. **Flexibility**: Handles single and multiple classes elegantly
2. **Extensibility**: Easy to add new classes without CLI changes
3. **Convention**: Follows established patterns in other tools
4. **Simplicity**: Single parameter reduces complexity
5. **Power users**: Supports complex combinations like `--crop=head,bird`

## Implementation Plan

### Phase 1: Core Structural Changes

#### 1.1 Module Renaming
- **File**: `src/head_detection.rs` → `src/detection.rs`
- **Imports**: Update all `use crate::head_detection` references
- **Tests**: Update test imports and function names
- **Build script**: Update any embedded model references if needed

#### 1.2 Configuration Updates
- **File**: `src/config.rs`
  - Rename `HeadCommand` → `DetectCommand`
  - Rename `HeadDetectionConfig` → `DetectionConfig`
  - Change `crop: bool` → `crop_classes: Vec<String>`
  - Add crop class parsing function
  - Update conversion methods

#### 1.3 CLI Command Changes
- **File**: `src/main.rs`
  - Change enum variant `Head(HeadCommand)` → `Detect(DetectCommand)`
  - Update command description and help text
  - Update processing logic and symbols

### Phase 2: Detection Logic Enhancement

#### 2.1 Multi-Class Detection Support
- **File**: `src/yolo_postprocessing.rs`
  - Add `class_id: u32` and `class_name: String` to `Detection` struct
  - Update postprocessing to extract class information from model output
  - Add class ID to name mapping (0: bird, 1: head, 2: eyes, 3: beak)
  - Ensure backward compatibility with single-class models

#### 2.2 Detection Result Updates
- **File**: `src/detection.rs` (formerly head_detection.rs)
  - Update `DetectionWithPath` to include class information
  - Update metadata serialization to include class data
  - Update result processing for multi-class scenarios

### Phase 3: Output Management Updates

#### 3.1 Crop Naming Logic
- **File**: `src/output_manager.rs`
  - Add method for class-specific crop naming: `name_crop_head.jpg`
  - Support multiple crops per detection per class
  - Handle numbering for multiple detections: `name_crop_head_1.jpg`, `name_crop_head_2.jpg`

#### 3.2 Crop Generation Logic
- **File**: `src/detection.rs`
  - Update crop generation to filter by requested classes
  - Generate crops only for requested classes
  - Update crop path storage in metadata

### Phase 4: Testing and CI Updates

#### 4.1 Unit Test Updates
- Update all test function names from `test_head_*` to `test_detect_*`
- Update test configurations to use new command structure
- Add tests for multi-class crop scenarios
- Test crop class parsing edge cases

#### 4.2 Integration Test Updates
- **File**: `tests/metadata_based_tests.rs`
  - Update all CLI command calls from `head` to `detect`
  - Add `--crop=head` to maintain equivalent functionality
  - Add new tests for multi-class scenarios

#### 4.3 CI Workflow Updates
- **File**: `.github/workflows/beaker-ci.yml`
  - Update CLI test commands from `head` to `detect --crop=head`
  - Ensure all test scenarios pass with new command structure

### Phase 5: Documentation and Polish

#### 5.1 Help Text and Documentation
- Update command descriptions and examples
- Add crop class documentation
- Update README examples if needed

#### 5.2 Version Information
- Update version command output to reflect detection model instead of head model
- Ensure model version reporting works correctly

## Detailed Technical Specifications

### Crop Class Parsing

```rust
fn parse_crop_classes(input: &str) -> Result<Vec<String>, String> {
    if input == "all" {
        return Ok(vec!["bird".to_string(), "head".to_string(), "eyes".to_string(), "beak".to_string()]);
    }
    
    let classes: Vec<String> = input
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .collect();
    
    for class in &classes {
        if !["bird", "head", "eyes", "beak"].contains(&class.as_str()) {
            return Err(format!("Unknown class: {}. Valid classes: bird, head, eyes, beak", class));
        }
    }
    
    Ok(classes)
}
```

### Class ID Mapping

```rust
fn class_id_to_name(class_id: u32) -> String {
    match class_id {
        0 => "bird".to_string(),
        1 => "head".to_string(), 
        2 => "eyes".to_string(),
        3 => "beak".to_string(),
        _ => format!("unknown_{}", class_id),
    }
}
```

### Crop Output Naming

```rust
// Example outputs:
// bird.jpg -> bird_crop_head.jpg, bird_crop_bird.jpg
// photo.png -> photo_crop_head_1.png, photo_crop_head_2.png (multiple heads)
// image.jpg -> image_crop_eyes.jpg, image_crop_beak.jpg
```

### Metadata Structure

```toml
[tool_results.detect]
model_version = "bird-detector-v2.0"
processing_time_ms = 245.7

[[tool_results.detect.detections]]
x = 100
y = 150  
width = 200
height = 180
confidence = 0.87
class_id = 1
class_name = "head"
crop_path = "bird_crop_head.jpg"

[[tool_results.detect.detections]]
x = 80
y = 120
width = 250
height = 220
confidence = 0.92
class_id = 0  
class_name = "bird"
crop_path = "bird_crop_bird.jpg"
```

## Backward Compatibility

**Intentionally Broken**: As specified in the requirements, backward compatibility is not maintained. Users must:
- Change `beaker head` to `beaker detect`
- Add `--crop=head` if they want cropping functionality
- Update any scripts or workflows using the old command

## Risk Assessment

### Low Risk Changes
- Module renaming (straightforward refactoring)
- CLI command structure (well-defined changes)
- Output naming (additive changes)

### Medium Risk Changes  
- Detection struct modifications (affects serialization)
- Multi-class postprocessing (logic complexity)
- Test updates (many files to change)

### High Risk Changes
- Model output interpretation (must handle both 1-class and 4-class models)
- CI workflow changes (could break automated testing)

## Testing Strategy

### Unit Tests
- Test crop class parsing with valid/invalid inputs
- Test class ID to name mapping
- Test output naming logic for various scenarios
- Test detection struct serialization/deserialization

### Integration Tests
- Test full pipeline with `--crop=head` (equivalent to old behavior)
- Test multi-class cropping with `--crop=head,bird`
- Test error handling for invalid class names
- Test metadata structure with multi-class detections

### Regression Tests
- Ensure single-class model still works (current embedded model)
- Ensure output quality and file naming is consistent
- Ensure performance is not degraded

## Implementation Order

1. **Create planning document** ✅
2. **Rename modules and update imports** (safe, reversible)
3. **Update configuration structures** (medium impact)
4. **Update CLI command structure** (high visibility)
5. **Add multi-class support to detection logic** (core functionality)
6. **Update output naming and crop generation** (user-facing)
7. **Update all tests** (validation)
8. **Update CI workflows** (automation)
9. **Final validation and testing** (quality assurance)

This phased approach minimizes risk by making incremental, testable changes while maintaining a working system at each step.