# Beaker Cleanup Issues - Index

This directory contains structured issue specifications for architectural bugs and technical debt in the Beaker codebase. Each issue is scoped to a single subagent task (1 person, 1 day or less).

## Issue Format

Each issue follows this structure:
- **Status**: 🔴 CRITICAL / 🟡 HIGH / ⚠️ DECISION REQUIRED
- **Priority**: P0 (must fix) → P3 (nice to have)
- **Problem Description**: What's broken and why
- **Root Cause**: Code references with line numbers
- **Impact Assessment**: User/system/data loss risk
- **Reproduction Steps**: Minimal example to verify bug
- **Proposed Solution**: Detailed implementation plan with code samples
- **Dependencies**: Blockers and related issues
- **Decisions Required**: Explicit callouts for maintainer decisions
- **Estimated Effort**: Hours breakdown
- **Success Metrics**: How to verify fix works

## Priority Levels

- **P0**: CRITICAL - Must fix before next release (data loss, corruption, broken UX)
- **P1**: HIGH - Should fix soon (inconsistencies, missing validation)
- **P2**: MEDIUM - Fix when convenient (optimizations, polish)
- **P3**: LOW - Nice to have (cleanup, documentation)

## Issue Categories

### 🔴 CRITICAL (P0) - Must Fix Immediately

#### Data Integrity
- **[CRITICAL-02](CRITICAL-02-partial-download-corruption.md)**: Partial Downloads Leave Corrupt Files
  - Interrupted downloads corrupt cache, must restart from 0%
  - Wastes bandwidth on slow connections
  - **Fix**: Atomic writes (temp file + rename)
  - **Effort**: 0.5 day
  - **Blocks**: Future resumable downloads enhancement

#### User Experience
- **[CRITICAL-04](CRITICAL-04-quality-command-zero-output.md)**: Quality Command Produces Zero Output
  - `beaker quality image.jpg` succeeds but creates no files
  - Wastes computation, confusing UX
  - **Fix**: Always save metadata OR generate visualization OR require --metadata
  - **Effort**: 0.5 day (Option A) or 1.5 days (Option B visualization)
  - **Decision**: Which output type? (metadata / visualization / require flag)

### 🟡 HIGH (P1) - Fix Soon

#### UX Improvements
- **[UX-01](UX-01-cli-validation-inconsistency.md)**: CLI Validation Inconsistency
  - DetectCommand can validate, Cutout/Quality cannot
  - Inconsistent developer experience
  - **Fix**: Standardize all to `Result<Self, String>`
  - **Effort**: 0.5 day
  - **Enables**: All other validation fixes

- **[UX-02](UX-02-conflicting-flags-silently-ignored.md)**: Conflicting Flags Silently Ignored
  - `--alpha-matting --background-color` ignores background color
  - **Fix**: Validation after UX-01
  - **Effort**: 1 hour
  - **Requires**: UX-01

- **[UX-03](UX-03-numeric-range-validation.md)**: Missing Numeric Range Validation
  - `--confidence 2.0` accepted (should be 0.0-1.0)
  - **Fix**: Add `value_parser` with range checks
  - **Effort**: 2 hours

### ⚠️ FEATURE DECISIONS (P2) - Maintainer Input Required

- **[FEATURE-01](FEATURE-01-yolo-removal-decision.md)**: YOLO Support - Remove or Keep?
  - 300+ lines of legacy code
  - Removing unlocks Apache 2.0 license (currently AGPL-3.0)
  - **Decision**: Remove immediately / Deprecate / Keep?
  - **Impact**: License change, 50KB binary reduction
  - **Effort**: 1 day if removed

- **[FEATURE-02](FEATURE-02-eye-beak-detection-decision.md)**: Eye/Beak Detection - Finish or Remove?
  - Half-implemented (works for crops, not visualization)
  - Unreliable quality
  - **Decision**: Document as experimental / Finish / Remove?
  - **Effort**: 30 min (docs) or 4 hours (finish) or 2 hours (remove)

- **[FEATURE-03](FEATURE-03-quality-debug-images-cleanup.md)**: Quality Debug Images Always Created
  - Empty `quality_debug_images_*/` directories pollute output
  - **Fix**: Only create when `-vv` debug logging enabled
  - **Effort**: 1 hour

### 🟢 MEDIUM (P2) - Fix When Convenient

See **[MEDIUM-PRIORITY-ISSUES.md](MEDIUM-PRIORITY-ISSUES.md)** for 10 smaller issues (1-2 hours each):
- Env var URL checksum bypass
- Inconsistent checksum algorithms
- Unbounded cache growth
- CoreML cache races
- No overwrite protection
- Metadata parse error recovery
- Quality results wrapper
- File naming repeated suffixes
- Redundant checksum verification

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Goal**: Prevent data loss and corruption

1. **CRITICAL-02** (partial downloads)
   - Use atomic temp file writes
   - **Effort**: 0.5 day
   - **Deliverable**: Safe interrupted downloads

2. **CRITICAL-04** (quality zero output)
   - Make quality command always save metadata (simplest fix)
   - **Effort**: 0.5 day
   - **Deliverable**: Quality command produces visible results

**Total Week 1**: 2 days of implementation

### Phase 2: Validation & UX (Week 2)
**Goal**: Consistent error handling and validation

1. **UX-01** (validation infrastructure)
   - Standardize config creation signatures
   - **Effort**: 0.5 day
   - **Blocks**: UX-02, UX-03

2. **UX-02** + **UX-03** (flag validation)
   - Add conflicting flag detection
   - Add numeric range checks
   - **Effort**: 0.5 day (both together after UX-01)

3. **FEATURE-03** (debug image cleanup)
   - Quick win, easy fix
   - **Effort**: 0.25 day

**Total Week 2**: 1.25 days of implementation

### Phase 3: Feature Decisions (Week 3)
**Goal**: Reduce technical debt

**Requires maintainer decisions BEFORE work**:

1. **FEATURE-01** (YOLO removal)
   - **Decision**: Remove or deprecate?
   - If remove: 1 day
   - **Outcome**: Simpler code + license flexibility

2. **FEATURE-02** (eye/beak detection)
   - **Decision**: Document / Finish / Remove?
   - 30 min to 4 hours depending on choice

**Total Week 3**: 0.5-2 days depending on decisions

### Phase 4: Medium Priority (Week 4+)
**Goal**: Polish and optimization

Pick from [MEDIUM-PRIORITY-ISSUES.md](MEDIUM-PRIORITY-ISSUES.md):
- Cache management (MEDIUM-03, MEDIUM-04)
- Overwrite protection (MEDIUM-05)
- Metadata recovery (MEDIUM-06)
- Checksum optimization (MEDIUM-09)

**Each**: 1-8 hours, can be done incrementally

## Decision Summary

Issues requiring maintainer decisions:

| Issue | Decision | Options | Recommendation |
|-------|----------|---------|----------------|
| CRITICAL-02 | Temp file naming? | `.tmp` / `.{pid}.tmp` / `.{timestamp}.tmp` | `.tmp` (simple) |
| CRITICAL-04 | Quality output? | Always metadata / Visualization / Require flag | Always metadata (simplest) |
| CRITICAL-04 | If visualization? | Heatmap / Histogram / Side-by-side / Reuse debug | Reuse debug images |
| UX-01 | Error type? | `String` / `anyhow::Error` / Custom enum | `String` (consistency) |
| FEATURE-01 | YOLO support? | Remove / Deprecate / Keep | Remove (unlocks Apache 2.0) |
| FEATURE-01 | License change? | Apache 2.0 / Dual license / Keep AGPL | Apache 2.0 |
| FEATURE-02 | Eye/Beak detection? | Document experimental / Finish / Remove | Document experimental |
| MEDIUM-02 | Checksum algorithm? | Switch to SHA256 / Keep MD5 | Add SHA256 support, migrate gradually |
| MEDIUM-05 | Overwrite behavior? | Always overwrite / Require --force | Require --force (safety) |

## Testing Strategy

### Unit Tests
Each issue includes specific test requirements in its file.

### Integration Tests
After Phase 1 (critical fixes):
```bash
# Test batch processing with collisions
beaker detect dir1/bird.jpg dir2/bird.jpg --output-dir output/

# Test quality output
beaker quality image.jpg
test -f image.beaker.toml || echo "FAIL"
```

### Regression Tests
- Existing `metadata_based_tests.rs` must pass
- CLI integration tests in `cli_model_tests.rs` must pass
- Add new tests for fixed issues

## Success Metrics

### Phase 1 Complete When:
- [x] Zero data corruption reports from concurrent downloads (CRITICAL-01 ✅ FIXED)
- [x] Interrupted downloads clean up properly (CRITICAL-01 ✅ FIXED)
- [x] Batch processing never loses files silently (CRITICAL-03 ✅ FIXED)
- [ ] Quality command always produces output

### Phase 2 Complete When:
- [x] All CLI validation is consistent (Result return types)
- [x] Invalid numeric ranges rejected at parse time
- [x] Conflicting flags error instead of silent ignore
- [x] No empty debug directories created

### Phase 3 Complete When:
- [x] YOLO decision made and implemented (remove or keep)
- [x] Eye/beak detection documented or improved
- [x] Technical debt reduced by 300+ lines

### Overall Success:
- No critical bugs remain
- All UX inconsistencies resolved
- Codebase is maintainable and well-documented
- License situation is clear (Apache 2.0 if YOLO removed)

## Contributing

### For Implementers
1. Pick an issue from this directory
2. Read the full issue specification
3. Check Dependencies section - implement blockers first
4. Flag any Decision Required sections to maintainer
5. Follow the Implementation Plan in the issue
6. Add tests per Testing Requirements section
7. Verify Success Metrics before submitting PR

### For Maintainers
1. Review Decision Required sections across all issues
2. Make architectural decisions (see Decision Summary table)
3. Prioritize issues based on user impact
4. Approve implementation approaches

## Questions?

Contact: @ericphanson
Report generated: 2025-01-14 (automated analysis by 6 parallel agents)
Analysis codebase commit: `85436bf note`

---

**Total Issues**: 12 detailed + 10 medium (summary)
**Total Estimated Effort**: ~2-3 weeks for critical + high priority
**Primary Agents**: ModelProcessor Integration, CLI Routing, Metadata Generation, Model Management, Output Management, Dead Code Analysis
