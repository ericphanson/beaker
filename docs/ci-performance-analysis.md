# CI Performance Analysis

## Executive Summary

`just ci` takes **7 minutes clean** or **3.5-4 minutes incremental**. This is too slow for AI agents who run it frequently during development. We need faster validation workflows.

## Detailed Timing Results

### Full Timing Breakdown (Clean Build)

| Step | Duration | % of Total |
|------|----------|-----------|
| Build Release | 152s (2m 32s) | 35.3% |
| Install cargo-nextest* | 73s (1m 13s) | 17.0% |
| Run Tests (nextest) | 200s (3m 20s) | 46.5% |
| Test CLI Help | 0s | 0.0% |
| Preload Models | 3s | 0.7% |
| Test CLI Detect | 0s | 0.0% |
| Test Execution Providers | 2s | 0.5% |
| **TOTAL** | **430s (7m 10s)** | **100%** |

*One-time cost if cargo-nextest not pre-installed

### Incremental Build Scenarios

| Scenario | Build Time | Test Time | Total | vs Clean |
|----------|-----------|-----------|-------|----------|
| Clean build | 152s | 200s | **430s** | Baseline |
| Incremental (code change) | 91-111s | 122s | **213-233s** | 50% faster |

**Key Finding:** Even incremental builds take 3.5-4 minutes because release mode has expensive optimizations.

## Bottleneck Analysis

### 1. Tests (46.5% of time, 200s clean / 122s incremental)

**Slow integration tests:**
- `collision_detection_tests`: 60+ seconds each
- `env_variable_tests`: 10-20 seconds each
- `metadata_based_tests`: 1-3 seconds each

**Why so slow?**
- Integration tests spawn actual CLI processes
- Each test loads ONNX models (even with caching)
- Collision tests appear to have timeouts or heavy I/O

### 2. Release Build (35.3% of time, 152s clean / 91-111s incremental)

**Why incremental is still slow:**
- Release mode optimizations (codegen-units, LTO, etc.)
- Large dependency graph requires recompilation
- Link time is expensive in release mode

### 3. cargo-nextest Installation (17% of time, 73s)

**One-time cost** - only happens when nextest not installed. Solution: document that agents should pre-install nextest.

## Optimization Strategies

### Strategy 1: Fast Dev CI (RECOMMENDED)

Add a `ci-dev` command that skips slow integration tests and uses debug builds for tests:

```bash
ci-dev:
    @echo "Running fast developer CI..."
    @cargo clippy --all-targets -- -D warnings
    @cargo nextest run --lib --release --failure-output=immediate-final
    @echo "✓ Fast CI passed!"
```

**Expected time:** 10-30 seconds (95% faster than current incremental)

**Trade-off:** Skips integration tests. Use full `just ci` before creating PR.

### Strategy 2: Parallel Test Execution

Current nextest already parallelizes, but slow tests dominate:

```bash
ci-parallel:
    @cargo nextest run --release --test-threads=16 --failure-output=immediate-final
```

**Expected time:** Minimal improvement (already parallel, bottleneck is individual slow tests)

### Strategy 3: Separate Slow Tests

Mark slow integration tests and allow skipping them:

```rust
#[test]
#[ignore] // Skip in fast CI
fn test_collision_detection_slow() { ... }
```

```bash
ci-fast:
    @cargo nextest run --release --all -- --skip ignored
```

**Expected time:** 60-90 seconds (70% faster)

### Strategy 4: Build Caching with sccache

Install and configure sccache for compilation caching:

```bash
# One-time setup
cargo install sccache
export RUSTC_WRAPPER=sccache
```

**Expected time:** First build 430s, subsequent code-only changes 10-30s

**Trade-off:** Requires additional setup and disk space

### Strategy 5: Debug Builds for Development

Use debug builds during development, release only for final validation:

```bash
ci-debug:
    @cargo clippy --all-targets -- -D warnings
    @cargo nextest run --all
```

**Expected time:** 30-60 seconds (85% faster)

**Trade-off:** Debug binaries are slower, some bugs only appear in release mode

## Recommendations for AI Agents

### Immediate Action (Add to justfile)

```bash
# Fast CI for incremental development (REQUIRED before pushing)
ci-dev:
    @echo "Running developer CI (incremental)..."
    @cargo clippy --all-targets -- -D warnings
    @cargo nextest run --release --lib --failure-output=immediate-final
    @cargo nextest run --release --bins --failure-output=immediate-final
    @echo "✓ Dev CI passed! Run 'just ci' before finalizing PR."

# Full CI (REQUIRED before creating PR)
ci:
    @echo "Running full CI workflow..."
    @just build-release
    @just test
    @just test-cli-full
    @echo "CI workflow complete!"
```

### Updated Development Workflow

**During development and before pushing (REQUIRED):**
```bash
just ci-dev  # 80s with changes, ~3s without
```

**Before creating PR (REQUIRED):**
```bash
just ci      # 3.5-7m full validation
```

## Expected Time Savings

| Command | Time (clean) | Time (incremental) | Use Case |
|---------|--------------|-------------------|----------|
| `just ci` | 430s (7m) | 213s (3.5m) | Final validation before PR (REQUIRED) |
| `just ci-dev` | 120s (2m) | **80s → 3s** | **Before pushing (REQUIRED)** |

**Result:** Agents can validate changes in 80s (or 3s when re-validating) instead of 3.5-4 minutes (**62-99% faster**)

## Implementation Priority

1. ✅ **HIGH:** Add `ci-dev` command to justfile (immediate 62-99% speedup for agents)
2. ✅ **HIGH:** Require `ci-dev` before pushing (prevents slow iteration)
3. **MEDIUM:** Document that cargo-nextest should be pre-installed
4. **MEDIUM:** Consider marking slow integration tests with `#[ignore]`
5. **LOW:** Investigate why collision tests take 60+ seconds
6. **LOW:** Evaluate sccache for build caching

## Conclusion

The current `just ci` is optimized for CI/CD pipelines (comprehensive validation), not for rapid iteration. Adding `ci-dev` as a required step before pushing gives agents a 62-99% faster feedback loop while maintaining quality through full `just ci` before PR creation.

**Workflow:**
- **Before pushing:** `just ci-dev` (REQUIRED, 80s with changes / 3s without)
- **Before PR:** `just ci` (REQUIRED, 3.5-7m full validation)
