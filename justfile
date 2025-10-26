# Beaker CI/CD justfile
# Common tasks for CI workflows and local development

# Import GUI commands from beaker-gui/justfile
import? 'beaker-gui/justfile'

# Default recipe shows available commands
default:
    @just --list

# ─────────────────────────────────────────────────────────────
# Linting & Formatting
# ─────────────────────────────────────────────────────────────

# Check code formatting
fmt-check:
    cargo fmt --all --check

# Format code
fmt:
    cargo fmt --all

# Run clippy with strict settings
clippy target="":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        cargo clippy --all-targets --target {{target}} -- -D warnings
    else
        cargo clippy --all-targets -- -D warnings
    fi

# Run all lint checks
lint target="":
    @echo "Running format check..."
    @just fmt-check
    @echo "Running clippy..."
    @just clippy {{target}}

# ─────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────

# Build release binary for target
build-release target="":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        cargo build --release --target {{target}}
    else
        cargo build --release
    fi

# Build and show binary info (Unix-like systems)
build-info target="":
    #!/usr/bin/env bash
    just build-release {{target}}

    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    if [ -f "$BINARY" ]; then
        echo "Binary information:"
        ls -lh "$BINARY"
        echo "Binary size: $(du -h "$BINARY" | cut -f1)"
        if command -v strip &> /dev/null; then
            strip "$BINARY" -o /tmp/beaker-stripped 2>/dev/null || true
            if [ -f /tmp/beaker-stripped ]; then
                echo "Stripped size: $(du -h /tmp/beaker-stripped | cut -f1)"
                rm /tmp/beaker-stripped
            fi
        fi
    fi

# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────

# Run all tests with nextest (recommended - ~5x faster via parallelization)
test target="":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        cargo nextest run --release --all --target {{target}}
    else
        cargo nextest run --release --all
    fi

# Run all tests with standard cargo test (legacy - slower but works without nextest)
test-legacy target="":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        cargo test --release --all --target {{target}} --verbose
    else
        cargo test --release --all --verbose
    fi

# Pre-download models by running a simple detect command
preload-models target="" device="auto":
    #!/usr/bin/env bash
    set +e  # Don't exit on error

    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    if [ -f "$BINARY" ] && [ -f "example.jpg" ]; then
        echo "Pre-downloading models..."
        "$BINARY" cutout example.jpg --device {{device}} > /dev/null 2>&1 || true
        echo "Model preload completed"
    else
        echo "Skipping model preload (binary or example image not found)"
    fi

# Test CLI help command
test-cli-help target="":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    echo "Testing CLI help..."
    "$BINARY" --help
    echo "CLI help works"

# Test CLI detect command
test-cli-detect target="" device="auto":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    echo "Testing CLI detect command..."
    echo "Device: {{device}}, Target: {{target}}"
    "$BINARY" detect example.jpg --confidence 0.5 --device {{device}} --metadata
    echo "CLI detect command works"

# Test execution providers with different devices
test-execution-providers target="" os="linux":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    echo "Testing execution provider details..."
    echo "Platform: {{os}}, Target: {{target}}"

    if [ "{{os}}" = "macos" ]; then
        echo "Expected: CoreML available and used"
    else
        echo "Expected: CoreML not available, fallback to CPU"
    fi

    echo ""
    echo "=== Testing with --device auto ==="
    "$BINARY" detect example.jpg --confidence 0.5 --device auto --metadata

    echo ""
    echo "=== Testing with --device cpu ==="
    "$BINARY" detect example.jpg --confidence 0.5 --device cpu --metadata

    echo "Execution provider testing complete"

# Test CLI cutout command
test-cli-cutout target="" device="auto":
    #!/usr/bin/env bash
    if [ -n "{{target}}" ]; then
        BINARY="target/{{target}}/release/beaker"
    else
        BINARY="target/release/beaker"
    fi

    echo "Testing CLI cutout command..."
    echo "Device: {{device}}, Target: {{target}}"
    "$BINARY" cutout example.jpg --device {{device}}
    echo "CLI cutout command works"

# Run smoke tests (basic validation that binary and models work)
smoke-test target="" device="auto":
    @echo "Running smoke tests..."
    @just test-cli-help {{target}}
    @just preload-models {{target}} {{device}}
    @just test-cli-detect {{target}} {{device}}
    @just test-cli-cutout {{target}} {{device}}
    @echo "All smoke tests passed!"

# Run full CLI test suite
test-cli-full target="" device="auto" os="linux":
    @just test-cli-help {{target}}
    @just preload-models {{target}} {{device}}
    @just test-cli-detect {{target}} {{device}}
    @just test-execution-providers {{target}} {{os}}

# ─────────────────────────────────────────────────────────────
# CI Workflows
# ─────────────────────────────────────────────────────────────

# Full CI workflow: lint, build, test
ci target="" device="auto" os="linux":
    @echo "Running full CI workflow..."
    @just build-release {{target}}
    @just test {{target}}
    @just test-cli-full {{target}} {{device}} {{os}}
    @echo "CI workflow complete!"

# Lint-only workflow (typically run once on Linux)
ci-lint target="":
    @echo "Running lint checks..."
    @just lint {{target}}
    @echo "Lint checks complete!"

# Fast CI for incremental development (agents should use this)
# Skips slow integration tests - run full 'ci' before creating PR
# Timing: ~80s with code changes, ~3s without changes (vs 213s for full incremental ci)
ci-dev target="":
    #!/usr/bin/env bash
    echo "Running developer CI (incremental, skips integration tests)..."
    if [ -n "{{target}}" ]; then
        cargo clippy --all-targets --target {{target}} -- -D warnings
        cargo nextest run --release --lib --bins --target {{target}} --failure-output=immediate-final
    else
        cargo clippy --all-targets -- -D warnings
        cargo nextest run --release --lib --bins --failure-output=immediate-final
    fi
    echo "✓ Dev CI passed! Run 'just ci' before finalizing PR."

# Ultra-fast smoke test (basic validation only)
# Timing: ~3-5s
ci-smoke target="":
    #!/usr/bin/env bash
    echo "Running smoke test..."
    if [ -n "{{target}}" ]; then
        cargo nextest run --release --lib --target {{target}}
    else
        cargo nextest run --release --lib
    fi
    just test-cli-help {{target}}
    echo "✓ Smoke test passed!"
