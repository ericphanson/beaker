# Testing Guide

## Running Tests

This project uses [cargo-nextest](https://nexte.st/) for parallel test execution, providing significant speedup over standard `cargo test`.

### Quick Start

```bash
# Install cargo-nextest (one-time setup)
cargo install cargo-nextest --locked

# Run all tests in parallel (using justfile - recommended)
just test

# Or run directly with cargo-nextest
cargo nextest run --release

# Run specific test suites
cargo nextest run --release --test metadata_based_tests
cargo nextest run --release --test cli_model_tests
cargo nextest run --release --test env_variable_tests
```

### Performance

Integration test performance on a 16-core system:

| Method | Time | Speedup |
|--------|------|---------|
| Sequential `cargo test` | ~81s | 1.0x baseline |
| Parallel `cargo test --test-threads=16` | ~27s | ~3.0x |
| **Parallel `cargo nextest` (recommended)** | **~16s** | **~5.0x** |

The speedup comes from:
- Running tests across all CPU cores in parallel
- Parallelizing execution across different test binaries
- Optimized test scheduling and resource management

### Configuration

Test parallelization is configured in `.config/nextest.toml`:
- `test-threads = "num-cpus"` - Uses all available CPU cores
- Automatic retry for flaky tests
- Clean output with immediate failure reporting

### CI Integration

GitHub Actions automatically uses `cargo-nextest` via the `taiki-e/install-action` for fast CI test runs across all platforms (Linux, macOS, Windows).

### Legacy Testing

If you need to use standard `cargo test` (e.g., if nextest is not installed):

```bash
# Using justfile
just test-legacy

# Or directly with cargo
cargo test --release
```

However, this is **not recommended** as it's significantly slower (~5x) than nextest.

### Using Justfile

The project includes a `justfile` with convenient test commands:

```bash
# Run all tests with nextest (default)
just test

# Run tests for specific target
just test x86_64-unknown-linux-gnu

# Run legacy cargo test
just test-legacy

# Full CI workflow (lint + build + test)
just ci
```
