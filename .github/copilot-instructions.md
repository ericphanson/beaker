# Beaker: Bird Detection CLI Tool

Always follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.

## Working Effectively

### Bootstrap and Build
Execute these commands in order to set up a working development environment:

```bash
# Navigate to main Rust codebase
cd beaker

# Check Rust toolchain (requires Rust 1.70+)
rustc --version && cargo --version

# Format and lint code
cargo fmt
cargo clippy --fix --allow-dirty -- -D warnings

# Build debug version (takes ~40 seconds)
cargo build

# Build release version - NEVER CANCEL: takes 1 minute 40 seconds. Set timeout to 3+ minutes.
cargo build --release

# Run tests - NEVER CANCEL: takes 2 minutes 50 seconds. Set timeout to 5+ minutes.
cargo test --release
```

### Validation and Testing
Always run these validation steps after making changes:

```bash
# Test CLI help
./target/release/beaker --help

# Copy test image for validation
cp ../example.jpg .

# Test basic head detection (requires --crop, --bounding-box, or --metadata)
./target/release/beaker head example.jpg --confidence 0.5 --crop

# Test metadata generation
./target/release/beaker head example.jpg --confidence 0.5 --metadata

# Test cutout functionality (takes ~3 seconds for model loading)
./target/release/beaker cutout example.jpg

# Test with two birds image
cp ../example-2-birds.jpg .
./target/release/beaker head example-2-birds.jpg --confidence 0.5 --crop

# Verify output files are created
ls -la example_crop.jpg example_cutout.png *.beaker.toml
```

### Pre-commit and Code Quality
Run before every commit - NEVER CANCEL: takes ~4 seconds:

```bash
# From repository root
pre-commit run --all-files

# Individual checks if pre-commit unavailable:
cd beaker
cargo fmt --check
cargo clippy -- -D warnings
cargo build --release
cargo test --release
cd ..
ruff check --config=ruff.toml --fix .
ruff format --config=ruff.toml .

# Update line counts
bash scripts/run_warloc.sh
```

## Critical Build Information

### Timing Expectations
- **Debug build**: ~40 seconds
- **Release build**: ~1 minute 40 seconds - NEVER CANCEL, set timeout to 180+ seconds
- **Tests**: ~2 minutes 50 seconds - NEVER CANCEL, set timeout to 300+ seconds
- **Pre-commit hooks**: ~4 seconds
- **CLI head detection**: ~0.2 seconds (after initial model load)
- **CLI cutout**: ~3 seconds (includes model download/loading)

### Model Download Behavior
- Build script automatically downloads ONNX models from GitHub releases during `cargo build`
- Models cached in `~/.cache/onnx-models` or `$ONNX_MODEL_CACHE_DIR`
- Requires internet access during first build
- Uses `GITHUB_TOKEN` environment variable if available (avoids rate limiting)
- Download failures may occur due to firewall restrictions

### Required Test Images
These images must be present for CLI testing:
- `example.jpg` - Single bird image (314KB)
- `example-2-birds.jpg` - Multiple birds image (258KB)

Copy from repository root: `cp ../example*.jpg .`

## Validation Scenarios

### Manual CLI Testing
Always test these scenarios after making changes:

1. **Help and Version Commands**:
   ```bash
   ./target/release/beaker --help
   ./target/release/beaker version
   ```

2. **Head Detection Workflow**:
   ```bash
   # Basic crop output
   ./target/release/beaker head example.jpg --confidence 0.5 --crop

   # Bounding box output
   ./target/release/beaker head example.jpg --confidence 0.5 --bounding-box

   # Metadata generation
   ./target/release/beaker head example.jpg --confidence 0.5 --metadata

   # Multiple outputs
   ./target/release/beaker head example.jpg --confidence 0.5 --crop --bounding-box --metadata

   # Test with two birds
   ./target/release/beaker head example-2-birds.jpg --confidence 0.5 --crop
   ```

3. **Cutout Workflow**:
   ```bash
   # Basic cutout
   ./target/release/beaker cutout example.jpg

   # With alpha matting
   ./target/release/beaker cutout example.jpg --alpha-matting --save-mask

   # Custom background color
   ./target/release/beaker cutout example.jpg --background-color "255,255,255,255"
   ```

4. **Verify Output Files**:
   ```bash
   # Check expected output files exist
   ls -la example_crop.jpg example_cutout.png
   ls -la *.beaker.toml *.beaker.json  # metadata files

   # Clean up test artifacts before committing
   rm -f *.beaker.toml *.beaker.json example_crop.jpg example_cutout.png
   ```

## CI Validation Requirements

The CI runs comprehensive tests on Linux, macOS, and Windows. Ensure changes pass these checks:

### Platform-Specific Behavior
- **macOS**: Uses CoreML execution provider for faster inference
- **Linux/Windows**: Falls back to CPU execution provider
- **All platforms**: Support both `auto` and `cpu` device modes

### Commands CI Runs
```bash
# Formatting and linting
cargo fmt --check
cargo clippy --target <platform> -- -D warnings

# Building
cargo build --target <platform>
cargo build --release --target <platform>

# CLI testing
./target/<platform>/release/beaker --help
./target/<platform>/release/beaker head example.jpg --confidence 0.5 --device auto --metadata
./target/<platform>/release/beaker cutout example.jpg --device auto

# Integration tests
cargo test --release --target <platform> --verbose
```

## Repository Structure

### Key Directories
- `beaker/` - Main Rust CLI codebase (focus here for CLI changes)
- `beaker/src/` - Rust source code
- `beaker/tests/` - Integration tests
- `.github/workflows/` - CI/CD pipeline definitions
- `scripts/` - Utility scripts (line counting, etc.)

### Important Files
- `beaker/Cargo.toml` - Rust project configuration and dependencies
- `beaker/build.rs` - Build script that downloads ONNX models
- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `ruff.toml` - Python linting configuration
- `example.jpg`, `example-2-birds.jpg` - Test images for CLI validation

### Build Artifacts to Ignore
Never commit these files:
- `beaker/target/` - Rust build artifacts
- `*.beaker.toml`, `*.beaker.json` - CLI metadata output
- `*_crop.jpg`, `*_cutout.png` - CLI image output
- Temporary test files

## Common Issues and Solutions

### Build Failures
- **Network/Firewall Issues**: Build script downloads ONNX models. Ensure access to:
  - `api.github.com`
  - `github.com`
  - `objects.githubusercontent.com`
- **Missing Rust**: Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Outdated toolchain**: Run `rustup update`
- **Disk space**: Model caching requires ~100MB free space

### CLI Errors
- **"No outputs requested"**: Always specify `--crop`, `--bounding-box`, or `--metadata`
- **Missing test images**: Copy from repo root: `cp ../example*.jpg .`
- **Model loading failures**: Check internet connectivity and firewall settings
- **Permission errors**: Ensure write access to current directory

### Pre-commit Hook Issues
- **Hook makes changes**: Review with `git diff`, stage changes, commit again
- **Clippy warnings**: Fix manually, then re-run `pre-commit run --all-files`
- **Format issues**: Run `cargo fmt` in `beaker/` directory
- **Emergency bypass**: `git commit --no-verify` (fix issues immediately after)

### Performance Issues
- **Slow builds**: Normal on first build due to dependency compilation
- **Slow tests**: Normal due to ONNX model loading - wait for completion
- **Model download timeouts**: Retry with better internet connection

## Development Workflow

### Typical Change Process
```bash
# 1. Navigate to Rust codebase
cd beaker

# 2. Make your code changes
# ... edit files ...

# 3. Format and fix issues
cargo fmt
cargo clippy --fix --allow-dirty -- -D warnings

# 4. Build and test - NEVER CANCEL: allow full time for completion
cargo build --release  # ~1m 40s
cargo test --release    # ~2m 50s

# 5. Manual CLI validation
cp ../example.jpg .
./target/release/beaker head example.jpg --confidence 0.5 --crop
./target/release/beaker cutout example.jpg

# 6. Update line counts and run pre-commit (from repo root)
cd ..
bash scripts/run_warloc.sh
pre-commit run --all-files

# 7. Clean up test artifacts
cd beaker
rm -f *.beaker.toml *.beaker.json example_crop.jpg example_cutout.png

# 8. Check git status and commit
git status  # verify no unintended files staged
git add .
git commit -m "Your change description"
```

### Code Style Standards
- **Rust**: Use `cargo fmt` for formatting, fix all `cargo clippy` warnings
- **Python**: Use `ruff` for linting and formatting
- **Emojis**: All emojis must go through `color_utils::symbols` functions
- **Error messages**: Include context and actionable suggestions
- **Breaking changes**: Acceptable during development

### File Naming Conventions
- Example test files: Keep `example.jpg`, `example-2-birds.jpg` (used in README)
- Output files: `*_crop.jpg`, `*_cutout.png`, `*.beaker.toml`
- Never commit output files to repository

## Environment Variables

### Build-time
- `GITHUB_TOKEN` - For GitHub API access (avoids rate limiting)
- `ONNX_MODEL_CACHE_DIR` - Custom model cache directory
- `CI` - Detected in CI environments
- `FORCE_DOWNLOAD` - Force model re-download

### Runtime
- `BEAKER_NO_COLOR` - Disable colored output
- `NO_COLOR` - Standard no-color environment variable
- `RUST_LOG` - Control logging verbosity (debug, info, warn, error)

## Required Tools

### Must Have
- Rust 1.70+ (`rustup`, `cargo`)
- Internet access (for model downloads)

### Recommended
- `pre-commit` - Automated code quality checks
- `ruff` - Python linting and formatting
- `python3` - For helper scripts

### Install Commands
```bash
# Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python tools (if needed)
pip install pre-commit ruff

# Pre-commit hooks (in repo root)
pre-commit install
```
