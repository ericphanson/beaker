# Beaker: Bird Detection CLI Tool

Always follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.

## Working Effectively

### Bootstrap and Build
Execute these commands in order to set up a working development environment:

```bash
# Repository uses a Cargo workspace structure with:
# - Repository root contains workspace Cargo.toml
# - beaker/ contains the main CLI codebase
# - beaker-stamp/ and beaker-stamp-derive/ contain proc macro crates

# Check Rust toolchain (requires Rust 1.70+)
rustc --version && cargo --version

# Build all workspace members (from repository root)
cargo build

# Build release version - NEVER CANCEL: takes 1 minute 40 seconds. Set timeout to 3+ minutes.
cargo build --release

# Format and lint code (from repository root)
cargo fmt --all
cargo clippy --all-targets --fix --allow-dirty -- -D warnings

# Run tests - NEVER CANCEL: takes 2 minutes 50 seconds. Set timeout to 5+ minutes.
cargo test --release --all
```

### Validation and Testing
Always run these validation steps after making changes:

```bash
# Test CLI help (from repository root)
cargo run --release --bin beaker -- --help

# Copy test image for validation (beaker binary reads from current working directory)
cp example.jpg example-2-birds.jpg ./

# Test basic detect command (requires --crop, --bounding-box, or --metadata)
cargo run --release --bin beaker -- detect example.jpg --confidence 0.5 --crop

# Test metadata generation
cargo run --release --bin beaker -- detect example.jpg --confidence 0.5 --metadata

# Test cutout functionality (takes ~6 seconds for model loading)
cargo run --release --bin beaker -- cutout example.jpg

# Test with two birds image
cargo run --release --bin beaker -- detect example-2-birds.jpg --confidence 0.5 --crop

# Verify output files are created
ls -la example_crop.jpg example_cutout.png *.beaker.toml
```

### Pre-commit and Code Quality
Run before every commit - NEVER CANCEL: takes ~10 seconds:

```bash
# From repository root
pre-commit run --all-files

# Individual checks if pre-commit unavailable:
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo build --release --all
cargo test --release --all
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
- **Pre-commit hooks**: ~10 seconds
- **CLI detection**: ~0.5 seconds (after initial model load)
- **CLI cutout**: ~6 seconds (includes model download/loading)

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
   cargo run --release -- --help
   cargo run --release -- version
   ```

2. **Detection Workflow**:
   ```bash
   # Basic crop output
   cargo run --release -- detect example.jpg --confidence 0.5 --crop

   # Bounding box output
   cargo run --release -- detect example.jpg --confidence 0.5 --bounding-box

   # Metadata generation
   cargo run --release -- detect example.jpg --confidence 0.5 --metadata

   # Multiple outputs
   cargo run --release -- detect example.jpg --confidence 0.5 --crop --bounding-box --metadata

   # Test with two birds
   cargo run --release -- detect example-2-birds.jpg --confidence 0.5 --crop
   ```

3. **Cutout Workflow**:
   ```bash
   # Basic cutout
   cargo run --release -- cutout example.jpg

   # With alpha matting
   cargo run --release -- cutout example.jpg --alpha-matting --save-mask

   # Custom background color
   cargo run --release -- cutout example.jpg --background-color "255,255,255,255"
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
# Formatting and linting (from repository root)
cargo fmt --all --check
cargo clippy --all-targets --target <platform> -- -D warnings

# Building (from repository root)
cargo build --target <platform>
cargo build --release --target <platform>

# CLI testing (binary located in target/<platform>/release/)
./target/<platform>/release/beaker --help
./target/<platform>/release/beaker detect example.jpg --confidence 0.5 --device auto --metadata
./target/<platform>/release/beaker cutout example.jpg --device auto

# Integration tests
cargo test --release --all --target <platform> --verbose
```

## Repository Structure

### Key Directories
- **Root**: Cargo workspace with top-level `Cargo.toml`
- `beaker/` - Main Rust CLI codebase (focus here for CLI changes)
- `beaker/src/` - Rust source code
- `beaker/tests/` - Integration tests
- `beaker-stamp/` - Core stamp traits and utilities (proc macro support)
- `beaker-stamp-derive/` - Procedural macro for automatic stamp generation
- `.github/workflows/` - CI/CD pipeline definitions
- `scripts/` - Utility scripts (line counting, etc.)

### Important Files
- **Root**: `Cargo.toml` - Workspace configuration and shared dependencies
- `beaker/Cargo.toml` - Main CLI crate configuration and dependencies
- `beaker-stamp/Cargo.toml` - Core stamp crate configuration
- `beaker-stamp-derive/Cargo.toml` - Proc macro crate configuration
- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `ruff.toml` - Python linting configuration
- `example.jpg`, `example-2-birds.jpg` - Test images for CLI validation

### Build Artifacts to Ignore
Never commit these files:
- `target/` - Rust build artifacts (workspace level)
- `beaker/target/` - Legacy target directory (if present)
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

**Pre-commit integration**: Commits will automatically run pre-commit hooks. If hooks make changes or find issues, the commit will be blocked until you stage the changes and try again.

**When pre-commit makes changes during commit:**
- The commit will be blocked with a message like "files were modified by this hook"
- Review the changes made: `git diff`
- Stage the fixed files: `git add .`
- Commit again: `git commit -m "your message"`

**When pre-commit finds unfixable issues:**
- Read the error output carefully to understand what needs fixing
- Fix the issues manually (e.g., resolve clippy warnings, fix YAML syntax)
- Stage your fixes: `git add .`
- Try committing again: `git commit -m "your message"`

**Common scenarios and solutions:**
- **Rust formatting/clippy issues**: Run `cargo fmt --all` and `cargo clippy --all-targets --fix --allow-dirty` from repository root
- **Python formatting/linting**: Run `ruff format --config=ruff.toml .` and `ruff check --config=ruff.toml --fix .`
- **Line counting updates**: The `cargo-warloc` hook updates `beaker/line_counts.md` automatically
- **YAML syntax errors**: Fix the YAML file syntax manually
- **Large files**: Remove or use Git LFS for files over the size limit

**Emergency bypass (use sparingly):**
If pre-commit is blocking critical work and fixes aren't immediately obvious:
```bash
git commit -m "your message" --no-verify
```
Then immediately follow up with a commit that fixes the pre-commit issues.

### Performance Issues
- **Slow builds**: Normal on first build due to dependency compilation
- **Slow tests**: Normal due to ONNX model loading - wait for completion
- **Model download timeouts**: Retry with better internet connection

## Development Workflow

### Typical Change Process
```bash
# 1. Repository uses Cargo workspace structure
# All commands run from repository root unless specified

# 2. Make your code changes
# ... edit files in beaker/, beaker-stamp/, or beaker-stamp-derive/ ...

# 3. Format and fix issues
cargo fmt --all
cargo clippy --all-targets --fix --allow-dirty -- -D warnings

# 4. Build and test - NEVER CANCEL: allow full time for completion
cargo build --release --all  # ~1m 40s
cargo test --release --all    # ~2m 50s

# 5. Manual CLI validation
cp example.jpg example-2-birds.jpg ./
cargo run --release --bin beaker -- detect example.jpg --confidence 0.5 --crop
cargo run --release --bin beaker -- cutout example.jpg

# 6. Update line counts and run pre-commit
bash scripts/run_warloc.sh
pre-commit run --all-files

# 7. Clean up test artifacts
rm -f *.beaker.toml *.beaker.json example_crop.jpg example_cutout.png

# 8. Check git status and commit
git status  # verify no unintended files staged
git add .
git commit -m "Your change description"

# 9. Push changes using report_progress tool
# Use report_progress to push all commits to the remote PR branch
# If it says "Error committing and pushing changes", resolve pre-commit issues and try again
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

## Pushing Changes and PR Management

### Pushing Changes Workflow
**Agents must push their changes before finishing work.** After making commits locally:
- Use `report_progress` to push changes to the remote PR branch
- If you have conflicts or other issues ("Error committing and pushing changes"), resolve them and then use `report_progress` again to push to the remote PR branch
- Never leave unpushed local commits - they will be lost

### Git Commit Troubleshooting
**If `git commit` fails due to pre-commit:**
1. **Fix the issues**: Address the problems identified by pre-commit hooks
2. **Stage the fixes**: Run `git add .` to stage your corrections
3. **Try committing again**: `git commit -m "your message"`
4. **If still failing**: Review the error output and repeat steps 1-3
5. **Emergency bypass only**: If fixes aren't obvious and you need to push urgently, use `git commit -m "your message" --no-verify`, then immediately create a follow-up commit with the fixes

**Always push your work**: Use `report_progress` to push all commits to the remote PR branch before finishing.

### PR Conflict Resolution
If your PR has conflicts with the main branch, merge the latest origin/main into the PR branch, resolving any conflicts. Then use your `report_progress` tool to push your changes to the PR branch. If it says "Error committing and pushing changes" then it did not work. Likely pre-commit blocked you. You need to resolve the pre-commit issues and try `report_progress` again.

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
