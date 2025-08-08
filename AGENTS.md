> Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all his sentences short, or that he avoid all detail and treat his subjects only in outline, but that he make every word tell.

— "Elementary Principles of Composition", _The Elements of Style_

# Agent Development Guide

This document outlines the checks and procedures that must pass before committing changes to ensure PRs pass CI on the first attempt.

## Focus Areas

- if the `beaker` CLI app or rust code is mentioned, then work should focus on the `beaker/` subdirectory, which contains the main Rust codebase. Likewise, `beaker-ci.yml` is the github workflow to look for.
- if python code or training is mentioned, these happen in separate top-level environments, look for changes there. New models may be tested against the CLI app in which case both directories may be useful.

## API Tool Usage Guidelines

### GitHub API vs Browser Tools

When accessing GitHub information, **always prioritize GitHub API tools over browser automation** when available:

**Use GitHub API tools for:**
- GitHub Actions workflows and logs (`github.com/.../actions/runs/...`)
- Pull requests and their details (`github.com/.../pull/...`)
- Issues and comments (`github.com/.../issues/...`)
- Repository contents, commits, and security alerts

**Only use browser tools (like Playwright) when:**
- The required information is not available via GitHub API
- Working with non-GitHub websites
- Testing actual user interface functionality

## Pre-Commit Setup

Tools like `cargo`, `pre-commit` and `ruff` should already be installed in your dev environment.

### Manual Execution
Run all pre-commit hooks manually before committing:
```bash
pre-commit run --all-files
```

Run specific hooks:
```bash
pre-commit run ruff --all-files          # Python linting
pre-commit run ruff-format --all-files   # Python formatting
pre-commit run clippy-cpu --all-files    # Rust linting
pre-commit run fmt --all-files           # Rust formatting
```

**Note**: If pre-commit is not available in your environment, you can run the individual tools manually as documented in the sections below.

## Required Checks

### Rust Code (beaker/ directory)

All Rust code **MUST** pass these checks:

#### 1. Code Formatting
```bash
cd beaker
cargo fmt --check
```
Fix formatting issues:
```bash
cargo fmt
```

#### 2. Linting (Clippy)
```bash
cd beaker
cargo clippy -- -D warnings -D clippy::uninlined_format_args
```
This treats all warnings as errors. Fix all clippy warnings before committing.

#### 3. Build (Debug & Release)
```bash
cd beaker
cargo build
cargo build --release
```

#### 4. Tests
```bash
cd beaker
cargo test
cargo test --release
```

#### 5. Line Count Analysis
```bash
# Install cargo-warloc if not available
cargo install cargo-warloc

# Run from repository root
bash scripts/run_warloc.sh
```
This updates `beaker/line_counts.md` with current code statistics.

**Note**: If cargo-warloc is not available, you can skip this step as it will be run automatically by the pre-commit hook.

### Python Code

Python code uses Ruff for linting and formatting:

#### 1. Linting
```bash
# Install ruff if not available (check with `which ruff`)
pip install ruff

# Run linting - `ruff.toml` is a top-level in repo so the path may need to be modified
ruff check --config=ruff.toml --fix .
```

#### 2. Formatting
```bash
ruff format --config=ruff.toml .
```

**Note**: Some legacy Python code may not pass all checks. This is acceptable, but new Python code should follow current standards. If you must skip ruff checks to be able to commit, execute `skip=RUFF git commit -m "..."`.

### General File Checks

These apply to all files:
- No trailing whitespace (except in `beaker/line_counts.md`)
- Proper end-of-file handling
- Valid YAML syntax
- No large files added
- No merge conflict markers

## CI Workflow Validation

The CI runs on multiple platforms (Linux, macOS, Windows) and architectures. Ensure your changes work across platforms:

### Platform-Specific Considerations
- **macOS**: Uses CoreML execution provider
- **Linux/Windows**: Falls back to CPU execution
- **Cross-compilation**: Currently disabled due to ONNX Runtime complexity

### CLI Testing
The CI tests basic CLI functionality:
```bash
cd beaker
# Copy test image
cp ../example.jpg .

# Test help
./target/release/beaker --help

# Test head detection
./target/release/beaker head example.jpg --confidence 0.5 --device auto
```

## Coding Standards

### DRY (Don't Repeat Yourself)
- Extract common functionality into reusable functions/modules
- Avoid duplicating code across files
- Use configuration files for repeated values
- **Implement functions ergonomically**: When possible, implement path-based functions via byte-based versions (e.g., `calculate_md5(path)` via `calculate_md5_bytes()`)

### YAGNI (You Aren't Gonna Need It)
- Implement only what's currently needed
- Avoid over-engineering solutions
- Keep interfaces simple and focused
- **Inline trivial wrappers**: Replace 1-line wrapper functions with direct calls to avoid unnecessary indirection

### Code Style Standards for Rust code
- **Emoji handling**: All emojis must go through `color_utils::symbols` functions to respect no-color settings
- **Consistent formatting**: Use `cargo fmt` for automatic code formatting
- **Clear error messages**: Include context and suggestions in error messages

### Good PR Characteristics
- **Small diff**: Minimize the number of changed lines
- **Simple and targeted**: Address one specific issue per PR
- **Breaking changes are acceptable**: Don't prioritize backwards compatibility at this development stage
- **No test artifacts**: Do not commit test output files (*.beaker.toml, *.beaker.json), temporary files, or build artifacts

## Complete Pre-Commit Checklist

Before committing, ensure all these pass:

### Automated Checks
- [ ] `cargo fmt --check` (in beaker/)
- [ ] `cargo clippy -- -D warnings` (in beaker/)
- [ ] `cargo build` (in beaker/)
- [ ] `cargo build --release` (in beaker/)
- [ ] `cargo test --release` (in beaker/)
- [ ] `ruff check --fix` (Python files)
- [ ] `ruff format` (Python files)
- [ ] No trailing whitespace
- [ ] Proper end-of-file handling
- [ ] Valid YAML files
- [ ] No large files
- [ ] No merge conflicts
- [ ] After staging files (`git add`) but before committing, run `pre-commit run` to auto-check the staged files. If there are errors, fix them and stage the files again before committing.

### Manual Verification
- [ ] CLI help works: `./target/release/beaker --help`
- [ ] Basic functionality: `./target/release/beaker head example.jpg --confidence 0.5`
- [ ] Line counts updated: `bash scripts/run_warloc.sh`
- [ ] **Check git status**: Verify no unintended files are staged (test artifacts, temporary files, etc.)

### Final Check for rust changes

```bash
# If pre-commit is available, run everything at once
pre-commit run --all-files

# If pre-commit is not available, run individual checks:
cd beaker
cargo fmt --check
cargo clippy -- -D warnings
cargo build --release
cargo test --release
cd ..
# Run Python checks if ruff is available
ruff check --config=ruff.toml . || echo "Ruff not available, skipping Python checks"
```

## Quick Start Command Sequence (for rust / beaker changes)

For a typical development session:

```bash
# 1. Navigate to beaker directory
cd beaker

# 2. Make your changes
# ... edit files ...

# 3. Format and fix issues
cargo fmt
cargo clippy --fix --allow-dirty -- -D warnings

# 4. Build and test
cargo build --release
cargo test --release

# 5. Test CLI
cp ../example.jpg .
./target/release/beaker head example.jpg --confidence 0.5

# 6. Update line counts (from repo root)
cd ..
bash scripts/run_warloc.sh

# 7. Run all pre-commit checks
pre-commit run --all-files

# 8. Check git status and clean up unintended files
git status
# Remove any test artifacts or temporary files before committing:
# git rm *.beaker.toml *.beaker.json (if accidentally added)

# 9. Commit your changes
git add .
git commit -m "Your commit message"
```

## Troubleshooting

### Network and Firewall Issues

#### Common Issues
- ONNX model downloads may fail due to firewall restrictions
- GitHub API access may be blocked
- Certificate validation errors in restricted environments

#### Solutions
If you encounter download or connection issues:
1. **Request allowlisting** for these URLs:
   - `api.github.com` (for GitHub API access)
   - `github.com` (for repository access)
   - `objects.githubusercontent.com` (for asset downloads)
   - Any ONNX Runtime download URLs

2. **Common Error Messages**:
   - `tls connection init failed`
   - `InvalidCertificate(UnknownIssuer)`
   - `ConnectionFailed`
   - `Transport { kind: ConnectionFailed }`

3. **Workarounds**:
   - Some commands like `cargo fmt --check` and `cargo clippy` may work without network access
   - Use pre-built binaries if available
   - Work in an environment with proper network access

Contact your network administrator to allowlist the required URLs.

### Build Failures
- **Network Issues**: The build script downloads ONNX models from GitHub. If builds fail with network/firewall errors, see the Network and Firewall Issues section above.
- Check Rust toolchain is up to date: `rustup update`
- Clear target directory: `cargo clean` then rebuild
- Set offline mode temporarily: `cargo build --offline` (may not work if dependencies are missing)

### Test Failures
- Ensure test images are present (`example.jpg`, `example-2-birds.jpg`)
- Check available disk space for model caching
- Verify ONNX_MODEL_CACHE_DIR environment variable if set

### Pre-commit Issues
- Reinstall hooks: `pre-commit clean && pre-commit install`
- Update hooks: `pre-commit autoupdate`
- Run specific hook: `pre-commit run <hook-name> --all-files`

### PR Conflicts
If your PR has conflicts with the main branch, merge the latest origin/main into the PR branch, resolving any conflicts. Then push to the PR branch.

### Pushing Changes
**Agents must push their changes before finishing work.** After making commits locally:
- Use `report_progress` to push changes to the remote PR branch
- If you have conflicts or other issues, resolve them and push your work to the remote PR branch
- Never leave unpushed local commits - they will be lost
- if `git commit` fails due to precommit, be sure to try to fix the issues, stage the files again `git add`, then try committing and pushing again. If this fails, you can skip pre-commit with `git commit -m "my message" --no-verify`. Then be sure to push your changes.

Remember: **The goal is PRs that pass CI on the first attempt.**
