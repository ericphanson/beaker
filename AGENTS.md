> Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all his sentences short, or that he avoid all detail and treat his subjects only in outline, but that he make every word tell.

— "Elementary Principles of Composition", _The Elements of Style_

# Agent Development Guide

This document outlines the checks and procedures that must pass before committing changes to ensure PRs pass CI on the first attempt.

## Focus Areas

- if the `beaker` CLI app or rust code is mentioned, then work should focus on the `beaker/` subdirectory, which contains the main Rust codebase. Likewise, `beaker-ci.yml` is the github workflow to look for.
- if python code or training is mentioned, these happen in separate top-level environments, look for changes there. New models may be tested against the CLI app in which case both directories may be useful.

## Proposal Planning & Technical Decision Making

When developing proposals, technical plans, or architectural changes, align with these core principles:

### Evidence-Based Development
- **No made-up numbers**: All performance claims must be backed by real benchmarks and measured data
- **Measure before optimizing**: Implement timing infrastructure and collect actual data before making optimization decisions
- **Script reproducible benchmarks**: Provide scripts that others can run to validate performance claims

### Simplicity and Incrementalism
- **Start simple**: especially if the evidence doesn't point to large benefits from a more complex approach
- **Eliminate redundancy**: Consolidate similar functions and remove code duplication aggressively
- **Generic solutions**: Prefer reusable, generic implementations over model-specific or use-case-specific code
- **Breaking changes are acceptable**: Don't prioritize backwards compatibility at this development stage

### Ergonomics Over Performance
- **User experience first**: When performance gains are modest (< 5%), prioritize user ergonomics and workflow simplification
- **Clean separation of concerns**: APIs should make sense on their own without requiring codebase-wide context. For example, models should be generally agnostic to their usage context (pipeline vs standalone).
- **Configuration-based coordination**: Use and expand upon existing config mechanisms to communicate across APIs and serialize those configs to metadata for testing and analysis.

### Parallel Development Planning
- **Orthogonal issues**: Structure work into independent, well-scoped GitHub issues that can be executed concurrently
- **Clear dependencies**: Explicitly document issue dependencies and ensure each maintains working test state
- **Conservative phasing**: Move complex optimizations (memory management, performance tuning) to later phases

### Technical Debt Reduction
- **Unify tooling**: Consolidate multiple scripts with similar purposes into single, comprehensive tools
- **Consistent patterns**: Apply the same architectural patterns across similar components
- **Maintainable interfaces**: Design APIs that automatically benefit from future enhancements

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

### Complete Pre-Commit Checklist

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
- [ ] **Pre-commit integration**: Commits will automatically run pre-commit hooks. If hooks make changes or find issues, the commit will be blocked until you stage the changes and try again.
- [ ] Then use your `report_progress` tool to push your changes to the PR branch. If it says "Error committing and pushing changes" then it did not work. Likely pre-commit blocked you. You need to resolve the pre-commit issues and try `report_progress` again.

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

### Quick Start Command Sequence (for rust / beaker changes)

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

# 10. Push your changes up with your `report_progress` tool! They must reach the PR. If it says there was an error, you  must fix it and try again.
```

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
- **Preserve example files**: Do not delete `example-no-bg.png` and `example_crop.jpg` as they are used in README and documentation
- **No test artifacts**: Do not commit test output files (*.beaker.toml, *.beaker.json), temporary files, or build artifacts

### Test Environment Variable Isolation
Environment variable functionality is tested using integration tests with subprocess execution, which avoids race conditions by setting environment variables only on subprocesses rather than modifying the global process environment.

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

#### When Pre-commit Makes Changes or Blocks Commits

Pre-commit hooks automatically fix many issues but may require manual intervention:

**If pre-commit makes changes during commit:**
1. The commit will be blocked with a message like "files were modified by this hook"
2. Review the changes made: `git diff`
3. Stage the fixed files: `git add .`
4. Commit again: `git commit -m "your message"`

**If pre-commit finds unfixable issues:**
1. Read the error output carefully to understand what needs fixing
2. Fix the issues manually (e.g., resolve clippy warnings, fix YAML syntax)
3. Stage your fixes: `git add .`
4. Try committing again: `git commit -m "your message"`

**Common scenarios and solutions:**
- **Rust formatting/clippy issues**: Run `cargo fmt` and `cargo clippy --fix --allow-dirty` in the `beaker/` directory
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

**Troubleshooting pre-commit itself:**
- Reinstall hooks: `pre-commit clean && pre-commit install`
- Update hooks: `pre-commit autoupdate`
- Run specific hook: `pre-commit run <hook-name> --all-files`
- Check hook status: `pre-commit run --all-files` (shows which hooks pass/fail)

### PR Conflicts
If your PR has conflicts with the main branch, merge the latest origin/main into the PR branch, resolving any conflicts. Then use your `report_progress` tool to push your changes to the PR branch. If it says "Error committing and pushing changes" then it did not work. Likely pre-commit blocked you. You need to resolve the pre-commit issues and try `report_progress` again.

### Pushing Changes
**Agents must push their changes before finishing work.** After making commits locally:
- Use `report_progress` to push changes to the remote PR branch
- If you have conflicts or other issues ("Error committing and pushing changes"), resolve them and then use `report_progress` again to push to the remote PR branch
- Never leave unpushed local commits - they will be lost

**If `git commit` fails due to pre-commit:**
1. **Fix the issues**: Address the problems identified by pre-commit hooks
2. **Stage the fixes**: Run `git add .` to stage your corrections
3. **Try committing again**: `git commit -m "your message"`
4. **If still failing**: Review the error output and repeat steps 1-3
5. **Emergency bypass only**: If fixes aren't obvious and you need to push urgently, use `git commit -m "your message" --no-verify`, then immediately create a follow-up commit with the fixes

**Always push your work**: Use `report_progress` to push all commits to the remote PR branch before finishing.

Remember: **The goal is PRs that pass CI on the first attempt.**
