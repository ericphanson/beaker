> Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all his sentences short, or that he avoid all detail and treat his subjects only in outline, but that he make every word tell.

— "Elementary Principles of Composition", _The Elements of Style_

# Agent Development Guide

This document provides development philosophy, planning guidance, and detailed troubleshooting for agents working in the Beaker codebase.

**For quick, actionable instructions and validated commands, see [`.github/copilot-instructions.md`](.github/copilot-instructions.md).** This guide focuses on decision-making, standards, and complex problem resolution.

## Focus Areas

- **Rust CLI (`beaker/`)**: Focus on the `beaker/` subdirectory for CLI app changes. Check `beaker-ci.yml` workflow.
- **Python/Training**: Separate top-level environments. New models test against CLI requiring both directories.

## Proposal Planning & Technical Decision Making

When developing proposals, technical plans, or architectural changes, align with these core principles:

### Evidence-Based Development
- **Back all performance claims with real benchmarks** - no made-up numbers
- **Measure before optimizing** - implement timing infrastructure and collect actual data
- **Script reproducible benchmarks** - provide scripts for others to validate claims

### Simplicity and Incrementalism
- **Start simple** - especially when evidence doesn't point to large benefits from complexity
- **Eliminate redundancy** - consolidate similar functions and remove code duplication aggressively
- **Use generic solutions** - prefer reusable implementations over model-specific code
- **Accept breaking changes** - don't prioritize backwards compatibility during development

### Ergonomics Over Performance
- **Prioritize user experience** - when performance gains are modest (< 5%), choose user ergonomics
- **Separate concerns cleanly** - APIs should make sense without requiring codebase-wide context
- **Use configuration-based coordination** - expand existing config mechanisms for cross-API communication

### Parallel Development Planning
- **Structure work into orthogonal issues** - independent, well-scoped GitHub issues for concurrent execution
- **Document dependencies explicitly** - ensure each maintains working test state
- **Phase conservatively** - move complex optimizations to later phases

### Technical Debt Reduction
- **Unify tooling** - consolidate multiple scripts with similar purposes
- **Apply consistent patterns** - use same architectural patterns across similar components
- **Design maintainable interfaces** - APIs that automatically benefit from future enhancements

## API Tool Usage Guidelines

**Always prioritize GitHub API tools over browser automation** when available:

**Use GitHub API tools for:**
- GitHub Actions workflows and logs
- Pull requests, issues, and comments
- Repository contents, commits, and security alerts

**Only use browser tools when:**
- Required information unavailable via GitHub API
- Testing actual user interface functionality
- Working with non-GitHub websites

## Development Workflow

**IMPORTANT: Always prefer `just` commands over raw cargo/bash commands.** The justfile centralizes all common development tasks with consistent behavior across local development and CI.

**For complete build/test commands and validated timings, see [copilot-instructions.md](.github/copilot-instructions.md).**

**For pre-commit troubleshooting and pushing changes, see [copilot-instructions.md](.github/copilot-instructions.md).**

### Quick Reference

```bash
# Install just (required for development)
cargo install just

# Common development tasks
just lint           # Format check + clippy
just fmt            # Auto-format code
just build-release  # Build release binary
just test           # Run all tests
just ci             # Full CI workflow locally

# See all available commands
just --list
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
- **Network Issues**: Build script downloads ONNX models. For network/firewall errors, see Network and Firewall Issues above.
- **Update toolchain**: `rustup update`
- **Clear build cache**: `cargo clean` then rebuild
- **Try offline mode**: `cargo build --offline` (may fail if dependencies missing)

### Test Failures
- **Check test images**: Ensure `example.jpg`, `example-2-birds.jpg` are present
- **Check disk space**: Model caching requires ~100MB
- **Verify environment**: Check `ONNX_MODEL_CACHE_DIR` if set

### Pre-commit Issues

**For detailed pre-commit troubleshooting, see [copilot-instructions.md](.github/copilot-instructions.md).**

Brief summary:
- **Emergency bypass**: Use `git commit --no-verify` sparingly, then immediately fix issues
- **Common fixes**: Run `cargo fmt`, `cargo clippy --fix --allow-dirty`, `ruff format`, `ruff check --fix`
- **Reinstall hooks**: `pre-commit clean && pre-commit install`

### PR Conflicts and Pushing Changes

**For complete pushing workflow and troubleshooting, see [copilot-instructions.md](.github/copilot-instructions.md).**

**Key points:**
- **Always push your work**: Use `report_progress` before finishing
- **Resolve conflicts**: Merge `origin/main`, fix conflicts, use `report_progress` again
- **Fix commit failures**: Address pre-commit issues, then retry `report_progress`
- **Emergency bypass**: `git commit --no-verify` only when urgent, then fix immediately

Remember: **The goal is PRs that pass CI on the first attempt.**
