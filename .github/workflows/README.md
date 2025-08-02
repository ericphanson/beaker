# CI/CD Workflows

This directory contains the GitHub Actions workflow for the bird-head-detector project.

## Workflow

### `ci.yml` - Complete Test Suite
- **Triggers**: Push/PR to main branch, manual dispatch
- **Platforms**: Ubuntu, macOS, Windows
- **Python**: 3.10, 3.11, 3.12 (with some Windows exclusions for efficiency)
- **Jobs**:
  - **test**: Cross-platform testing with tool installation verification
  - **lint**: Code style and formatting checks with ruff
  - **security**: Security audit with pip-audit

## Test Features

### Enhanced Debugging
Tests include comprehensive debug information on failure:
- Command output (stdout/stderr)
- File system state
- Tool installation status
- System environment details
- Directory contents with file sizes

### Artifacts
- Test debug artifacts uploaded on failure (7-day retention)

## Usage

The workflow runs automatically on push/PR to main. You can also manually trigger it:

1. Go to the Actions tab in GitHub
2. Select "Test Suite"
3. Click "Run workflow"

## Local Testing

To run the same tests locally:

```bash
# End-to-end tests
uv run python test/run_tests.py

# Linting
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .

# Security audit
uv run pip-audit --desc
```
