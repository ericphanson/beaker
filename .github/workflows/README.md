# CI/CD Workflows

This directory contains GitHub Actions workflows for the bird-head-detector project.

## Workflows

### `test.yml` - Basic Test Suite
- **Triggers**: Push/PR to main branch
- **Platform**: Ubuntu only
- **Python**: 3.12 only
- **Purpose**: Quick feedback on basic functionality
- **Actions**:
  - Install uv and dependencies
  - Build package
  - Run end-to-end tests
  - Upload debug artifacts on failure

### `comprehensive-test.yml` - Full Test Matrix
- **Triggers**: Push/PR to main branch, manual dispatch
- **Platforms**: Ubuntu, macOS, Windows
- **Python**: 3.10, 3.11, 3.12 (with some exclusions)
- **Purpose**: Comprehensive cross-platform testing
- **Jobs**:
  - **test**: Full test matrix with tool installation verification
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
- Distribution packages uploaded from main job (30-day retention)

## Usage

The workflows run automatically on push/PR to main. You can also manually trigger the comprehensive test suite:

1. Go to the Actions tab in GitHub
2. Select "Comprehensive Test Suite"
3. Click "Run workflow"

## Local Testing

To run the same tests locally:

```bash
# Basic test
uv run python test/run_tests.py

# With linting
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .

# Security audit
uv run pip-audit --desc
```
