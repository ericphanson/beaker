# Make Integration Tests

This directory contains tests to verify that beaker works correctly with Make's dependency tracking system.

## Files

- `Makefile` - Example Makefile showing how to integrate beaker with Make
- `test_make_integration.sh` - Automated test script that verifies incremental build behavior
- `test.jpg` - Test image (copied from repository root during testing)

## Running Tests

### Manual Testing

```bash
# Run the automated test
./test_make_integration.sh

# Or test manually
make clean
make all            # Initial build
make all            # Should do nothing
touch test.jpg
make all            # Should rebuild both targets
```

### Environment Variables

- `BEAKER_STAMP_DIR` - Override default stamp directory (for testing isolation)

## What is Tested

1. **Initial Build** - Both cutout and detection targets are built from scratch
2. **No-op Rebuild** - Immediate rebuild does nothing (correct incremental behavior)
3. **Input Change** - Touching input image triggers rebuild of both targets
4. **Dependency Files** - Verifies .d files are created with correct dependencies
5. **Stamp Files** - Verifies stamp files are created in isolated directory

## Expected Behavior

- Make should only rebuild targets when their dependencies actually change
- Stamp files provide fine-grained dependency tracking for configuration changes
- Metadata (TOML) files are not included as prerequisites (per design)
- Dependency files (.d) list all real byte-affecting inputs

## Integration with CI

This test can be run in CI to ensure Make compatibility is maintained across changes to the dependency tracking system. See `.github/workflows/make-integration.yml` for the automated CI workflow that runs these tests on every PR.
