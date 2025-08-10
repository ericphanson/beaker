#!/bin/bash
# Integration test for Make compatibility
# Tests that beaker works correctly with Make's dependency tracking

set -e  # Exit on any error

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IMG="$TEST_DIR/test.jpg"
CUTOUT_OUTPUT="$TEST_DIR/test_cutout.png"
DETECT_OUTPUT="$TEST_DIR/test_crop.jpg"

# Use isolated stamp directory for testing (concurrency safe)
export BEAKER_STAMP_DIR="$TEST_DIR/stamps"

echo "=== Make Integration Test ==="
echo "Test directory: $TEST_DIR"
echo "Stamp directory: $BEAKER_STAMP_DIR"

# Change to test directory
cd "$TEST_DIR"

# Copy test image from repository root
if [ ! -f "$TEST_IMG" ]; then
    echo "Copying test image..."
    cp "../../../example.jpg" "$TEST_IMG"
fi

# Clean start
echo "Cleaning up previous test artifacts..."
make clean
rm -rf "$BEAKER_STAMP_DIR"

echo ""
echo "=== Test 1: Initial build (should build both targets) ==="
time make all

if [ ! -f "$CUTOUT_OUTPUT" ]; then
    echo "ERROR: Cutout output not created"
    exit 1
fi

if [ ! -f "$DETECT_OUTPUT" ]; then
    echo "ERROR: Detection output not created"
    exit 1
fi

echo "✅ Initial build successful"

echo ""
echo "=== Test 2: Immediate rebuild (should do nothing) ==="
OUTPUT=$(make all 2>&1)
if echo "$OUTPUT" | grep -q "is up to date"; then
    echo "✅ Make correctly detected no rebuild needed"
elif echo "$OUTPUT" | grep -q "Nothing to be done"; then
    echo "✅ Make correctly detected no rebuild needed"
else
    echo "⚠️  Make output: $OUTPUT"
    echo "Expected no rebuild, but got different output"
fi

echo ""
echo "=== Test 3: Touch input (should rebuild both) ==="
touch "$TEST_IMG"
time make all

echo "✅ Rebuild after input change successful"

echo ""
echo "=== Test 4: Check dependency files ==="
if [ -f "${CUTOUT_OUTPUT}.d" ]; then
    echo "✅ Cutout dependency file created"
    echo "Dependencies: $(cat "${CUTOUT_OUTPUT}.d" | head -1)"
else
    echo "❌ Cutout dependency file missing"
    exit 1
fi

if [ -f "${DETECT_OUTPUT}.d" ]; then
    echo "✅ Detection dependency file created"
    echo "Dependencies: $(cat "${DETECT_OUTPUT}.d" | head -1)"
else
    echo "❌ Detection dependency file missing"
    exit 1
fi

echo ""
echo "=== Test 5: Check stamp files ==="
STAMP_COUNT=$(find "$BEAKER_STAMP_DIR" -name "*.stamp" | wc -l)
echo "Found $STAMP_COUNT stamp files in $BEAKER_STAMP_DIR"

if [ "$STAMP_COUNT" -gt 0 ]; then
    echo "✅ Stamp files created successfully"
    find "$BEAKER_STAMP_DIR" -name "*.stamp" | head -3 | while read stamp; do
        echo "  $(basename "$stamp"): $(cat "$stamp")"
    done
else
    echo "❌ No stamp files found"
    exit 1
fi

echo ""
echo "=== Cleaning up ==="
make clean
rm -rf "$BEAKER_STAMP_DIR"
rm -f "$TEST_IMG"

echo "✅ All Make integration tests passed!"
