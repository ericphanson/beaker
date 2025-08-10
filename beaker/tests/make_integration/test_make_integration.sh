#!/bin/bash
# Clean Make Integration Test
# Tests core dependency tracking functionality with minimal noise

set -e  # Exit on any error

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAKER="../../../target/release/beaker"

# Use isolated stamp directory for testing
export BEAKER_STAMP_DIR="$TEST_DIR/test_stamps"

echo "=== Make Integration Test ==="

# Utility functions
fail() {
    echo "❌ FAIL: $1"
    exit 1
}

pass() {
    echo "✅ PASS: $1"
}

# Check if file exists and is not empty
check_file_exists() {
    local file="$1"
    local desc="$2"
    [ -f "$file" ] && [ -s "$file" ] || fail "$desc: $file missing or empty"
    pass "$desc: $file exists"
}

# Check Make says target is up to date
check_up_to_date() {
    local target="$1"
    local output
    output=$(make "$target" 2>&1)
    if echo "$output" | grep -q "is up to date\|Nothing to be done"; then
        pass "No rebuild: $target"
    else
        fail "Unexpected rebuild: $target (output: $output)"
    fi
}

# Check Make rebuilds target
check_rebuilds() {
    local target="$1"
    local output
    output=$(make "$target" 2>&1)
    if echo "$output" | grep -q "Building\|Compiling"; then
        pass "Rebuilds correctly: $target"
    else
        fail "Should have rebuilt: $target (output: $output)"
    fi
}

# Setup
cd "$TEST_DIR"

# Verify beaker binary
[ -x "$BEAKER" ] || fail "Beaker binary not found or not executable: $BEAKER"
"$BEAKER" --help >/dev/null 2>&1 || fail "Beaker binary doesn't work"
pass "Beaker binary verification"

# Copy test images
cp "../../../example.jpg" "example.jpg"
cp "../../../example-2-birds.jpg" "example-2-birds.jpg"

# Clean start
make clean >/dev/null 2>&1
rm -rf "$BEAKER_STAMP_DIR"

echo ""
echo "=== Core Dependency Tracking Tests ==="

# Test 1: Initial build creates expected files
echo "Test 1: Initial build"
make all >/dev/null
check_file_exists "example_cutout.png" "Cutout output"
check_file_exists "example_bounding-box.jpg" "Detection output"
check_file_exists "example_cutout.png.d" "Cutout depfile"
check_file_exists "example_bounding-box.jpg.d" "Detection depfile"

# Test 2: No rebuild when nothing changed
echo ""
echo "Test 2: No spurious rebuilds"
check_up_to_date "example_cutout.png"
check_up_to_date "example_bounding-box.jpg"
check_up_to_date "all"

# Test 3: Rebuild when input changes
echo ""
echo "Test 3: Rebuild after input change"
touch "example.jpg"
check_rebuilds "example_cutout.png"

# Test 4: Verify depfile content format
echo ""
echo "Test 4: Depfile content validation"
# Check that depfiles have correct format: target: dependencies
for depfile in example_cutout.png.d example_bounding-box.jpg.d; do
    if grep -q "^[^:]*: .*example\.jpg" "$depfile"; then
        pass "Depfile format: $depfile"
    else
        fail "Invalid depfile format: $depfile (content: $(cat $depfile))"
    fi
done

# Test 5: Stamp files created
echo ""
echo "Test 5: Stamp file generation"
stamp_count=$(find "$BEAKER_STAMP_DIR" -name "*.stamp" 2>/dev/null | wc -l)
[ "$stamp_count" -gt 0 ] || fail "No stamp files found in $BEAKER_STAMP_DIR"
pass "Stamp files created ($stamp_count files)"

# Test 6: Cross-tool dependency isolation
echo ""
echo "Test 6: Cross-tool dependency isolation"
make clean >/dev/null
make example_bounding-box.jpg >/dev/null  # Build detect first
make example_cutout.png >/dev/null        # Then cutout
# Detect should not rebuild when we request it again
check_up_to_date "example_bounding-box.jpg"

# Test 7: Verify specific targets work independently
echo ""
echo "Test 7: Individual target builds"
make clean >/dev/null
make example_crop.jpg >/dev/null
check_file_exists "example_crop.jpg" "Crop target output"
check_file_exists "example_crop.jpg.d" "Crop target depfile"
check_up_to_date "example_crop.jpg"

echo ""
echo "=== Advanced Dependency Tests ==="

# Test 8: Stamp file changes trigger rebuilds
echo "Test 8: Stamp file dependency tracking"
make clean >/dev/null
make example_cutout.png >/dev/null
# Find a stamp file and touch it
stamp_file=$(find "$BEAKER_STAMP_DIR" -name "*.stamp" | head -1)
if [ -f "$stamp_file" ]; then
    sleep 1  # Ensure timestamp difference
    touch "$stamp_file"
    check_rebuilds "example_cutout.png"
    pass "Stamp file dependency tracking"
else
    fail "No stamp file found for dependency test"
fi

echo ""
echo "=== Cleanup and Summary ==="
make clean >/dev/null
rm -rf "$BEAKER_STAMP_DIR"
rm -f example.jpg example-2-birds.jpg

echo "✅ All tests passed!"
