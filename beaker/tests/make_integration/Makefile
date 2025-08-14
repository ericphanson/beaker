# Makefile for testing beaker incremental build compatibility
# This verifies that beaker integrates properly with Make for dependency tracking

# Test images and outputs
TEST_IMG := example.jpg
CUTOUT_OUTPUT := example_cutout.png
DETECT_OUTPUT := example_bounding-box.jpg

# Beaker binary (use cargo run in development, or from environment)
BEAKER ?= ../../../target/release/beaker

# Test targets
.PHONY: all test_incremental clean detect-confidence-0.3 detect-with-metadata cutout-with-metadata

# Alternative confidence test
detect-confidence-0.3:
	$(BEAKER) detect $(TEST_IMG) --confidence 0.3 --crop bird --depfile example_crop_03.jpg.d

# Test targets with metadata for integration testing
detect-with-metadata: example_crop_head.jpg

example_crop_head.jpg: $(TEST_IMG)
	@echo "Building detection with metadata from $<"
	$(BEAKER) detect $< --confidence 0.5 --crop head --metadata --depfile $@.d

cutout-with-metadata: $(CUTOUT_OUTPUT)

all: $(CUTOUT_OUTPUT) $(DETECT_OUTPUT)

# Cutout target - includes depfile generation
$(CUTOUT_OUTPUT): $(TEST_IMG)
	@echo "Building cutout from $<"
	$(BEAKER) cutout $< --metadata --depfile $@.d

# Detection target - includes depfile generation
$(DETECT_OUTPUT): $(TEST_IMG)
	@echo "Building detection crops from $<"
	$(BEAKER) detect $< --confidence 0.1 --bounding-box --metadata --depfile $@.d

# Include dependency files if they exist
-include $(CUTOUT_OUTPUT).d
-include $(DETECT_OUTPUT).d
-include example_crop_head.jpg.d

# Test incremental behavior
test_incremental: clean
	@echo "=== Testing incremental builds ==="
	@echo "1. Initial build (should build both targets)"
	$(MAKE) all
	@echo "2. Immediate rebuild (should do nothing)"
	$(MAKE) all
	@echo "3. Touch input (should rebuild both)"
	touch $(TEST_IMG)
	$(MAKE) all
	@echo "=== Incremental build test complete ==="

clean:
	rm -f $(CUTOUT_OUTPUT) $(DETECT_OUTPUT)
	rm -f $(CUTOUT_OUTPUT).d $(DETECT_OUTPUT).d
	rm -f example_crop.jpg
	rm -f example_crop.jpg.d
	rm -f example_crop_head.jpg
	rm -f example_crop_head.jpg.d
	rm -f example_crop_03.jpg example_crop_03.jpg.d
	rm -f example_bbox.jpg example_bbox.jpg.d
	rm -f *.beaker.toml *.beaker.json
