# Test Makefile for beaker Make compatibility
BEAKER = ./target/release/beaker

.PHONY: all clean

all: example_crop.jpg example_cutout.png

example_crop.jpg: example.jpg
	$(BEAKER) detect $< --crop head --metadata --depfile $@.d
-include example_crop.jpg.d

example_cutout.png: example.jpg
	$(BEAKER) cutout $< --metadata --depfile $@.d
-include example_cutout.png.d

clean:
	rm -f *.jpg *.png *.toml *.d
	rm -rf ~/.cache/beaker/stamps/
