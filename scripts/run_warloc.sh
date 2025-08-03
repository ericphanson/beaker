#!/bin/bash
set -euo pipefail

# Ensure the output directory exists
mkdir -p beaker

# Run the command
cargo warloc --by-file > beaker/line_counts.txt
