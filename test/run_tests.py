#!/usr/bin/env python3
"""
Simple test runner for the bird-head-detector end-to-end tests.
"""

import os
import sys
from pathlib import Path

# Configure UTF-8 encoding to handle Unicode emoji characters on all platforms
os.environ['PYTHONIOENCODING'] = 'utf-8'
# Reconfigure stdout/stderr to use UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add the test directory to the path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))

from test_e2e import run_tests

if __name__ == "__main__":
    print("🧪 Running bird-head-detector end-to-end tests...")
    print("=" * 60)

    success = run_tests()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    sys.exit(0 if success else 1)
