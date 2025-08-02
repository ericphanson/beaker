#!/usr/bin/env python3
"""
End-to-end test suite for bird-head-detector tool.

Tests the complete workflow:
1. Build and install the tool
2. Run inference with various options
3. Verify correct output files are created
4. Test directory processing
5. Test different argument combinations
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class BirdHeadDetectorE2ETest(unittest.TestCase):
    """End-to-end tests for bird-head-detector tool."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment - build and install the tool."""
        cls.repo_root = Path(__file__).parent.parent
        cls.example_image = cls.repo_root / "example.jpg"

        # Verify example image exists
        if not cls.example_image.exists():
            raise FileNotFoundError(f"Example image not found: {cls.example_image}")

        # Create temporary directory for installation
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="bird_detector_test_"))
        cls.test_images_dir = cls.temp_dir / "test_images"
        cls.test_images_dir.mkdir()

        print(f"Test directory: {cls.temp_dir}")
        print(f"Test images directory: {cls.test_images_dir}")

        # Copy example image to test directory with different names
        cls.single_image = cls.test_images_dir / "test_bird.jpg"
        cls.png_image = cls.test_images_dir / "test_bird.png"
        cls.batch_dir = cls.test_images_dir / "batch_test"
        cls.batch_dir.mkdir()

        # Copy and rename images for testing
        shutil.copy2(cls.example_image, cls.single_image)
        shutil.copy2(cls.example_image, cls.png_image)
        shutil.copy2(cls.example_image, cls.batch_dir / "bird1.jpg")
        shutil.copy2(cls.example_image, cls.batch_dir / "bird2.jpg")
        shutil.copy2(cls.example_image, cls.batch_dir / "bird3.png")

        # Build and install the tool
        cls._build_and_install_tool()

    @classmethod
    def _build_and_install_tool(cls):
        """Build and install the bird-head-detector tool."""
        print("Building bird-head-detector package...")

        # Build the package
        build_result = subprocess.run(
            ["uv", "build"], cwd=cls.repo_root, capture_output=True, text=True
        )

        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build package: {build_result.stderr}")

        print("Installing bird-head-detector tool...")

        # Install the tool
        install_result = subprocess.run(
            ["uv", "tool", "install", str(cls.repo_root), "--force"], capture_output=True, text=True
        )

        if install_result.returncode != 0:
            raise RuntimeError(f"Failed to install tool: {install_result.stderr}")

        print("✅ Tool installed successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Uninstall the tool
        subprocess.run(["uv", "tool", "uninstall", "bird-head-detector"], capture_output=True)

        # Clean up temp directory
        shutil.rmtree(cls.temp_dir)
        print(f"Cleaned up test directory: {cls.temp_dir}")

    def setUp(self):
        """Set up each test."""
        # Clean any existing output files
        for file_path in self.test_images_dir.rglob("*-crop.*"):
            file_path.unlink()
        for file_path in self.test_images_dir.rglob("*-bounding-box.*"):
            file_path.unlink()

    def _run_detector(self, args, expect_success=True):
        """Run bird-head-detector with given arguments."""
        # First try using the installed tool
        cmd = ["bird-head-detector"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)

        if expect_success and result.returncode != 0:
            # Collect detailed debug information for failure
            debug_info = self._collect_debug_info(cmd, result, args)
            self.fail(f"Command failed: {' '.join(cmd)}\n{debug_info}")

        return result

    def _collect_debug_info(self, cmd, result, args):
        """Collect comprehensive debug information for test failures."""
        debug_lines = [
            "=" * 80,
            "COMMAND FAILURE DEBUG INFORMATION",
            "=" * 80,
            f"Command: {' '.join(cmd)}",
            f"Working directory: {self.temp_dir}",
            f"Return code: {result.returncode}",
            "",
            "STDOUT:",
            "-" * 40,
            result.stdout or "(empty)",
            "",
            "STDERR:",
            "-" * 40,
            result.stderr or "(empty)",
            "",
            "ENVIRONMENT:",
            "-" * 40,
            f"Test directory: {self.temp_dir}",
            f"Test images directory: {self.test_images_dir}",
            f"Repository root: {self.repo_root}",
            "",
            "FILE SYSTEM STATE:",
            "-" * 40,
        ]

        # Add file system information
        try:
            debug_lines.append("Test directory contents:")
            for item in sorted(self.temp_dir.rglob("*")):
                rel_path = item.relative_to(self.temp_dir)
                if item.is_file():
                    debug_lines.append(f"  📄 {rel_path} ({item.stat().st_size} bytes)")
                else:
                    debug_lines.append(f"  📁 {rel_path}/")
        except Exception as e:
            debug_lines.append(f"Error listing test directory: {e}")

        debug_lines.extend(
            [
                "",
                "TOOL INSTALLATION STATUS:",
                "-" * 40,
            ]
        )

        # Check tool installation
        try:
            tool_check = subprocess.run(
                ["uv", "tool", "list"], capture_output=True, text=True, timeout=10
            )
            if "bird-head-detector" in tool_check.stdout:
                debug_lines.append("✅ bird-head-detector is installed via uv tool")
            else:
                debug_lines.append("❌ bird-head-detector NOT found in uv tool list")
            debug_lines.extend(
                [
                    "uv tool list output:",
                    tool_check.stdout or "(empty)",
                ]
            )
        except Exception as e:
            debug_lines.append(f"Error checking tool installation: {e}")

        # Test direct command execution
        debug_lines.extend(
            [
                "",
                "DIRECT COMMAND TEST:",
                "-" * 40,
            ]
        )

        try:
            help_result = subprocess.run(
                ["bird-head-detector", "--help"], capture_output=True, text=True, timeout=10
            )
            debug_lines.extend(
                [
                    f"bird-head-detector --help return code: {help_result.returncode}",
                    f"Help output length: {len(help_result.stdout)} chars",
                ]
            )
            if help_result.returncode != 0:
                debug_lines.extend(
                    [
                        "Help command stderr:",
                        help_result.stderr or "(empty)",
                    ]
                )
        except Exception as e:
            debug_lines.append(f"Error testing help command: {e}")

        # Add system information
        debug_lines.extend(
            [
                "",
                "SYSTEM INFORMATION:",
                "-" * 40,
                f"Python executable: {subprocess.sys.executable}",
                f"Current working directory: {os.getcwd()}",
                "PATH environment variable:",
            ]
        )

        try:
            path_dirs = os.environ.get("PATH", "").split(":")
            for path_dir in path_dirs[:10]:  # Show first 10 PATH entries
                debug_lines.append(f"  {path_dir}")
            if len(path_dirs) > 10:
                debug_lines.append(f"  ... and {len(path_dirs) - 10} more directories")
        except Exception as e:
            debug_lines.append(f"Error listing PATH: {e}")

        debug_lines.append("=" * 80)

    def _assert_file_exists(self, file_path, description="file"):
        """Assert that a file exists with verbose debug information on failure."""
        if not file_path.exists():
            debug_info = self._collect_file_debug_info(file_path, description)
            self.fail(f"Expected {description} not found: {file_path}\n{debug_info}")

    def _assert_file_not_exists(self, file_path, description="file"):
        """Assert that a file does NOT exist with verbose debug information on failure."""
        if file_path.exists():
            debug_info = self._collect_file_debug_info(file_path, description)
            self.fail(f"Unexpected {description} found: {file_path}\n{debug_info}")

    def _collect_file_debug_info(self, target_file, description):
        """Collect debug information about file expectations."""
        debug_lines = [
            "=" * 60,
            f"FILE ASSERTION FAILURE: {description}",
            "=" * 60,
            f"Expected file: {target_file}",
            f"Parent directory: {target_file.parent}",
            f"File exists: {target_file.exists()}",
            "",
            "DIRECTORY CONTENTS:",
            "-" * 30,
        ]

        try:
            parent_dir = target_file.parent
            if parent_dir.exists():
                debug_lines.append(f"📁 {parent_dir}:")
                for item in sorted(parent_dir.iterdir()):
                    if item.is_file():
                        size = item.stat().st_size
                        debug_lines.append(f"  📄 {item.name} ({size} bytes)")
                    else:
                        debug_lines.append(f"  📁 {item.name}/")

                # Also check subdirectories for relevant files
                for item in sorted(parent_dir.rglob("*crop*")):
                    if item != target_file:
                        rel_path = item.relative_to(parent_dir)
                        if item.is_file():
                            size = item.stat().st_size
                            debug_lines.append(
                                f"  🔍 Found crop-related: {rel_path} ({size} bytes)"
                            )

                for item in sorted(parent_dir.rglob("*bounding-box*")):
                    if item != target_file:
                        rel_path = item.relative_to(parent_dir)
                        if item.is_file():
                            size = item.stat().st_size
                            debug_lines.append(
                                f"  🔍 Found bbox-related: {rel_path} ({size} bytes)"
                            )
            else:
                debug_lines.append(f"❌ Parent directory does not exist: {parent_dir}")
        except Exception as e:
            debug_lines.append(f"Error listing directory contents: {e}")

        debug_lines.append("=" * 60)
        return "\n".join(debug_lines)

    def test_default_behavior_single_image(self):
        """Test default behavior: should create crop by default."""
        # Run detector with minimal arguments
        result = self._run_detector([str(self.single_image)])

        # Check that crop was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self._assert_file_exists(expected_crop, "crop file")

        # Check that bounding box was NOT created (not requested)
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"
        self._assert_file_not_exists(expected_bbox, "bounding box file")

        # Verify output mentions crop creation
        self.assertIn("Created crop:", result.stdout)
        self.assertIn("Created 1 square head crops", result.stdout)

    def test_skip_crop_option(self):
        """Test --skip-crop option."""
        result = self._run_detector([str(self.single_image), "--skip-crop"])

        # Check that NO crop was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self._assert_file_not_exists(expected_crop, "crop file")

        # Verify output doesn't mention crops
        self.assertNotIn("Created crop:", result.stdout)
        self.assertNotIn("Created 1 square head crops", result.stdout)

    def test_save_bounding_box_option(self):
        """Test --save-bounding-box option."""
        result = self._run_detector([str(self.single_image), "--save-bounding-box"])

        # Check that both crop AND bounding box were created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self._assert_file_exists(expected_crop, "crop file")
        self._assert_file_exists(expected_bbox, "bounding box file")

        # Verify output mentions both
        self.assertIn("Created crop:", result.stdout)
        self.assertIn("Created bounding box image:", result.stdout)

    def test_skip_crop_with_bounding_box(self):
        """Test --skip-crop with --save-bounding-box (only bounding box should be created)."""
        result = self._run_detector(
            [str(self.single_image), "--skip-crop", "--save-bounding-box"]
        )

        # Check that only bounding box was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self.assertFalse(expected_crop.exists(), f"Unexpected crop file found: {expected_crop}")
        self.assertTrue(
            expected_bbox.exists(), f"Expected bounding box file not found: {expected_bbox}"
        )

        # Verify output
        self.assertNotIn("Created crop:", result.stdout)
        self.assertIn("Created bounding box image:", result.stdout)

    def test_png_format_preservation(self):
        """Test that PNG format is preserved in outputs."""
        result = self._run_detector([str(self.png_image), "--save-bounding-box"])

        # Check that outputs maintain PNG format
        expected_crop = self.png_image.parent / "test_bird-crop.png"
        expected_bbox = self.png_image.parent / "test_bird-bounding-box.png"

        self.assertTrue(
            expected_crop.exists(), f"Expected PNG crop file not found: {expected_crop}"
        )
        self.assertTrue(
            expected_bbox.exists(), f"Expected PNG bounding box file not found: {expected_bbox}"
        )

    def test_output_dir_option(self):
        """Test --output-dir option."""
        output_dir = self.temp_dir / "custom_output"

        result = self._run_detector(
            [
                str(self.single_image),
                "--save-bounding-box",
                "--output-dir",
                str(output_dir),
            ]
        )

        # Check that files were created in output directory with original names
        expected_crop = output_dir / "test_bird.jpg"
        expected_bbox = output_dir / "test_bird.jpg"  # Same name, will overwrite

        self.assertTrue(output_dir.exists(), f"Output directory not created: {output_dir}")
        self.assertTrue(
            expected_crop.exists(), f"Expected crop in output dir not found: {expected_crop}"
        )

        # Verify no files created next to original
        original_dir_crop = self.single_image.parent / "test_bird-crop.jpg"
        original_dir_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self.assertFalse(
            original_dir_crop.exists(),
            f"Unexpected file in original directory: {original_dir_crop}",
        )
        self.assertFalse(
            original_dir_bbox.exists(),
            f"Unexpected file in original directory: {original_dir_bbox}",
        )

    def test_directory_processing(self):
        """Test processing entire directory of images."""
        result = self._run_detector([str(self.batch_dir)])

        # Check that crops were created for all images
        expected_files = [
            self.batch_dir / "bird1-crop.jpg",
            self.batch_dir / "bird2-crop.jpg",
            self.batch_dir / "bird3-crop.png",
        ]

        for expected_file in expected_files:
            self.assertTrue(
                expected_file.exists(), f"Expected crop file not found: {expected_file}"
            )

        # Verify output mentions multiple crops
        self.assertIn("Created 3 square head crops", result.stdout)

    def test_directory_with_output_dir(self):
        """Test directory processing with output directory."""
        output_dir = self.temp_dir / "batch_output"

        result = self._run_detector(
            [str(self.batch_dir), "--output-dir", str(output_dir)]
        )

        # Check that files were created in output directory with original names
        expected_files = [
            output_dir / "bird1.jpg",
            output_dir / "bird2.jpg",
            output_dir / "bird3.png",
        ]

        for expected_file in expected_files:
            self.assertTrue(
                expected_file.exists(), f"Expected file in output dir not found: {expected_file}"
            )

        # Verify no -crop files created in original directory
        for crop_file in self.batch_dir.glob("*-crop.*"):
            self.fail(f"Unexpected crop file in original directory: {crop_file}")

    def test_padding_option(self):
        """Test --padding option (basic functionality test)."""
        result = self._run_detector([str(self.single_image), "--padding", "0.5"])

        # Check that crop was created (padding affects crop size, not filename)
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertTrue(expected_crop.exists(), f"Expected crop file not found: {expected_crop}")

    def test_confidence_threshold(self):
        """Test --conf option (basic functionality test)."""
        result = self._run_detector([str(self.single_image), "--conf", "0.1"])

        # Check that crop was created (confidence affects detection, not filename)
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertTrue(expected_crop.exists(), f"Expected crop file not found: {expected_crop}")

    def test_help_option(self):
        """Test --help option."""
        result = subprocess.run(["bird-head-detector", "--help"], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("Bird head detection inference", result.stdout)
        self.assertIn("--skip-crop", result.stdout)
        self.assertIn("--save-bounding-box", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--padding", result.stdout)


def run_tests():
    """Run the test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BirdHeadDetectorE2ETest)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
