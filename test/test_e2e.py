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
            ["uv", "build"],
            cwd=cls.repo_root,
            capture_output=True,
            text=True
        )

        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build package: {build_result.stderr}")

        print("Installing bird-head-detector tool...")

        # Install the tool
        install_result = subprocess.run(
            ["uv", "tool", "install", str(cls.repo_root), "--force"],
            capture_output=True,
            text=True
        )

        if install_result.returncode != 0:
            raise RuntimeError(f"Failed to install tool: {install_result.stderr}")

        print("✅ Tool installed successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Uninstall the tool
        subprocess.run(["uv", "tool", "uninstall", "bird-head-detector"],
                      capture_output=True)

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
            self.fail(f"Command failed: {' '.join(cmd)}\nStderr: {result.stderr}\nStdout: {result.stdout}")

        return result

    def test_default_behavior_single_image(self):
        """Test default behavior: should create crop by default."""
        # Run detector with minimal arguments
        result = self._run_detector(["--source", str(self.single_image)])

        # Check that crop was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertTrue(expected_crop.exists(),
                       f"Expected crop file not found: {expected_crop}")

        # Check that bounding box was NOT created (not requested)
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"
        self.assertFalse(expected_bbox.exists(),
                        f"Unexpected bounding box file found: {expected_bbox}")

        # Verify output mentions crop creation
        self.assertIn("Created crop:", result.stdout)
        self.assertIn("Created 1 square head crops", result.stdout)

    def test_skip_crop_option(self):
        """Test --skip-crop option."""
        result = self._run_detector(["--source", str(self.single_image), "--skip-crop"])

        # Check that NO crop was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertFalse(expected_crop.exists(),
                        f"Unexpected crop file found: {expected_crop}")

        # Verify output doesn't mention crops
        self.assertNotIn("Created crop:", result.stdout)
        self.assertNotIn("Created 1 square head crops", result.stdout)

    def test_save_bounding_box_option(self):
        """Test --save-bounding-box option."""
        result = self._run_detector(["--source", str(self.single_image), "--save-bounding-box"])

        # Check that both crop AND bounding box were created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self.assertTrue(expected_crop.exists(),
                       f"Expected crop file not found: {expected_crop}")
        self.assertTrue(expected_bbox.exists(),
                       f"Expected bounding box file not found: {expected_bbox}")

        # Verify output mentions both
        self.assertIn("Created crop:", result.stdout)
        self.assertIn("Created bounding box image:", result.stdout)

    def test_skip_crop_with_bounding_box(self):
        """Test --skip-crop with --save-bounding-box (only bounding box should be created)."""
        result = self._run_detector([
            "--source", str(self.single_image),
            "--skip-crop",
            "--save-bounding-box"
        ])

        # Check that only bounding box was created
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        expected_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self.assertFalse(expected_crop.exists(),
                        f"Unexpected crop file found: {expected_crop}")
        self.assertTrue(expected_bbox.exists(),
                       f"Expected bounding box file not found: {expected_bbox}")

        # Verify output
        self.assertNotIn("Created crop:", result.stdout)
        self.assertIn("Created bounding box image:", result.stdout)

    def test_png_format_preservation(self):
        """Test that PNG format is preserved in outputs."""
        result = self._run_detector([
            "--source", str(self.png_image),
            "--save-bounding-box"
        ])

        # Check that outputs maintain PNG format
        expected_crop = self.png_image.parent / "test_bird-crop.png"
        expected_bbox = self.png_image.parent / "test_bird-bounding-box.png"

        self.assertTrue(expected_crop.exists(),
                       f"Expected PNG crop file not found: {expected_crop}")
        self.assertTrue(expected_bbox.exists(),
                       f"Expected PNG bounding box file not found: {expected_bbox}")

    def test_output_dir_option(self):
        """Test --output-dir option."""
        output_dir = self.temp_dir / "custom_output"

        result = self._run_detector([
            "--source", str(self.single_image),
            "--save-bounding-box",
            "--output-dir", str(output_dir)
        ])

        # Check that files were created in output directory with original names
        expected_crop = output_dir / "test_bird.jpg"
        expected_bbox = output_dir / "test_bird.jpg"  # Same name, will overwrite

        self.assertTrue(output_dir.exists(), f"Output directory not created: {output_dir}")
        self.assertTrue(expected_crop.exists(),
                       f"Expected crop in output dir not found: {expected_crop}")

        # Verify no files created next to original
        original_dir_crop = self.single_image.parent / "test_bird-crop.jpg"
        original_dir_bbox = self.single_image.parent / "test_bird-bounding-box.jpg"

        self.assertFalse(original_dir_crop.exists(),
                        f"Unexpected file in original directory: {original_dir_crop}")
        self.assertFalse(original_dir_bbox.exists(),
                        f"Unexpected file in original directory: {original_dir_bbox}")

    def test_directory_processing(self):
        """Test processing entire directory of images."""
        result = self._run_detector(["--source", str(self.batch_dir)])

        # Check that crops were created for all images
        expected_files = [
            self.batch_dir / "bird1-crop.jpg",
            self.batch_dir / "bird2-crop.jpg",
            self.batch_dir / "bird3-crop.png"
        ]

        for expected_file in expected_files:
            self.assertTrue(expected_file.exists(),
                           f"Expected crop file not found: {expected_file}")

        # Verify output mentions multiple crops
        self.assertIn("Created 3 square head crops", result.stdout)

    def test_directory_with_output_dir(self):
        """Test directory processing with output directory."""
        output_dir = self.temp_dir / "batch_output"

        result = self._run_detector([
            "--source", str(self.batch_dir),
            "--output-dir", str(output_dir)
        ])

        # Check that files were created in output directory with original names
        expected_files = [
            output_dir / "bird1.jpg",
            output_dir / "bird2.jpg",
            output_dir / "bird3.png"
        ]

        for expected_file in expected_files:
            self.assertTrue(expected_file.exists(),
                           f"Expected file in output dir not found: {expected_file}")

        # Verify no -crop files created in original directory
        for crop_file in self.batch_dir.glob("*-crop.*"):
            self.fail(f"Unexpected crop file in original directory: {crop_file}")

    def test_padding_option(self):
        """Test --padding option (basic functionality test)."""
        result = self._run_detector([
            "--source", str(self.single_image),
            "--padding", "0.5"
        ])

        # Check that crop was created (padding affects crop size, not filename)
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertTrue(expected_crop.exists(),
                       f"Expected crop file not found: {expected_crop}")

    def test_confidence_threshold(self):
        """Test --conf option (basic functionality test)."""
        result = self._run_detector([
            "--source", str(self.single_image),
            "--conf", "0.1"
        ])

        # Check that crop was created (confidence affects detection, not filename)
        expected_crop = self.single_image.parent / "test_bird-crop.jpg"
        self.assertTrue(expected_crop.exists(),
                       f"Expected crop file not found: {expected_crop}")

    def test_help_option(self):
        """Test --help option."""
        result = subprocess.run(
            ["bird-head-detector", "--help"],
            capture_output=True,
            text=True
        )

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
