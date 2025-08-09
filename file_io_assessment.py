#!/usr/bin/env python3
"""
File I/O Performance Assessment Script

This script reproduces the file I/O timing measurements used in the pipeline
planning analysis. It runs beaker commands and parses the resulting metadata
files to extract timing information and report performance characteristics.
"""

import subprocess

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib_fallback

    tomllib = None
import os
import glob
import sys
import tempfile


def run_beaker_command(cmd, cwd=None):
    """Run a beaker command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def extract_timing_from_metadata(metadata_file):
    """Extract timing information from a beaker metadata file"""
    try:
        with open(metadata_file, "rb") as f:
            if tomllib:
                metadata = tomllib.load(f)
            else:
                # Fallback for older Python versions
                with open(metadata_file, "r") as text_f:
                    metadata = tomllib_fallback.load(text_f)

        timing_data = {}

        # Extract head timing if present
        if "head" in metadata and "execution" in metadata["head"]:
            exec_data = metadata["head"]["execution"]
            timing_data["head"] = {
                "model_processing_time_ms": exec_data.get("model_processing_time_ms"),
                "file_io": exec_data.get("file_io", {}),
            }

        # Extract cutout timing if present
        if "cutout" in metadata and "execution" in metadata["cutout"]:
            exec_data = metadata["cutout"]["execution"]
            timing_data["cutout"] = {
                "model_processing_time_ms": exec_data.get("model_processing_time_ms"),
                "file_io": exec_data.get("file_io", {}),
            }

        return timing_data
    except Exception as e:
        print(f"Error reading metadata from {metadata_file}: {e}")
        return {}


def analyze_timing_data(timing_data, operation_name):
    """Analyze timing data and report performance breakdown"""
    print(f"\n=== {operation_name} Performance Analysis ===")

    total_model_time = 0
    total_io_read_time = 0
    total_io_write_time = 0

    for tool, data in timing_data.items():
        model_time = data.get("model_processing_time_ms", 0)
        file_io = data.get("file_io", {})
        read_time = file_io.get("read_time_ms", 0)
        write_time = file_io.get("write_time_ms", 0)

        print(f"\n{tool.upper()} Model:")
        print(f"  Model processing: {model_time:.1f}ms")
        print(f"  File I/O read:    {read_time:.1f}ms")
        print(f"  File I/O write:   {write_time:.1f}ms")

        total_model_time += model_time if model_time else 0
        total_io_read_time += read_time if read_time else 0
        total_io_write_time += write_time if write_time else 0

    total_io_time = total_io_read_time + total_io_write_time
    total_time = total_model_time + total_io_time

    print("\n=== SUMMARY ===")
    print(
        f"Total model processing: {total_model_time:.1f}ms ({100*total_model_time/total_time:.1f}%)"
    )
    print(
        f"Total file I/O:         {total_io_time:.1f}ms ({100*total_io_time/total_time:.1f}%)"
    )
    print(f"  - Read operations:    {total_io_read_time:.1f}ms")
    print(f"  - Write operations:   {total_io_write_time:.1f}ms")
    print(f"Total time:             {total_time:.1f}ms")

    if total_time > 0:
        pipeline_savings = total_io_write_time  # Eliminate intermediate writes
        pipeline_improvement = 100 * pipeline_savings / total_time
        print(
            f"\nPipeline potential savings: {pipeline_savings:.1f}ms ({pipeline_improvement:.1f}%)"
        )
        print("(Eliminates intermediate file writes between pipeline steps)")

    return {
        "total_model_time": total_model_time,
        "total_io_time": total_io_time,
        "total_time": total_time,
        "io_percentage": 100 * total_io_time / total_time if total_time > 0 else 0,
        "pipeline_savings_pct": 100 * total_io_write_time / total_time
        if total_time > 0
        else 0,
    }


def main():
    """Main assessment function"""
    # Find the beaker binary - only release paths, debug timing info is garbage
    beaker_path = None
    possible_paths = [
        "./beaker/target/release/beaker",
        "./target/release/beaker",
        "../target/release/beaker",
        "../target/debug/beaker",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            beaker_path = path
            break

    if not beaker_path:
        print(
            "Error: Could not find beaker binary. Please build first with 'cargo build --release'"
        )
        return 1

    print(f"Using beaker binary: {beaker_path}")

    # Find test images
    test_images = []
    possible_images = ["../example.jpg", "../../example.jpg", "example.jpg"]
    for img in possible_images:
        if os.path.exists(img):
            test_images.append(img)

    if not test_images:
        print("Error: Could not find example.jpg test image")
        return 1

    print(f"Using test images: {test_images}")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Output directory: {temp_dir}")

        # Test 1: Head detection with timing
        print("\n" + "=" * 60)
        print("Testing HEAD detection with file I/O timing...")
        print("=" * 60)

        cmd = f"{beaker_path} head {test_images[0]} --confidence 0.5 --crop --metadata --output-dir {temp_dir}"
        success, stdout, stderr = run_beaker_command(cmd)

        if not success:
            print(f"Head detection failed: {stderr}")
            return 1

        # Find and analyze head metadata
        head_metadata_files = glob.glob(os.path.join(temp_dir, "*.beaker.toml"))
        if head_metadata_files:
            timing_data = extract_timing_from_metadata(head_metadata_files[0])
            analyze_timing_data(timing_data, "Head Detection")

        # Test 2: Cutout processing with timing
        print("\n" + "=" * 60)
        print("Testing CUTOUT processing with file I/O timing...")
        print("=" * 60)

        # Clean temp directory
        for f in glob.glob(os.path.join(temp_dir, "*")):
            os.remove(f)

        cmd = f"{beaker_path} cutout {test_images[0]} --save-mask --metadata --output-dir {temp_dir}"
        success, stdout, stderr = run_beaker_command(cmd)

        if not success:
            print(f"Cutout processing failed: {stderr}")
            return 1

        # Find and analyze cutout metadata
        cutout_metadata_files = glob.glob(os.path.join(temp_dir, "*.beaker.toml"))
        if cutout_metadata_files:
            timing_data = extract_timing_from_metadata(cutout_metadata_files[0])
            analyze_timing_data(timing_data, "Cutout Processing")

        # Test 3: Combined pipeline simulation (head + cutout sequentially)
        print("\n" + "=" * 60)
        print("Testing PIPELINE simulation (head -> cutout sequence)...")
        print("=" * 60)

        # Clean temp directory
        for f in glob.glob(os.path.join(temp_dir, "*")):
            os.remove(f)

        # Step 1: Head detection with crop output
        cmd = f"{beaker_path} head {test_images[0]} --confidence 0.5 --crop --metadata --output-dir {temp_dir}"
        success, stdout, stderr = run_beaker_command(cmd)

        if not success:
            print(f"Pipeline head step failed: {stderr}")
            return 1

        # Find crop output for next step
        crop_files = glob.glob(os.path.join(temp_dir, "*crop*.jpg"))
        if not crop_files:
            print("No crop files generated for pipeline simulation")
            return 1

        # Step 2: Cutout on crop result
        crop_file = crop_files[0]  # Use first crop
        cmd = f"{beaker_path} cutout {crop_file} --metadata --output-dir {temp_dir}"
        success, stdout, stderr = run_beaker_command(cmd)

        if not success:
            print(f"Pipeline cutout step failed: {stderr}")
            return 1

        # Analyze combined timing
        all_metadata_files = glob.glob(os.path.join(temp_dir, "*.beaker.toml"))
        combined_timing = {}

        for metadata_file in all_metadata_files:
            timing_data = extract_timing_from_metadata(metadata_file)
            combined_timing.update(timing_data)

        if combined_timing:
            pipeline_analysis = analyze_timing_data(
                combined_timing, "Combined Pipeline"
            )

            print("\n" + "=" * 60)
            print("PIPELINE FEASIBILITY ASSESSMENT")
            print("=" * 60)
            print(
                f"Current file I/O overhead: {pipeline_analysis['io_percentage']:.1f}% of total time"
            )
            print(
                f"Potential pipeline improvement: {pipeline_analysis['pipeline_savings_pct']:.1f}%"
            )

            if pipeline_analysis["pipeline_savings_pct"] < 2.0:
                print("\n⚠️  CONCLUSION: Limited performance benefit")
                print(
                    "   Pipeline value should focus on ergonomics and workflow simplification"
                )
            else:
                print("\n✅ CONCLUSION: Meaningful performance benefit possible")
                print("   Pipeline implementation justified by performance gains")

    print("\n" + "=" * 60)
    print("Assessment complete. Metadata files with timing data generated.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
