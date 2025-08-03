#!/usr/bin/env python3
"""
Benchmark script for comparing beaker vs rembg performance.

Tests:
1. Single image processing (head detection, cutout)
2. Batch processing (20x copies of each image)
3. End-to-end time comparison
4. Loading time vs inference time breakdown
"""

import subprocess
import time
import os
import shutil
import tempfile
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.load_times = []
        self.inference_times = []
        self.total_times = []
        self.success = True
        self.error_message = ""

    def add_run(self, load_time: float, inference_time: float, total_time: float):
        self.load_times.append(load_time)
        self.inference_times.append(inference_time)
        self.total_times.append(total_time)

    def get_stats(self) -> Dict:
        if not self.load_times:
            return {"error": self.error_message}

        return {
            "runs": len(self.load_times),
            "load_time": {
                "mean": statistics.mean(self.load_times),
                "median": statistics.median(self.load_times),
                "min": min(self.load_times),
                "max": max(self.load_times),
                "stdev": statistics.stdev(self.load_times)
                if len(self.load_times) > 1
                else 0,
            },
            "inference_time": {
                "mean": statistics.mean(self.inference_times),
                "median": statistics.median(self.inference_times),
                "min": min(self.inference_times),
                "max": max(self.inference_times),
                "stdev": statistics.stdev(self.inference_times)
                if len(self.inference_times) > 1
                else 0,
            },
            "total_time": {
                "mean": statistics.mean(self.total_times),
                "median": statistics.median(self.total_times),
                "min": min(self.total_times),
                "max": max(self.total_times),
                "stdev": statistics.stdev(self.total_times)
                if len(self.total_times) > 1
                else 0,
            },
        }


def run_command(cmd: List[str], timeout: int = 120) -> Tuple[bool, str, str, float]:
    """Run a command and return success, stdout, stderr, and execution time."""
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/Users/eph/bird-head-detector/beaker-rs",
        )
        end_time = time.time()
        return (
            result.returncode == 0,
            result.stdout,
            result.stderr,
            end_time - start_time,
        )
    except subprocess.TimeoutExpired:
        end_time = time.time()
        return False, "", f"Command timed out after {timeout}s", end_time - start_time
    except Exception as e:
        end_time = time.time()
        return False, "", str(e), end_time - start_time


def parse_beaker_output(stdout: str) -> Tuple[float, float]:
    """Parse beaker output to extract load time and inference time."""
    load_time = 0.0
    inference_time = 0.0

    lines = stdout.split("\n")
    for line in lines:
        if "Loaded" in line and "in" in line and "ms" in line:
            # Extract load time: "🤖 Loaded embedded ONNX model (12183794 bytes) in 266.318ms"
            try:
                parts = line.split(" in ")
                if len(parts) > 1:
                    time_part = parts[1].split("ms")[0]
                    load_time = float(time_part)
            except (ValueError, IndexError):
                pass
        elif "Inference completed in" in line or "completed in" in line:
            # Extract inference time: "⚡ Inference completed in 31.875ms"
            try:
                parts = line.split(" in ")
                if len(parts) > 1:
                    time_part = parts[1].split("ms")[0]
                    inference_time = float(time_part)
            except (ValueError, IndexError):
                pass
        elif "Processed" in line and "in" in line and "ms" in line:
            # For cutout: "✅ Processed ../example.jpg in 5549.1ms → ../example_cutout.png"
            try:
                parts = line.split(" in ")
                if len(parts) > 1:
                    time_part = parts[1].split("ms")[0]
                    inference_time = float(time_part)
            except (ValueError, IndexError):
                pass

    return load_time, inference_time


def benchmark_beaker_head(
    image_path: str, device: str = "auto", runs: int = 3
) -> BenchmarkResult:
    """Benchmark beaker head detection."""
    result = BenchmarkResult(f"beaker_head_{device}")

    for run in range(runs):
        cmd = [
            "./target/release/beaker",
            "head",
            image_path,
            "--device",
            device,
            "--verbose",
            "--no-metadata",
        ]
        success, stdout, stderr, total_time = run_command(cmd)

        if not success:
            result.success = False
            result.error_message = f"Run {run+1} failed: {stderr}"
            break

        load_time, inference_time = parse_beaker_output(stdout)
        result.add_run(load_time, inference_time, total_time * 1000)  # Convert to ms

    return result


def benchmark_beaker_cutout(
    image_path: str, device: str = "auto", runs: int = 3
) -> BenchmarkResult:
    """Benchmark beaker cutout."""
    result = BenchmarkResult(f"beaker_cutout_{device}")

    for run in range(runs):
        cmd = [
            "./target/release/beaker",
            "cutout",
            image_path,
            "--device",
            device,
            "--verbose",
            "--no-metadata",
        ]
        success, stdout, stderr, total_time = run_command(cmd)

        if not success:
            result.success = False
            result.error_message = f"Run {run+1} failed: {stderr}"
            break

        load_time, inference_time = parse_beaker_output(stdout)
        result.add_run(load_time, inference_time, total_time * 1000)  # Convert to ms

    return result


def benchmark_rembg(image_path: str, runs: int = 3) -> BenchmarkResult:
    """Benchmark rembg CLI tool."""
    result = BenchmarkResult("rembg")

    # Check if rembg is installed
    check_cmd = ["rembg", "--version"]
    success, _, _, _ = run_command(check_cmd)
    if not success:
        result.success = False
        result.error_message = "rembg not installed or not in PATH"
        return result

    for run in range(runs):
        output_path = f"/tmp/rembg_output_{run}.png"
        cmd = ["rembg", "i", "-m", "isnet-general-use", image_path, output_path]
        success, stdout, stderr, total_time = run_command(cmd)

        if not success:
            result.success = False
            result.error_message = f"Run {run+1} failed: {stderr}"
            break

        # rembg doesn't separate load vs inference time, so we put everything in inference
        result.add_run(0.0, total_time * 1000, total_time * 1000)  # Convert to ms

        # Clean up output file
        if os.path.exists(output_path):
            os.remove(output_path)

    return result


def create_batch_images(image_path: str, count: int = 20) -> str:
    """Create a temporary directory with multiple copies of the image."""
    temp_dir = tempfile.mkdtemp(prefix="beaker_batch_")
    image_name = Path(image_path).stem
    image_ext = Path(image_path).suffix

    for i in range(count):
        dest_path = os.path.join(temp_dir, f"{image_name}_{i:02d}{image_ext}")
        shutil.copy2(image_path, dest_path)

    return temp_dir


def benchmark_beaker_batch(
    temp_dir: str, command: str, device: str = "auto", runs: int = 3
) -> BenchmarkResult:
    """Benchmark beaker batch processing."""
    result = BenchmarkResult(f"beaker_{command}_batch_{device}")

    for run in range(runs):
        cmd = [
            "./target/release/beaker",
            command,
            temp_dir,
            "--device",
            device,
            "--verbose",
            "--no-metadata",
        ]
        success, stdout, stderr, total_time = run_command(
            cmd, timeout=300
        )  # 5 min timeout for batch

        if not success:
            result.success = False
            result.error_message = f"Run {run+1} failed: {stderr}"
            break

        load_time, inference_time = parse_beaker_output(stdout)
        result.add_run(load_time, inference_time, total_time * 1000)  # Convert to ms

    return result


def benchmark_rembg_batch(temp_dir: str, runs: int = 3) -> BenchmarkResult:
    """Benchmark rembg batch processing."""
    result = BenchmarkResult("rembg_batch")

    # Check if rembg is installed
    check_cmd = ["rembg", "--version"]
    success, _, _, _ = run_command(check_cmd)
    if not success:
        result.success = False
        result.error_message = "rembg not installed or not in PATH"
        return result

    for run in range(runs):
        output_dir = tempfile.mkdtemp(prefix="rembg_batch_output_")

        # Use rembg's batch processing facility: rembg p input_folder output_folder
        cmd = ["rembg", "p", "-m", "isnet-general-use", temp_dir, output_dir]
        success, stdout, stderr, total_time = run_command(
            cmd, timeout=300
        )  # 5 min timeout for batch

        if not success:
            result.success = False
            result.error_message = f"Run {run+1} failed: {stderr}"
            break

        # rembg doesn't separate load vs inference time, so we put everything in inference
        result.add_run(0.0, total_time * 1000, total_time * 1000)  # Convert to ms

        # Clean up output directory
        shutil.rmtree(output_dir, ignore_errors=True)

    return result


def print_results(results: Dict[str, BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print("-" * 40)

        if not result.success:
            print(f"  ERROR: {result.error_message}")
            continue

        stats = result.get_stats()

        print(f"  Runs: {stats['runs']}")
        print("  Load Time (ms):")
        print(
            f"    Mean: {stats['load_time']['mean']:.1f} ± {stats['load_time']['stdev']:.1f}"
        )
        print(
            f"    Range: {stats['load_time']['min']:.1f} - {stats['load_time']['max']:.1f}"
        )

        print("  Inference Time (ms):")
        print(
            f"    Mean: {stats['inference_time']['mean']:.1f} ± {stats['inference_time']['stdev']:.1f}"
        )
        print(
            f"    Range: {stats['inference_time']['min']:.1f} - {stats['inference_time']['max']:.1f}"
        )

        print("  Total Time (ms):")
        print(
            f"    Mean: {stats['total_time']['mean']:.1f} ± {stats['total_time']['stdev']:.1f}"
        )
        print(
            f"    Range: {stats['total_time']['min']:.1f} - {stats['total_time']['max']:.1f}"
        )


def save_results_json(
    results: Dict[str, BenchmarkResult], filename: str = "benchmark_results.json"
):
    """Save results to JSON file."""
    json_results = {}
    for name, result in results.items():
        if result.success:
            json_results[name] = result.get_stats()
        else:
            json_results[name] = {"error": result.error_message}

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {filename}")


def main():
    print("🔬 Starting Beaker vs rembg Benchmark")
    print("=" * 50)

    # Test images
    example1 = "../example.jpg"
    example2 = "../example-2-birds.jpg"

    # Check if test images exist
    if not os.path.exists(example1):
        print(f"❌ Test image not found: {example1}")
        return
    if not os.path.exists(example2):
        print(f"❌ Test image not found: {example2}")
        return

    results = {}

    # Single image benchmarks
    print("\n📸 Single Image Benchmarks")
    print("-" * 30)

    devices = ["cpu", "coreml", "auto"]
    images = [("example", example1), ("example-2-birds", example2)]

    for img_name, img_path in images:
        print(f"\nTesting {img_name}...")

        # Beaker head detection
        for device in devices:
            print(f"  Beaker head ({device})...", end=" ", flush=True)
            key = f"beaker_head_{img_name}_{device}"
            results[key] = benchmark_beaker_head(img_path, device)
            if results[key].success:
                mean_time = results[key].get_stats()["total_time"]["mean"]
                print(f"✅ ({mean_time:.0f}ms)")
            else:
                print("❌")

        # Beaker cutout
        for device in devices:
            print(f"  Beaker cutout ({device})...", end=" ", flush=True)
            key = f"beaker_cutout_{img_name}_{device}"
            results[key] = benchmark_beaker_cutout(img_path, device)
            if results[key].success:
                mean_time = results[key].get_stats()["total_time"]["mean"]
                print(f"✅ ({mean_time:.0f}ms)")
            else:
                print("❌")

        # rembg
        print("  rembg...", end=" ", flush=True)
        key = f"rembg_{img_name}"
        results[key] = benchmark_rembg(img_path)
        if results[key].success:
            mean_time = results[key].get_stats()["total_time"]["mean"]
            print(f"✅ ({mean_time:.0f}ms)")
        else:
            print("❌")

    # Batch benchmarks (20x copies)
    print("\n📦 Batch Processing Benchmarks (20x images)")
    print("-" * 45)

    for img_name, img_path in images:
        print(f"\nCreating batch for {img_name}...")
        batch_dir = create_batch_images(img_path, 20)

        try:
            # Beaker head batch
            for device in devices:
                print(f"  Beaker head batch ({device})...", end=" ", flush=True)
                key = f"beaker_head_batch_{img_name}_{device}"
                results[key] = benchmark_beaker_batch(batch_dir, "head", device)
                if results[key].success:
                    mean_time = (
                        results[key].get_stats()["total_time"]["mean"] / 20
                    )  # Per image
                    print(f"✅ ({mean_time:.0f}ms/img)")
                else:
                    print("❌")

            # Beaker cutout batch
            for device in devices:
                print(f"  Beaker cutout batch ({device})...", end=" ", flush=True)
                key = f"beaker_cutout_batch_{img_name}_{device}"
                results[key] = benchmark_beaker_batch(batch_dir, "cutout", device)
                if results[key].success:
                    mean_time = (
                        results[key].get_stats()["total_time"]["mean"] / 20
                    )  # Per image
                    print(f"✅ ({mean_time:.0f}ms/img)")
                else:
                    print("❌")

            # rembg batch
            print("  rembg batch...", end=" ", flush=True)
            key = f"rembg_batch_{img_name}"
            results[key] = benchmark_rembg_batch(batch_dir)
            if results[key].success:
                mean_time = (
                    results[key].get_stats()["total_time"]["mean"] / 20
                )  # Per image
                print(f"✅ ({mean_time:.0f}ms/img)")
            else:
                print("❌")

        finally:
            # Clean up batch directory
            shutil.rmtree(batch_dir, ignore_errors=True)

    # Print and save results
    print_results(results)
    save_results_json(results)

    print("\n🎉 Benchmark completed!")


if __name__ == "__main__":
    main()
