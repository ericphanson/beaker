#!/usr/bin/env python3
"""
Backfill ONNX models to existing GitHub releases.

This script:
1. Lists all existing releases that have .pt files
2. Downloads the .pt files
3. Converts them to ONNX format
4. Uploads both .pt (with standardized name) and .onnx files back to the releases

Requirements:
- gh CLI tool installed and authenticated
- ultralytics package installed
- ONNX dependencies (install with: uv sync --extra onnx)

Usage:
    # Dry run - show what would be done
    uv run python backfill_onnx.py --dry-run

    # Backfill all releases
    uv run python backfill_onnx.py

    # Backfill specific release
    uv run python backfill_onnx.py --tag bird-head-detector-v1.0.0
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics package not found.")
    print("   Install with: uv sync --extra onnx")
    sys.exit(1)


def run_command(cmd, capture=True, check=True, timeout=120):
    """Run a shell command and return the result."""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=check, timeout=timeout
            )
            return result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check, timeout=timeout)
            return None, None
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds: {cmd}")
        return "", "Command timed out"
    except subprocess.CalledProcessError as e:
        if capture:
            return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
        raise


def check_gh_cli():
    """Check if GitHub CLI is available and authenticated."""
    # Check if gh CLI is installed
    stdout, stderr = run_command("gh --version", check=False)
    if not stdout:
        print("❌ GitHub CLI (gh) is not installed.")
        print("   Install it from: https://cli.github.com/")
        return False

    # Check if gh is authenticated
    stdout, stderr = run_command("gh auth status", check=False)
    if "Logged in to github.com" not in stderr and "Logged in to github.com" not in stdout:
        print("❌ GitHub CLI is not authenticated.")
        print("   Run: gh auth login")
        return False

    return True


def get_repo_info():
    """Get repository owner/name from remote origin."""
    stdout, stderr = run_command("git remote get-url origin", check=False)

    print(f"🔍 Git remote URL: {stdout}")
    if stderr:
        print(f"⚠️ Git remote stderr: {stderr}")

    if not stdout or "github.com" not in stdout:
        print("⚠️ Could not detect GitHub repository from git remote")
        print("   Please ensure you're in a git repository with GitHub remote")
        # Try to get repo info from gh CLI
        gh_stdout, gh_stderr = run_command("gh repo view --json owner,name", check=False)
        if gh_stdout:
            try:
                import json

                repo_info = json.loads(gh_stdout)
                detected_repo = f"{repo_info['owner']['login']}/{repo_info['name']}"
                print(f"📋 Detected repository via gh CLI: {detected_repo}")
                return detected_repo
            except:
                pass

        # Final fallback
        print("🔄 Using fallback repository: ericphanson/beaker")
        return "ericphanson/beaker"

    # Extract owner/repo from URL
    if stdout.startswith("https://"):
        # https://github.com/owner/repo.git
        parts = stdout.replace("https://github.com/", "").replace(".git", "").split("/")
    else:
        # git@github.com:owner/repo.git
        parts = stdout.replace("git@github.com:", "").replace(".git", "").split("/")

    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"

    return "ericphanson/beaker"


def get_releases(repo, tag=None):
    """Get list of releases from GitHub."""
    if tag:
        # First get basic release info
        cmd = f"gh release view {tag} --repo {repo} --json tagName"
        print(f"🔧 Running: {cmd}")
        stdout, stderr = run_command(cmd, check=False)
        if not stdout:
            print(f"❌ Release {tag} not found")
            if stderr:
                print(f"   Error: {stderr}")
            return []

        try:
            release = json.loads(stdout)
            # Get assets separately
            assets_cmd = f"gh release view {tag} --repo {repo}"
            assets_stdout, assets_stderr = run_command(assets_cmd, check=False)

            # Parse assets from the text output
            assets = []
            if assets_stdout:
                lines = assets_stdout.split("\n")
                for line in lines:
                    # Look for lines that start with "asset:"
                    if line.startswith("asset:"):
                        filename = line.replace("asset:", "").strip()
                        if filename:
                            assets.append({"name": filename})

            release["assets"] = assets
            return [release]
        except json.JSONDecodeError:
            print(f"❌ Could not parse release data for {tag}")
            return []
    else:
        # Get all releases without assets first
        cmd = f"gh release list --repo {repo} --json tagName --limit 50"
        print(f"🔧 Running: {cmd}")
        stdout, stderr = run_command(cmd, check=False)
        if not stdout:
            print("❌ Could not fetch releases")
            if stderr:
                print(f"   Error: {stderr}")
            print("💡 This could be due to:")
            print("   - No releases exist in the repository")
            print("   - Repository name is incorrect")
            print("   - GitHub CLI authentication issues")
            print("   - Repository is private and you don't have access")
            return []

        try:
            releases = json.loads(stdout)
            print(f"📋 Found {len(releases)} total releases")

            # For each release, get the assets
            for release in releases:
                tag_name = release["tagName"]
                assets_cmd = f"gh release view {tag_name} --repo {repo}"
                assets_stdout, assets_stderr = run_command(assets_cmd, check=False)

                # Parse assets from the text output
                assets = []
                if assets_stdout:
                    lines = assets_stdout.split("\n")
                    for line in lines:
                        # Look for lines that start with "asset:"
                        if line.startswith("asset:"):
                            filename = line.replace("asset:", "").strip()
                            if filename:
                                assets.append({"name": filename})
                                print(f"   📄 Found asset: {filename}")

                release["assets"] = assets

            return releases
        except json.JSONDecodeError:
            print("❌ Could not parse releases data")
            print(f"   Raw output: {stdout[:200]}...")
            return []


def generate_onnx_from_pt(pt_file_path, output_dir):
    """Generate ONNX model from PyTorch model."""
    try:
        print(f"🔄 Converting {pt_file_path.name} to ONNX...")

        # Load the model
        model = YOLO(str(pt_file_path))

        # Generate ONNX file path
        onnx_path = output_dir / f"{pt_file_path.stem}.onnx"

        # Export to ONNX with standard settings (same as release.py)
        export_path = model.export(
            format="onnx",
            imgsz=640,  # Standard input size
            dynamic=False,  # Static input shape for better compatibility
            simplify=True,  # Simplify the ONNX model
            opset=11,  # ONNX opset 11 for good compatibility
            half=False,  # Use FP32 for CPU compatibility
        )

        # Move to desired location if needed
        export_path = Path(export_path)
        if export_path != onnx_path:
            if onnx_path.exists():
                onnx_path.unlink()
            export_path.rename(onnx_path)

        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✅ ONNX model generated: {onnx_path.name} ({onnx_size:.1f} MB)")
        return onnx_path

    except Exception as e:
        print(f"❌ Failed to generate ONNX model: {e}")
        return None


def backfill_release(repo, release, dry_run=False):
    """Backfill ONNX model for a specific release."""
    tag_name = release["tagName"]
    assets = release.get("assets", [])

    # Check for different possible PT file names
    pt_filename = None
    for asset in assets:
        if asset["name"] in ["bird-head-detector.pt", "best.pt", "last.pt"]:
            pt_filename = asset["name"]
            break

    # Check if release already has ONNX file
    has_onnx = any(
        asset["name"] in ["bird-head-detector.onnx", "best.onnx", "last.onnx"] for asset in assets
    )

    if not pt_filename:
        print(
            f"⏭️  Skipping {tag_name}: No PyTorch model file found (searched for: bird-head-detector.pt, best.pt, last.pt)"
        )
        return False

    if has_onnx:
        print(f"⏭️  Skipping {tag_name}: Already has ONNX file")
        return False

    print(f"🎯 Processing {tag_name} (found model: {pt_filename})...")

    if dry_run:
        if pt_filename != "bird-head-detector.pt":
            print(
                f"   [DRY RUN] Would download {pt_filename}, rename to bird-head-detector.pt, and generate ONNX"
            )
        else:
            print(f"   [DRY RUN] Would download {pt_filename} and generate ONNX")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download the PT file
        print(f"📥 Downloading {pt_filename} from {tag_name}...")
        cmd = f'gh release download {tag_name} --repo {repo} --pattern "{pt_filename}" --dir "{temp_path}"'
        stdout, stderr = run_command(cmd, timeout=300)

        pt_file = temp_path / pt_filename
        if not pt_file.exists():
            print(f"❌ Failed to download PT file for {tag_name}")
            return False

        # Generate ONNX model
        onnx_file = generate_onnx_from_pt(pt_file, temp_path)
        if not onnx_file:
            return False

        # Upload both PT and ONNX files to release with standardized names
        print(f"📤 Uploading model files to {tag_name}...")

        # Upload the PT file with standardized name (if it's not already named correctly)
        if pt_filename != "bird-head-detector.pt":
            print(f"   📝 Renaming {pt_filename} → bird-head-detector.pt")
            pt_upload_cmd = (
                f'gh release upload {tag_name} --repo {repo} "{pt_file}"#bird-head-detector.pt'
            )
            stdout, stderr = run_command(pt_upload_cmd, timeout=300)

            if stderr and "already exists" not in stderr:
                print(f"⚠️ Warning during PT upload: {stderr}")

        # Upload ONNX file with standardized name
        upload_cmd = (
            f'gh release upload {tag_name} --repo {repo} "{onnx_file}"#bird-head-detector.onnx'
        )
        stdout, stderr = run_command(upload_cmd, timeout=300)

        if stderr and "already exists" not in stderr:
            print(f"⚠️ Warning during ONNX upload: {stderr}")

        print(f"✅ Successfully backfilled models for {tag_name}")
        return True


def main():
    """Main backfill process."""
    parser = argparse.ArgumentParser(description="Backfill ONNX models to existing GitHub releases")
    parser.add_argument("--tag", type=str, help="Specific release tag to backfill")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("🔄 ONNX Backfill Script")
    print("=" * 40)

    if not check_gh_cli():
        sys.exit(1)

    repo = get_repo_info()
    print(f"📋 Repository: {repo}")

    # Get releases
    releases = get_releases(repo, args.tag)
    if not releases:
        print("❌ No releases found")
        sys.exit(1)

    print(f"🔍 Found {len(releases)} release(s) to check")

    if args.dry_run:
        print("🧪 DRY RUN MODE - No changes will be made")

    # Process each release
    processed = 0
    skipped = 0

    for release in releases:
        if backfill_release(repo, release, args.dry_run):
            processed += 1
        else:
            skipped += 1

    print("\n📊 Summary:")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")

    if args.dry_run:
        print("\n💡 Run without --dry-run to actually perform the backfill")
    elif processed > 0:
        print(f"\n🎉 Successfully backfilled {processed} release(s)!")


if __name__ == "__main__":
    main()
