#!/usr/bin/env python3
"""
Release script for bird-head-detector models.

This script:
1. Checks if the repository is clean (no uncommitted changes)
2. Shows existing tags/versions
3. Prompts for a new version number
4. Creates a git tag
5. Uploads the best model as a GitHub release asset
6. Pushes everything to the remote repository

Requirements:
- gh CLI tool installed and authenticated
- Clean git repository (no uncommitted changes)
- Trained model exists at the expected location

Usage:
    uv run python release.py
"""

import subprocess
import sys
from pathlib import Path
import re
import os


def run_command(cmd, capture=True, check=True):
    """Run a shell command and return the result."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            return result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return None, None
    except subprocess.CalledProcessError as e:
        if capture:
            return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
        raise


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")

    # Check if gh CLI is installed
    stdout, stderr = run_command("gh --version", check=False)
    if not stdout:
        print("❌ GitHub CLI (gh) is not installed.")
        print("   Install it from: https://cli.github.com/")
        return False
    print(f"✅ GitHub CLI found: {stdout.split()[2]}")

    # Check if gh is authenticated
    stdout, stderr = run_command("gh auth status", check=False)
    if "Logged in to github.com" not in stderr and "Logged in to github.com" not in stdout:
        print("❌ GitHub CLI is not authenticated.")
        print("   Run: gh auth login")
        return False
    print("✅ GitHub CLI authenticated")

    # Check if we're in a git repository
    stdout, stderr = run_command("git rev-parse --git-dir", check=False)
    if not stdout:
        print("❌ Not in a git repository.")
        return False
    print("✅ Git repository detected")

    return True


def check_repo_clean():
    """Check if the repository has no uncommitted changes."""
    print("🧹 Checking repository status...")

    # Check for uncommitted changes
    stdout, stderr = run_command("git status --porcelain")
    if stdout:
        print("❌ Repository has uncommitted changes:")
        print(stdout)
        print("\n   Please commit or stash your changes before releasing.")
        return False

    print("✅ Repository is clean")
    return True


def get_existing_tags():
    """Get list of existing git tags."""
    stdout, stderr = run_command("git tag --sort=-version:refname")
    if stdout:
        return stdout.split('\n')
    return []


def validate_version(version):
    """Validate version format (semantic versioning)."""
    pattern = r'^v?\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$'
    return re.match(pattern, version) is not None


def get_model_path():
    """Find the best model file."""
    # Try different possible paths
    possible_paths = [
        "runs/detect/bird_head_yolov8n/weights/best.pt",
        "runs/detect/bird_head_yolov8n_debug/weights/best.pt",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return Path(path)

    return None


def create_release(version, model_path):
    """Create a GitHub release with the model as an asset."""
    print(f"🚀 Creating release {version}...")

    # Ensure version starts with 'v'
    if not version.startswith('v'):
        version = f'v{version}'

    # Create git tag
    print(f"📝 Creating git tag: {version}")
    stdout, stderr = run_command(f'git tag -a {version} -m "Release {version}"')
    if stderr and "already exists" in stderr:
        print(f"❌ Tag {version} already exists!")
        return False

    # Push tag to remote
    print(f"📤 Pushing tag to remote...")
    stdout, stderr = run_command(f"git push origin {version}")

    # Create release with model asset
    print(f"🎁 Creating GitHub release...")
    release_title = f"Bird Head Detector {version}"
    release_notes = f"""# Bird Head Detector {version}

This release includes a trained YOLOv8n model for bird head detection.

## Model Details
- **Architecture**: YOLOv8n
- **Dataset**: CUB-200-2011 (bird head parts)
- **Training Images**: ~5,990 (train) + ~5,794 (val)
- **Classes**: 1 (bird_head)

## Usage
Download the model file and use with the inference script:
```bash
uv run python infer.py --model {model_path.name} --source your_image.jpg --show
```

## Files
- `{model_path.name}`: Trained YOLOv8n model weights
"""

    # Create the release
    cmd = f'gh release create {version} "{model_path}" --title "{release_title}" --notes "{release_notes}"'
    stdout, stderr = run_command(cmd)

    if stderr and "already exists" in stderr:
        print(f"❌ Release {version} already exists!")
        # Clean up the tag we just created
        run_command(f"git tag -d {version}", check=False)
        run_command(f"git push origin --delete {version}", check=False)
        return False

    print(f"✅ Release {version} created successfully!")
    print(f"🔗 View at: https://github.com/{get_repo_info()}/releases/tag/{version}")

    return True


def get_repo_info():
    """Get repository owner/name from remote origin."""
    stdout, stderr = run_command("git remote get-url origin")
    if "github.com" in stdout:
        # Extract owner/repo from URL
        if stdout.startswith("https://"):
            # https://github.com/owner/repo.git
            parts = stdout.replace("https://github.com/", "").replace(".git", "").split("/")
        else:
            # git@github.com:owner/repo.git
            parts = stdout.replace("git@github.com:", "").replace(".git", "").split("/")

        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

    return "unknown/unknown"


def main():
    """Main release process."""
    print("🚀 Bird Head Detector Release Script")
    print("=" * 40)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Check if repo is clean
    if not check_repo_clean():
        sys.exit(1)

    # Find model file
    model_path = get_model_path()
    if not model_path:
        print("❌ No trained model found!")
        print("   Expected locations:")
        print("   - runs/detect/bird_head_yolov8n/weights/best.pt")
        print("   - runs/detect/bird_head_yolov8n_debug/weights/best.pt")
        print("\n   Train a model first using: uv run python train.py")
        sys.exit(1)

    print(f"✅ Found model: {model_path}")
    model_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Size: {model_size:.1f} MB")

    # Show existing tags
    existing_tags = get_existing_tags()
    if existing_tags:
        print(f"\n📋 Existing versions:")
        for tag in existing_tags[:10]:  # Show last 10 tags
            print(f"   {tag}")
        if len(existing_tags) > 10:
            print(f"   ... and {len(existing_tags) - 10} more")
    else:
        print("\n📋 No existing versions found")

    # Get version from user
    print(f"\n🏷️  Enter new version number:")
    while True:
        version = input("   Version (e.g., 1.0.0 or v1.0.0): ").strip()

        if not version:
            print("   Version cannot be empty!")
            continue

        if not validate_version(version):
            print("   Invalid version format! Use semantic versioning (e.g., 1.0.0)")
            continue

        # Normalize version (add 'v' prefix if missing)
        normalized_version = version if version.startswith('v') else f'v{version}'

        if normalized_version in existing_tags:
            print(f"   Version {normalized_version} already exists!")
            continue

        break

    # Confirm release
    print(f"\n📋 Release Summary:")
    print(f"   Version: {normalized_version}")
    print(f"   Model: {model_path}")
    print(f"   Size: {model_size:.1f} MB")
    print(f"   Repository: {get_repo_info()}")

    confirm = input(f"\n❓ Create release {normalized_version}? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Release cancelled")
        sys.exit(0)

    # Create the release
    if create_release(version, model_path):
        print(f"\n🎉 Release {normalized_version} completed successfully!")
    else:
        print(f"\n❌ Release failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
