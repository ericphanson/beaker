#!/usr/bin/env python3
"""
Rename existing Git tags and GitHub releases to the new format.

This script:
1. Lists all existing tags
2. Identifies tags that need to be renamed (old format: v1.0.0 -> new format: bird-head-detector-v1.0.0)
3. For each tag to be renamed:
   - Downloads the release assets (if it's a GitHub release)
   - Creates a new tag with the new name
   - Creates a new GitHub release with the new name and all assets
   - Deletes the old tag and release
4. Pushes all changes to the remote repository

Requirements:
- gh CLI tool installed and authenticated
- Clean git repository (no uncommitted changes)
- Write access to the repository

Usage:
    # Preview what will be renamed (dry run)
    uv run python rename_tags.py --dry-run

    # Actually perform the rename
    uv run python rename_tags.py

    # Force rename even if working directory is not clean
    uv run python rename_tags.py --force
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


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
        print("\n   Please commit or stash your changes before renaming tags.")
        return False

    print("✅ Repository is clean")
    return True


def get_repo_info():
    """Get repository owner/name from remote origin."""
    stdout, stderr = run_command("git remote get-url origin", check=False)
    if not stdout or "github.com" not in stdout:
        # Fallback to hardcoded repo info
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


def get_existing_tags():
    """Get list of existing git tags."""
    stdout, stderr = run_command("git tag --sort=-version:refname")
    if stdout:
        return stdout.split("\n")
    return []


def get_tags_to_rename():
    """Identify tags that need to be renamed to the new format."""
    existing_tags = get_existing_tags()
    tags_to_rename = []

    for tag in existing_tags:
        # Skip tags that already have the new format
        if tag.startswith("bird-head-detector-"):
            continue

        # Skip tags that don't look like version tags
        if not tag.startswith("v") or not any(char.isdigit() for char in tag):
            continue

        # This tag needs to be renamed
        new_tag = f"bird-head-detector-{tag}"
        tags_to_rename.append((tag, new_tag))

    return tags_to_rename


def get_release_info(tag):
    """Get release information for a tag."""
    repo = get_repo_info()
    cmd = f"gh release view {tag} --repo {repo} --json name,body,assets,isDraft,isPrerelease"
    stdout, stderr = run_command(cmd, check=False)

    if not stdout:
        return None

    try:
        import json

        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def download_release_assets(tag, target_dir):
    """Download all assets from a release."""
    repo = get_repo_info()
    cmd = f'gh release download {tag} --repo {repo} --dir "{target_dir}"'
    stdout, stderr = run_command(cmd, timeout=300, check=False)

    if stderr and "not found" not in stderr.lower():
        print(f"⚠️ Warning downloading assets: {stderr}")

    # Return list of downloaded files
    downloaded_files = []
    if target_dir.exists():
        downloaded_files = list(target_dir.iterdir())

    return downloaded_files


def create_new_release(old_tag, new_tag, release_info, assets_dir, dry_run=False):
    """Create a new release with the new tag name."""
    repo = get_repo_info()

    if dry_run:
        print(f"   [DRY RUN] Would create release {new_tag}")
        if assets_dir and assets_dir.exists():
            assets = list(assets_dir.iterdir())
            if assets:
                print(f"   [DRY RUN] Would upload {len(assets)} assets")
        return True

    # Get the commit hash for the old tag
    stdout, stderr = run_command(f"git rev-list -n 1 {old_tag}")
    if not stdout:
        print(f"❌ Could not find commit for tag {old_tag}")
        return False
    commit_hash = stdout

    # Create the new tag pointing to the same commit
    print(f"📝 Creating new tag: {new_tag}")
    stdout, stderr = run_command(f'git tag -a {new_tag} {commit_hash} -m "Renamed from {old_tag}"')
    if stderr and "already exists" in stderr:
        print(f"❌ Tag {new_tag} already exists!")
        return False

    # Push the new tag
    print("📤 Pushing new tag to remote...")
    stdout, stderr = run_command(f"git push origin {new_tag}")

    # Create the new release
    print("🎁 Creating new release...")

    # Prepare release notes and title
    release_name = release_info.get("name", new_tag)
    release_body = release_info.get("body", "")

    # Update the title to use new tag name
    if old_tag in release_name:
        release_name = release_name.replace(old_tag, new_tag)

    # Add a note about the rename
    if release_body:
        release_body += f"\n\n---\n*This release was renamed from `{old_tag}` to follow the new naming convention.*"
    else:
        release_body = (
            f"*This release was renamed from `{old_tag}` to follow the new naming convention.*"
        )

    # Write release notes to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(release_body)
        notes_file = f.name

    try:
        # Create the release
        cmd_parts = [
            f"gh release create {new_tag}",
            f'--title "{release_name}"',
            f'--notes-file "{notes_file}"',
            f"--repo {repo}",
        ]

        if release_info.get("isDraft", False):
            cmd_parts.append("--draft")
        if release_info.get("isPrerelease", False):
            cmd_parts.append("--prerelease")

        cmd = " ".join(cmd_parts)
        stdout, stderr = run_command(cmd)

        if stderr and "already exists" in stderr:
            print(f"❌ Release {new_tag} already exists!")
            return False

        print("✅ New release created successfully!")
    finally:
        # Clean up temporary file
        try:
            Path(notes_file).unlink()
        except:
            pass

    # Upload assets if any
    if assets_dir and assets_dir.exists():
        assets = list(assets_dir.iterdir())
        if assets:
            print(f"📦 Uploading {len(assets)} assets...")
            asset_paths = [f'"{asset}"' for asset in assets]
            assets_str = " ".join(asset_paths)

            upload_cmd = f"gh release upload {new_tag} {assets_str} --repo {repo}"
            stdout, stderr = run_command(upload_cmd, timeout=300)

            if stderr:
                print(f"⚠️ Warning during asset upload: {stderr}")

            print(f"✅ Uploaded {len(assets)} assets")

    return True


def delete_old_release(tag, dry_run=False):
    """Delete the old tag and release."""
    repo = get_repo_info()

    if dry_run:
        print(f"   [DRY RUN] Would delete release and tag {tag}")
        return True

    # Delete the GitHub release
    print("🗑️ Deleting old release...")
    stdout, stderr = run_command(f"gh release delete {tag} --repo {repo} --yes", check=False)

    # Delete the local tag
    print("🗑️ Deleting old tag locally...")
    stdout, stderr = run_command(f"git tag -d {tag}")

    # Delete the remote tag
    print("🗑️ Deleting old tag from remote...")
    stdout, stderr = run_command(f"git push origin --delete {tag}")

    return True


def rename_tag_and_release(old_tag, new_tag, dry_run=False):
    """Rename a single tag and its associated release."""
    print(f"\n🔄 Processing {old_tag} -> {new_tag}")

    # Check if this tag has a GitHub release
    release_info = get_release_info(old_tag)

    if release_info:
        print(f"📋 Found GitHub release for {old_tag}")

        # Download release assets
        with tempfile.TemporaryDirectory() as temp_dir:
            assets_dir = Path(temp_dir)
            assets = download_release_assets(old_tag, assets_dir)

            if assets:
                print(f"📥 Downloaded {len(assets)} assets")

            # Create new release
            if not create_new_release(old_tag, new_tag, release_info, assets_dir, dry_run):
                return False
    else:
        print(f"📋 No GitHub release found for {old_tag}, creating tag only")

        if not dry_run:
            # Get the commit hash for the old tag
            stdout, stderr = run_command(f"git rev-list -n 1 {old_tag}")
            if not stdout:
                print(f"❌ Could not find commit for tag {old_tag}")
                return False
            commit_hash = stdout

            # Create the new tag pointing to the same commit
            print(f"📝 Creating new tag: {new_tag}")
            stdout, stderr = run_command(
                f'git tag -a {new_tag} {commit_hash} -m "Renamed from {old_tag}"'
            )
            if stderr and "already exists" in stderr:
                print(f"❌ Tag {new_tag} already exists!")
                return False

            # Push the new tag
            print("📤 Pushing new tag to remote...")
            stdout, stderr = run_command(f"git push origin {new_tag}")
        else:
            print(f"   [DRY RUN] Would create tag {new_tag}")

    # Delete the old tag and release
    if not delete_old_release(old_tag, dry_run):
        return False

    print(f"✅ Successfully renamed {old_tag} -> {new_tag}")
    return True


def main():
    """Main rename process."""
    parser = argparse.ArgumentParser(
        description="Rename Git tags and GitHub releases to new format"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be renamed without making changes"
    )
    parser.add_argument(
        "--force", action="store_true", help="Proceed even if working directory is not clean"
    )

    args = parser.parse_args()

    print("🏷️ Tag and Release Renaming Script")
    print("=" * 45)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Check if repo is clean (unless forced)
    if not args.force and not check_repo_clean():
        print("\n💡 Use --force to proceed anyway (not recommended)")
        sys.exit(1)

    # Get tags that need to be renamed
    tags_to_rename = get_tags_to_rename()

    if not tags_to_rename:
        print("✅ No tags found that need renaming!")
        print("   All tags are already in the correct format.")
        sys.exit(0)

    print(f"\n📋 Found {len(tags_to_rename)} tags to rename:")
    for old_tag, new_tag in tags_to_rename:
        print(f"   {old_tag} -> {new_tag}")

    if not args.dry_run:
        print(f"\n⚠️  WARNING: This will rename {len(tags_to_rename)} tags and releases.")
        print("   This action cannot be easily undone!")

        confirm = input("\n❓ Proceed with renaming? (y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("❌ Rename cancelled")
            sys.exit(0)

    # Rename each tag
    success_count = 0
    for old_tag, new_tag in tags_to_rename:
        try:
            if rename_tag_and_release(old_tag, new_tag, args.dry_run):
                success_count += 1
            else:
                print(f"❌ Failed to rename {old_tag}")
        except Exception as e:
            print(f"❌ Error renaming {old_tag}: {e}")

    # Summary
    print("\n📊 Summary:")
    if args.dry_run:
        print(f"   Would rename {len(tags_to_rename)} tags/releases")
    else:
        print(f"   Successfully renamed: {success_count}/{len(tags_to_rename)}")

        if success_count == len(tags_to_rename):
            print("🎉 All tags and releases renamed successfully!")
        else:
            print("⚠️ Some renames failed. Check the output above for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
