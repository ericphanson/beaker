#!/bin/bash

# Script to create a git tag, GitHub release, and upload folder contents as assets
# Usage: ./create_release.sh <tag_name> <release_title> <folder_path> [release_notes]

set -e  # Exit on any error

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <tag_name> <release_title> <folder_path> [release_notes]"
    echo "Example: $0 v1.0.0 'Release v1.0.0' ./output4 'Initial release with trained models'"
    exit 1
fi

TAG_NAME="$1"
RELEASE_TITLE="$2"
FOLDER_PATH="$3"
RELEASE_NOTES="${4:-Release $TAG_NAME}"

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder '$FOLDER_PATH' does not exist"
    exit 1
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: gh CLI is not installed. Please install it first:"
    echo "brew install gh"
    exit 1
fi

# Check if we're authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub. Please run 'gh auth login' first"
    exit 1
fi

echo "Creating release '$RELEASE_TITLE' with tag '$TAG_NAME'..."

# Create the release (this also creates the tag)
gh release create "$TAG_NAME" \
    --title "$RELEASE_TITLE" \
    --notes "$RELEASE_NOTES"

echo "Release created successfully!"

# Upload all files from the specified folder
echo "Uploading assets from '$FOLDER_PATH'..."

# Find all files in the folder (not subdirectories)
find "$FOLDER_PATH" -maxdepth 1 -type f | while read -r file; do
    echo "Uploading: $(basename "$file")"
    gh release upload "$TAG_NAME" "$file" --clobber
done

echo "All assets uploaded successfully!"
echo "Release URL: $(gh release view "$TAG_NAME" --json url --jq .url)"
