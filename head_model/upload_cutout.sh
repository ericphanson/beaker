#!/usr/bin/env bash
#
# Uses GitHub CLI (https://cli.github.com/) to:
#   1. Create an annotated git tag.
#   2. Push the tag to the default remote.
#   3. Create (or reuse) a GitHub release from that tag.
#   4. Download the ONNX model, verify its MD5 checksum,
#      and upload it as a release asset.
#
# Exit on first error, unset variable, or failed pipe.
set -euo pipefail

TAG="beaker-cutout-model-v1"
ASSET_URL="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
ASSET_FILE="isnet-general-use.onnx"
MD5_EXPECTED="fc16ebd8b0c10d971d3513d564d01e29"

# --- Dependency checks -------------------------------------------------------
for cmd in git gh curl md5sum; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Error: '$cmd' is required but not installed." >&2
    exit 1
  }
done

# --- Repository sanity -------------------------------------------------------
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Error: run this script inside a git repository." >&2
  exit 1
}

# --- Tag creation & push -----------------------------------------------------
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag '$TAG' already exists locally."
else
  git tag -a "$TAG" -m "$TAG"
  git push origin "$TAG"
fi

# --- Prepare temp workspace --------------------------------------------------
tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

curl -L --fail --silent --show-error "$ASSET_URL" -o "$tmpdir/$ASSET_FILE"

# --- MD5 verification --------------------------------------------------------
md5_downloaded="$(md5sum "$tmpdir/$ASSET_FILE" | awk '{print $1}')"
if [[ "$md5_downloaded" != "$MD5_EXPECTED" ]]; then
  echo "MD5 mismatch: expected $MD5_EXPECTED, got $md5_downloaded" >&2
  exit 1
fi
echo "Checksum verified."

# --- Release creation --------------------------------------------------------
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release '$TAG' already exists."
else
  gh release create "$TAG" \
    --title "$TAG" \
    --notes "ISNet-General ONNX model for Beaker cut-out (v1)."
fi

# --- Asset upload (overwrite if present) -------------------------------------
gh release upload "$TAG" "$tmpdir/$ASSET_FILE" --clobber

echo "✔ Release '$TAG' is ready with asset '$ASSET_FILE'."
