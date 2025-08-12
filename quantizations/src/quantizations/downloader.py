"""
Download ONNX models from GitHub releases.
"""

import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com/repos/ericphanson/beaker"


def get_latest_release_by_pattern(pattern: str) -> dict[str, Any] | None:
    """Get the latest GitHub release matching a pattern."""
    try:
        response = requests.get(f"{GITHUB_API_BASE}/releases", timeout=30)
        response.raise_for_status()
        releases = response.json()

        # Find the latest release matching the pattern
        for release in releases:
            tag_name = release.get("tag_name", "")
            if pattern in tag_name:
                logger.debug(f"Found matching release: {tag_name}")
                return release

        logger.error(f"No release found matching pattern: {pattern}")
        return None

    except requests.RequestException as e:
        logger.error(f"Failed to fetch releases: {e}")
        return None


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to the specified path."""
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Downloaded: {output_path}")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_detect_model(output_dir: Path) -> Path | None:
    """Download the latest head detection model."""
    release = get_latest_release_by_pattern("bird-head-detector")
    if not release:
        return None

    # Find the ONNX model asset
    assets = release.get("assets", [])
    onnx_asset = None

    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".onnx"):
            onnx_asset = asset
            break

    if not onnx_asset:
        logger.error("No ONNX model found in head detector release")
        return None

    download_url = onnx_asset.get("browser_download_url")
    if not download_url:
        logger.error("No download URL found for ONNX asset")
        return None

    model_name = onnx_asset.get("name", "bird-head-detector.onnx")
    output_path = output_dir / model_name

    if download_file(download_url, output_path):
        return output_path
    return None


def download_cutout_model(output_dir: Path) -> Path | None:
    """Download the latest cutout model."""
    release = get_latest_release_by_pattern("cutout-model")
    if not release:
        return None

    # Find the ONNX model asset
    assets = release.get("assets", [])
    onnx_asset = None

    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".onnx"):
            onnx_asset = asset
            break

    if not onnx_asset:
        logger.error("No ONNX model found in cutout model release")
        return None

    download_url = onnx_asset.get("browser_download_url")
    if not download_url:
        logger.error("No download URL found for ONNX asset")
        return None

    model_name = onnx_asset.get("name", "cutout-model.onnx")
    output_path = output_dir / model_name

    if download_file(download_url, output_path):
        return output_path
    return None


def get_model_checksum(model_path: Path) -> str:
    """Calculate SHA256 checksum of a model file."""
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()
