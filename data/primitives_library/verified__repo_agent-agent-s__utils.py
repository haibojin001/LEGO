"""General utility."""

import io
import os
import platform
import zipfile

import requests


def download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform=platform.system().lower(),
):
    """Download and extract the appropriate KB ZIP file for the current OS.

    Args:
        version (str): Prefix in the asset name (e.g., "s1" or "s2")
        release_tag (str): Tag of the release that has the assets (e.g., "v0.2.2")
        download_dir (str): Where to extract the downloaded files
        platform (str): OS (e.g., "windows", "darwin", "linux")
    """
    if platform not in ["windows", "darwin", "linux"]:
        raise RuntimeError(f"Unsupported OS: {platform}")

    asset_name = f"{version}_{platform}.zip"
    download_url = (
        f"https://github.com/simular-ai/Agent-S/releases/download/"
        f"{release_tag}/{asset_name}"
    )

    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading {asset_name} from {download_url} ...")
    response = requests.get(download_url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {asset_name}. "
            f"HTTP status: {response.status_code} - {response.reason}"
        )

    archive = io.BytesIO(response.content)
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    print(f"Extracted {asset_name} to ./{download_dir}")