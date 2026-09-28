import logging
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    """Determine whether url is syntactically valid and optionally reachable."""
    try:
        text = str(url)
        parsed = urllib.parse.urlparse(text)
        assert parsed.scheme and parsed.netloc
        if check:
            return urllib.request.urlopen(url).getcode() == 200
        return True
    except (AssertionError, urllib.error.URLError):
        return False


def gsutil_getsize(url=""):
    """Return an object's byte size as reported by gsutil."""
    result = subprocess.check_output(["gsutil", "du", url], encoding="utf-8")
    return int(result.split()[0]) if result else 0


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Use curl to download url into filename."""
    quiet = "sS" if silent else ""
    completed = subprocess.run(
        [
            "curl",
            "-#",
            f"-{quiet}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ],
        check=False,
    )
    return completed.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """Download file and remove it when the resulting download is too small."""
    from utils.general import LOGGER

    destination = Path(file)
    message = f"Downloaded file '{destination}' does not exist or size is < min_bytes={min_bytes}"

    try:
        LOGGER.info(f"Downloading {url} to {destination}...")
        torch.hub.download_url_to_file(url, str(destination), progress=LOGGER.level <= logging.INFO)
        assert destination.exists() and destination.stat().st_size > min_bytes, message
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        alternate = url2 or url
        LOGGER.warning(f"{exc}\nRe-attempting {alternate} to {destination}...")
        curl_download(alternate, destination)
    finally:
        if not destination.exists() or destination.stat().st_size < min_bytes:
            if destination.exists():
                destination.unlink()
            LOGGER.error(f"{message}\n{error_msg}")
        LOGGER.info("")


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """Return a local path, downloading recognized release assets when necessary."""
    from utils.general import LOGGER

    def github_assets(repository, version="latest"):
        suffix = f"tags/{version}" if version != "latest" else version
        payload = requests.get(f"https://api.github.com/repos/{repository}/releases/{suffix}").json()
        return payload["tag_name"], [asset["name"] for asset in payload["assets"]]

    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name

        if str(file).startswith(("http:/", "https:/")):
            url = str(file).replace(":/", "://")
            file = name.split("?")[0]
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        assets = [
            f"yolov5{size}{suffix}.pt"
            for size in "nsmlx"
            for suffix in ("", "6", "-cls", "-seg")
        ]

        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)
            except Exception:
                try:
                    tag = subprocess.check_output(
                        "git tag", shell=True, stderr=subprocess.STDOUT
                    ).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)