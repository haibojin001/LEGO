from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ... import files as F
from ...extension_bus import bus
from ...settings import PATHS
from ...version import __version__
from ._deps import state

router = APIRouter(tags=["system"])
log = logging.getLogger("automate.system")


@router.get("/health")
def health():
    return {"ok": True, "version": __version__}


@router.get("/status")
def status(s=Depends(state)):
    providers = s.db.list_providers()
    connections = s.db.list_connections()
    return {
        "version": __version__,
        "active_provider": s.providers.active_provider_id(),
        "active_model": s.providers.active_model(),
        "providers": len(providers),
        "providers_configured": sum(
            1 for provider in providers if provider["api_key_set"]
        ),
        "integrations": len(connections),
        "integrations_connected": sum(
            1 for connection in connections if connection["status"] == "connected"
        ),
        "tools": len(s.registry.all()),
        "extension_connected": bus.connected,
    }


class StorageIn(BaseModel):
    path: str


@router.get("/system/storage")
def get_storage(s=Depends(state)):
    selected = s.db.get_setting(F.STORAGE_DIR_KEY)
    return {
        "current": str(F.storage_dir(s.db)),
        "default": str(PATHS.files),
        "is_default": not selected,
        "used_bytes": F.total_size(s.db),
    }


@router.put("/system/storage")
def put_storage(body: StorageIn, s=Depends(state)):
    try:
        result = F.set_storage_dir(s.db, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"current": str(result), "is_default": False}


DEFAULT_REPO = "yuruotong1/autoMate"
DEFAULT_API = f"https://api.github.com/repos/{DEFAULT_REPO}/releases/latest"

_CACHE: dict = {"at": 0.0, "data": None, "url": ""}
_CACHE_TTL = 6 * 60 * 60


@router.get("/system/update_check")
def update_check(mirror: str = Query(""), force: bool = Query(False)):
    source_url = (os.environ.get("AUTOMATE_UPDATE_URL") or DEFAULT_API).strip()
    identity = f"{source_url}|{mirror}"
    checked_at = time.time()

    if (
        not force
        and _CACHE.get("url") == identity
        and checked_at - _CACHE["at"] < _CACHE_TTL
    ):
        return _CACHE["data"]

    try:
        request = urllib.request.Request(
            source_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "automate-update-check",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            release = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        return {
            "current": __version__,
            "error": f"update server unreachable: {exc.reason}",
        }
    except (json.JSONDecodeError, KeyError) as exc:
        return {
            "current": __version__,
            "error": f"malformed update payload: {exc}",
        }
    except Exception as exc:
        return {"current": __version__, "error": str(exc)}

    latest = (release.get("tag_name") or "").lstrip("v")
    release_url = release.get("html_url", "")
    release_body = release.get("body") or ""
    assets = release.get("assets") or []

    data = {
        "current": __version__,
        "latest": latest or None,
        "newer": _is_newer(latest, __version__) if latest else False,
        "release_notes_url": release_url,
        "release_notes_body": release_body[:4000],
        "download_urls": _index_assets(assets, mirror=mirror),
        "source": "mirror" if mirror else "github",
    }
    _CACHE.update({"at": checked_at, "data": data, "url": identity})
    return data


_PLATFORM_PATTERNS = {
    "windows": ("windows-x64.zip",),
    "macos": ("macos-arm64.zip", "darwin-arm64.zip"),
    "linux": ("linux-x64.tar.gz", "linux-amd64.tar.gz"),
    "wheel": (".whl",),
    "sdist": (".tar.gz",),
    "extension": ("extension.zip",),
}


def _index_assets(assets: list[dict], *, mirror: str) -> dict[str, str]:
    indexed: dict[str, str] = {}
    used_names: set[str] = set()

    for platform, suffixes in _PLATFORM_PATTERNS.items():
        for asset in assets:
            filename = asset.get("name") or ""
            if filename in used_names:
                continue
            for suffix in suffixes:
                if filename.endswith(suffix):
                    download_url = asset.get("browser_download_url") or ""
                    indexed[platform] = _via_mirror(download_url, mirror)
                    used_names.add(filename)
                    break

    return indexed


def _via_mirror(url: str, mirror: str) -> str:
    if not mirror or not url.startswith("https://github.com/"):
        return url

    prefix = mirror.rstrip("/")
    if not prefix.startswith("http"):
        prefix = f"https://{prefix}"
    return f"{prefix}/{url}"


def _is_newer(latest: str, current: str) -> bool:
    def version_parts(value: str) -> list[int]:
        return [
            int(part)
            for part in value.replace("-", ".").split(".")
            if part.isdigit()
        ]

    try:
        return version_parts(latest) > version_parts(current)
    except Exception:
        return False