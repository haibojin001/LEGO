from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .store import Database, get_db

CONNECTION_ID = "automate_cloud"


@dataclass
class Session:
    token: str
    email: str
    tier: str
    expires_at: float | None = None


class CloudUnavailable(Exception):
    """Raised when the cloud endpoint is unavailable or unconfigured."""


def cloud_url() -> str:
    configured = os.environ.get("AUTOMATE_CLOUD_URL") or ""
    return configured.rstrip("/")


def is_configured() -> bool:
    return bool(cloud_url())


def get_session(db: Database | None = None) -> Session | None:
    database = db or get_db()
    record = database.get_connection(CONNECTION_ID, decrypt=True)

    if not record or not record.get("token"):
        return None

    expiration = record.get("expires_at")
    if expiration and time.time() > float(expiration):
        return None

    metadata = json.loads(record.get("metadata_json") or "{}")
    return Session(
        token=record["token"],
        email=metadata.get("email", ""),
        tier=metadata.get("tier", "free"),
        expires_at=expiration,
    )


def store_session(
    db: Database,
    *,
    token: str,
    email: str,
    tier: str,
    expires_at: float | None = None,
) -> Session:
    db.upsert_connection(
        id=CONNECTION_ID,
        display_name="autoMate Cloud",
        auth_kind="bearer",
        status="connected",
        token=token,
        expires_at=expires_at,
        metadata={"email": email, "tier": tier},
    )
    return Session(
        token=token,
        email=email,
        tier=tier,
        expires_at=expires_at,
    )


def clear_session(db: Database) -> None:
    db.delete_connection(CONNECTION_ID)


def _http(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    token: str | None = None,
    timeout: int = 15,
) -> dict:
    base_url = cloud_url()
    if not base_url:
        raise CloudUnavailable("AUTOMATE_CLOUD_URL not configured")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    encoded = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base_url + path,
        data=encoded,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        try:
            response_data = json.loads(error.read().decode())
        except Exception:
            response_data = {"error": str(error)}
        raise CloudUnavailable(
            response_data.get("error") or f"HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise CloudUnavailable(f"unreachable: {error.reason}") from error


def login(db: Database, *, email: str, password: str) -> Session:
    result = _http(
        "POST",
        "/api/auth/login",
        body={"email": email, "password": password},
    )
    return store_session(
        db,
        token=result["token"],
        email=email,
        tier=result.get("tier", "free"),
        expires_at=result.get("expires_at"),
    )


def logout(db: Database) -> None:
    current = get_session(db)
    if current:
        try:
            _http(
                "POST",
                "/api/auth/logout",
                token=current.token,
                timeout=5,
            )
        except CloudUnavailable:
            pass
    clear_session(db)


def me(db: Database) -> dict:
    current = get_session(db)
    if not current:
        return {
            "logged_in": False,
            "cloud_configured": is_configured(),
        }
    return {
        "logged_in": True,
        "email": current.email,
        "tier": current.tier,
        "cloud_configured": True,
    }