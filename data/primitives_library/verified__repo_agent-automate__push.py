from __future__ import annotations

import base64
import json
import logging
import time
from typing import Iterable

from .store import Database

log = logging.getLogger("automate.push")


def _b64url(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(data)
    return encoded.rstrip(b"=").decode()


def _generate_vapid() -> tuple[str, str]:
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.private_numbers()
    scalar = numbers.private_value.to_bytes(32, byteorder="big")

    public = private_key.public_key().public_numbers()
    point = (
        b"\x04"
        + public.x.to_bytes(32, byteorder="big")
        + public.y.to_bytes(32, byteorder="big")
    )
    return _b64url(point), _b64url(scalar)


def ensure_vapid(db: Database) -> dict:
    existing = db.fetchone("SELECT * FROM push_keys WHERE id = 1")
    if existing:
        return {
            "public": existing["vapid_public"],
            "private": existing["vapid_private"],
            "subject": existing["vapid_subject"],
        }

    public_key, private_key = _generate_vapid()
    subject = "mailto:admin@automate.local"
    db.execute(
        "INSERT INTO push_keys "
        "(id, vapid_private, vapid_public, vapid_subject) "
        "VALUES (1, ?, ?, ?)",
        (private_key, public_key, subject),
    )
    return {"public": public_key, "private": private_key, "subject": subject}


def save_subscription(
    db: Database,
    *,
    endpoint: str,
    p256dh: str,
    auth: str,
    user_agent: str = "",
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO push_subscriptions "
        "(endpoint, p256dh, auth, user_agent, created_at) VALUES (?, ?, ?, ?, ?)",
        (endpoint, p256dh, auth, user_agent, time.time()),
    )


def list_subscriptions(db: Database) -> list[dict]:
    return db.fetchall("SELECT * FROM push_subscriptions")


def delete_subscription(db: Database, endpoint: str) -> None:
    db.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))


def _to_pem(b64url_priv: str) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    scalar_bytes = base64.urlsafe_b64decode(b64url_priv + "==")
    scalar = int.from_bytes(scalar_bytes, byteorder="big")
    key = ec.derive_private_key(scalar, ec.SECP256R1())
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem_bytes.decode()


def push(
    db: Database,
    *,
    title: str,
    body: str = "",
    url: str = "/",
    tag: str = "automate",
) -> dict:
    keys = ensure_vapid(db)
    subscriptions = list_subscriptions(db)

    if not subscriptions:
        return {"sent": 0, "failed": 0, "reason": "no subscribers"}

    try:
        from pywebpush import WebPushException, webpush  # type: ignore
    except ImportError:
        log.warning(
            "pywebpush not installed; push disabled. Install with `pip install pywebpush`."
        )
        return {"sent": 0, "failed": 0, "reason": "pywebpush not installed"}

    message = json.dumps(
        {"title": title, "body": body, "url": url, "tag": tag}
    )
    sent = 0
    failed = 0

    for subscription in subscriptions:
        endpoint = subscription["endpoint"]
        try:
            webpush(
                subscription_info={
                    "endpoint": endpoint,
                    "keys": {
                        "p256dh": subscription["p256dh"],
                        "auth": subscription["auth"],
                    },
                },
                data=message,
                vapid_private_key=_to_pem(keys["private"]),
                vapid_claims={"sub": keys["subject"]},
            )
            sent += 1
        except WebPushException as exc:
            failed += 1
            log.warning("push failed for %s: %s", endpoint[:40], exc)
            if exc.response and exc.response.status_code in (404, 410):
                delete_subscription(db, endpoint)
        except Exception as exc:
            failed += 1
            log.warning("push error: %s", exc)

    return {"sent": sent, "failed": failed}