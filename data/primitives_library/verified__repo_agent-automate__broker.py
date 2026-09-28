from __future__ import annotations

import json as _json
import os as _os
import secrets as _secrets
import time as _time
import urllib.error as _urlerror
import urllib.parse as _urlparse
import urllib.request as _urlrequest
from dataclasses import dataclass

DEFAULT_BROKER_URL = "https://broker.automate.cloud"


def broker_url() -> str:
    configured = _os.environ.get("AUTOMATE_OAUTH_BROKER_URL")
    return (configured or DEFAULT_BROKER_URL).rstrip("/")


def make_flow_id() -> str:
    return _secrets.token_urlsafe(32)


@dataclass
class BrokerStartResponse:
    authorize_url: str
    flow_id: str


class BrokerError(RuntimeError):
    pass


def request_authorize_url(
    *,
    provider_id: str,
    flow_id: str,
    local_redirect: str,
    scopes: tuple[str, ...] = (),
) -> str:
    base_url = broker_url()
    query_string = _urlparse.urlencode(
        {
            "flow_id": flow_id,
            "local_redirect": local_redirect,
            "scopes": " ".join(scopes),
        }
    )
    request_url = (
        f"{base_url}/oauth/{provider_id}/start?{query_string}"
    )

    try:
        with _urlrequest.urlopen(request_url, timeout=10) as response:
            payload = _json.loads(response.read().decode())
    except _urlerror.HTTPError as error:
        body = error.read().decode()[:200]
        raise BrokerError(
            f"broker returned {error.code}: {body}"
        ) from error
    except (_urlerror.URLError, TimeoutError) as error:
        raise BrokerError(
            f"broker unreachable at {base_url}: {error}"
        ) from error

    authorize_url = payload.get("authorize_url")
    if not authorize_url:
        raise BrokerError(
            f"broker /start payload missing authorize_url: {payload}"
        )

    return authorize_url


def fetch_result(*, flow_id: str, max_attempts: int = 30) -> dict:
    base_url = broker_url()
    request_url = (
        f"{base_url}/oauth/result?flow_id={_urlparse.quote(flow_id)}"
    )
    last_error: Exception | None = None

    for _ in range(max_attempts):
        try:
            with _urlrequest.urlopen(request_url, timeout=10) as response:
                return _json.loads(response.read().decode())
        except _urlerror.HTTPError as error:
            if error.code == 404:
                last_error = error
                _time.sleep(0.5)
                continue

            body = error.read().decode()[:200]
            raise BrokerError(
                f"broker returned {error.code} on /result: {body}"
            ) from error
        except (_urlerror.URLError, TimeoutError) as error:
            last_error = error
            _time.sleep(0.5)

    raise BrokerError(
        f"broker /result timed out after {max_attempts} tries: {last_error}"
    )