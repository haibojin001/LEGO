from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .catalog import OAuthSpec


@dataclass
class PendingState:
    provider_id: str
    redirect_uri: str
    flow: str
    pkce_verifier: str | None = None
    broker_flow_id: str | None = None
    created_at: float = 0.0


class OAuthFlow:
    PENDING: dict[str, PendingState] = {}

    @classmethod
    def remember(cls, state: str, ps: PendingState) -> None:
        cls._gc()
        cls.PENDING[state] = ps

    @classmethod
    def pop(cls, state: str) -> PendingState | None:
        cls._gc()
        return cls.PENDING.pop(state, None)

    @classmethod
    def _gc(cls) -> None:
        timestamp = time.time()
        expired_states = [
            state
            for state, pending_state in cls.PENDING.items()
            if timestamp - pending_state.created_at > 600
        ]
        for state in expired_states:
            cls.PENDING.pop(state, None)

    @staticmethod
    def authorize_url_pkce(
        spec: OAuthSpec,
        *,
        client_id: str,
        redirect_uri: str,
        state: str,
        code_challenge: str,
        scopes: tuple[str, ...] | None = None,
    ) -> str:
        values = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        requested_scopes = spec.scopes if scopes is None else scopes
        if requested_scopes:
            values["scope"] = " ".join(requested_scopes)
        values.update(spec.extra_auth_params)
        return f"{spec.authorize_url}?{urllib.parse.urlencode(values)}"


def exchange_code_pkce(
    spec: OAuthSpec,
    *,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str,
) -> dict:
    encoded_data = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        }
    ).encode()

    token_request = urllib.request.Request(
        spec.token_url,
        data=encoded_data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(token_request, timeout=30) as token_response:
        response_text = token_response.read().decode()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return dict(urllib.parse.parse_qsl(response_text))