from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from ...oauth import OAuthFlow, broker_fetch_result, exchange_code_pkce, get_oauth_spec
from ._deps import state as app_state

router = APIRouter(tags=["oauth"])

_RESULT_HTML = """\
<!doctype html>
<html><head><meta charset=utf-8><title>autoMate · {title}</title>
<style>
  body{{font-family:-apple-system,system-ui,sans-serif;max-width:520px;
        margin:80px auto;color:#111;padding:0 20px;}}
  h1{{font-size:22px;margin:0 0 12px;}} p{{color:#555;line-height:1.55;}}
  .pill{{display:inline-block;padding:4px 10px;border-radius:999px;
        font-size:12px;background:{bg};color:{fg};margin-bottom:16px;}}
</style></head>
<body><span class=pill>{pill}</span><h1>{title}</h1><p>{body}</p>
<p style="color:#888;font-size:12px;">This window will close automatically.</p>
<script>
  try {{
    if (window.opener && !window.opener.closed) {{
      window.opener.postMessage(
        {{kind: "automate-oauth", ok: {ok_js}, provider: "{provider}", message: {msg_js} }},
        window.location.origin
      );
    }}
  }} catch (e) {{ }}
  setTimeout(() => {{ try {{ window.close(); }} catch (e) {{}} }}, 1500);
</script>
</body></html>"""


def _page(
    *,
    title: str,
    body: str,
    ok: bool,
    provider: str,
    code: int = 200,
) -> HTMLResponse:
    if ok:
        pill = "Success"
        background = "#d1fae5"
        foreground = "#065f46"
    else:
        pill = "Error"
        background = "#fde8e8"
        foreground = "#9b1c1c"

    content = _RESULT_HTML.format(
        title=title,
        body=body,
        pill=pill,
        bg=background,
        fg=foreground,
        ok_js="true" if ok else "false",
        provider=provider,
        msg_js=json.dumps(body),
    )
    return HTMLResponse(content, status_code=code)


def _err(provider: str, title: str, body: str, code: int = 400) -> HTMLResponse:
    return _page(
        title=title,
        body=body,
        ok=False,
        provider=provider,
        code=code,
    )


@router.get("/oauth/{cid}/callback", response_class=HTMLResponse)
def callback(
    cid: str,
    code: str | None = Query(None),
    state: str | None = Query(None),
    flow_id: str | None = Query(None),
    error: str | None = Query(None),
    s=Depends(app_state),
):
    spec = get_oauth_spec(cid)

    if error:
        return _err(cid, "Authorization failed", f"Provider returned: {error}")

    if spec is None:
        return _err(
            cid,
            "Unknown provider",
            f"No OAuth spec registered for '{cid}'.",
        )

    if state:
        pending = OAuthFlow.pop(state)
    elif flow_id:
        pending = OAuthFlow.pop(f"flow:{flow_id}")
    else:
        return _err(
            cid,
            "Missing state",
            "No state or flow_id in callback.",
        )

    if pending is None or pending.provider_id != cid:
        return _err(
            cid,
            "State mismatch",
            "The OAuth state token did not match an in-flight request. "
            "Try clicking Connect again — these tokens expire after 10 minutes.",
        )

    if pending.flow == "pkce":
        if not code:
            return _err(cid, "Missing code", "Provider didn't return an auth code.")

        if not pending.pkce_verifier or not spec.pkce_client_id:
            return _err(
                cid,
                "PKCE state lost",
                "The verifier or client_id wasn't available at exchange time.",
            )

        try:
            result = exchange_code_pkce(
                spec,
                code=code,
                redirect_uri=pending.redirect_uri,
                client_id=spec.pkce_client_id,
                code_verifier=pending.pkce_verifier,
            )
        except Exception as exc:
            return _err(
                cid,
                "Token exchange failed",
                f"{type(exc).__name__}: {exc}",
            )

        return _store_payload(cid, spec, result, s)

    if pending.flow == "broker":
        if not pending.broker_flow_id:
            return _err(
                cid,
                "Broker flow_id lost",
                "The pending entry had no broker_flow_id.",
            )

        try:
            result = broker_fetch_result(flow_id=pending.broker_flow_id)
        except Exception as exc:
            return _err(
                cid,
                "Broker fetch failed",
                f"{type(exc).__name__}: {exc}. The broker may be down — "
                "try the API-key path for this provider as a fallback.",
            )

        return _store_payload(cid, spec, result, s)

    return _err(cid, "Unknown flow", f"flow={pending.flow}")


def _store_payload(cid: str, spec, payload: dict, s) -> HTMLResponse:
    access_token = payload.get(spec.token_field) or payload.get("access_token")
    refresh_token = payload.get(spec.refresh_field or "")
    expires_in = payload.get("expires_in")
    expires_at = time.time() + float(expires_in) if expires_in else None

    if not access_token:
        return _err(
            cid,
            "No access_token in response",
            f"Raw response keys: {list(payload.keys())}",
        )

    excluded = {spec.token_field, spec.refresh_field or ""}
    raw_metadata = {
        key: value
        for key, value in payload.items()
        if key not in excluded
    }

    s.db.upsert_connection(
        id=cid,
        display_name=spec.display_name,
        auth_kind="oauth",
        status="connected",
        token=access_token,
        refresh=refresh_token,
        expires_at=expires_at,
        metadata={"raw": raw_metadata},
    )

    return _page(
        title=f"{spec.display_name} connected",
        body="autoMate now has authorization for this account. You may close this tab.",
        ok=True,
        provider=cid,
    )