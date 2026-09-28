from __future__ import annotations

import inspect
import json
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...oauth import (
    OAuthFlow,
    PendingState,
    broker_authorize_url,
    broker_url,
    get_oauth_spec,
    make_challenge,
    make_flow_id,
    make_verifier,
)
from ...store import Vault
from ._deps import state

router = APIRouter(tags=["integrations"], prefix="/integrations")

TOKEN_HELP = {
    "github": {"label": "Personal Access Token (classic)", "url": "https://github.com/settings/tokens/new?scopes=repo,read:user,read:org&description=autoMate", "hint": "Click the link → 'Generate token' → copy."},
    "gitlab": {"label": "Personal Access Token", "url": "https://gitlab.com/-/user_settings/personal_access_tokens", "hint": "Scopes: api, read_user."},
    "gitee": {"label": "私人令牌", "url": "https://gitee.com/profile/personal_access_tokens/new", "hint": "勾选 user_info, projects, issues。"},
    "notion": {"label": "Internal Integration Token", "url": "https://www.notion.so/my-integrations", "hint": "New integration → copy 'Internal Integration Token'."},
    "slack": {"label": "Bot User OAuth Token (xoxb-…)", "url": "https://api.slack.com/apps", "hint": "Your app → OAuth & Permissions → Bot User OAuth Token."},
    "linear": {"label": "Personal API Key", "url": "https://linear.app/settings/api", "hint": "Settings → API → Create new key."},
    "jira": {"label": "API token (atlassian.com)", "url": "https://id.atlassian.com/manage-profile/security/api-tokens", "hint": "Create token. Use email:token format if asked."},
    "confluence": {"label": "API token (atlassian.com)", "url": "https://id.atlassian.com/manage-profile/security/api-tokens", "hint": "Same token works for Confluence and Jira."},
    "trello": {"label": "API key + token", "url": "https://trello.com/app-key", "hint": "Get key → click 'Token' link to generate the matching token."},
    "asana": {"label": "Personal Access Token", "url": "https://app.asana.com/0/my-apps", "hint": "+ New Personal Access Token."},
    "monday": {"label": "API Token v2", "url": "https://monday.com/developers/v2", "hint": "Profile → Developer → My access tokens."},
    "hubspot": {"label": "Private App access token", "url": "https://app.hubspot.com/private-apps", "hint": "Create a private app, set scopes, copy access token."},
    "airtable": {"label": "Personal Access Token", "url": "https://airtable.com/create/tokens", "hint": "Create token, choose bases + scopes."},
    "stripe": {"label": "Secret key (sk_live_… / sk_test_…)", "url": "https://dashboard.stripe.com/apikeys", "hint": "Standard or restricted key."},
    "shopify": {"label": "Admin API access token", "url": "https://help.shopify.com/manual/apps/app-types/custom-apps", "hint": "Create custom app per shop, install, copy token."},
    "telegram": {"label": "Bot token", "url": "https://t.me/BotFather", "hint": "Talk to @BotFather → /newbot → copy token."},
    "discord": {"label": "Bot token", "url": "https://discord.com/developers/applications", "hint": "Application → Bot → Reset Token."},
    "teams": {"label": "Microsoft Graph access token", "url": "https://learn.microsoft.com/en-us/graph/auth/auth-concepts", "hint": "Register an app on Entra, grant permissions, mint a token."},
    "zoom": {"label": "Server-to-Server OAuth token", "url": "https://marketplace.zoom.us/develop/create", "hint": "Create a S2S OAuth app → copy the access token."},
    "twitter": {"label": "Bearer Token (v2 API)", "url": "https://developer.twitter.com/en/portal/dashboard", "hint": "Project → Keys & Tokens → Bearer Token."},
    "sendgrid": {"label": "API Key", "url": "https://app.sendgrid.com/settings/api_keys", "hint": "Full or restricted access."},
    "mailchimp": {"label": "API key (with -usX suffix)", "url": "https://us1.admin.mailchimp.com/account/api/", "hint": "Profile → Extras → API keys."},
    "twilio": {"label": "Auth token (account SID + auth token)", "url": "https://console.twilio.com/", "hint": "Console homepage → Account Info."},
    "sentry": {"label": "Auth token", "url": "https://sentry.io/settings/account/api/auth-tokens/", "hint": "Create new token, scopes: project:read, event:read."},
    "wecom": {"label": "应用 corpid + secret", "url": "https://work.weixin.qq.com/wework_admin/frame#apps", "hint": "应用管理 → 自建应用 → 复制 corpid 和 Secret(用 ':' 拼接)。"},
    "weixin": {"label": "公众号 AppID + AppSecret", "url": "https://mp.weixin.qq.com/", "hint": "开发 → 基本配置 → AppID + AppSecret(用 ':' 拼接)。"},
    "weibo": {"label": "Access token", "url": "https://open.weibo.com/apps", "hint": "应用管理 → 高级信息 → OAuth2 Access Token。"},
    "yuque": {"label": "User Token", "url": "https://www.yuque.com/settings/tokens", "hint": "新建 Token → 选择团队/空间 → 复制。"},
    "amap": {"label": "Web 服务 key", "url": "https://console.amap.com/dev/key/app", "hint": "应用管理 → 添加 Key → 服务平台选 Web 服务。"},
    "feishu": {"label": "tenant_access_token (or app_id:app_secret)", "url": "https://open.feishu.cn/app", "hint": "若不想走 OAuth,直接用应用的 app_id:app_secret(冒号拼)。"},
    "dingtalk": {"label": "AppKey + AppSecret", "url": "https://open-dev.dingtalk.com/fe/app", "hint": "AppKey:AppSecret 用冒号拼。"},
}

INTEGRATION_CATALOG = [
    {"id": "github", "display_name": "GitHub", "category": "DevOps", "auth": "oauth"},
    {"id": "gitlab", "display_name": "GitLab", "category": "DevOps", "auth": "apikey"},
    {"id": "notion", "display_name": "Notion", "category": "Productivity", "auth": "oauth"},
    {"id": "slack", "display_name": "Slack", "category": "Messaging", "auth": "oauth"},
    {"id": "linear", "display_name": "Linear", "category": "Project", "auth": "oauth"},
    {"id": "jira", "display_name": "Jira", "category": "Project", "auth": "apikey"},
    {"id": "confluence", "display_name": "Confluence", "category": "Productivity", "auth": "apikey"},
    {"id": "trello", "display_name": "Trello", "category": "Project", "auth": "apikey"},
    {"id": "asana", "display_name": "Asana", "category": "Project", "auth": "apikey"},
    {"id": "monday", "display_name": "Monday.com", "category": "Project", "auth": "apikey"},
    {"id": "hubspot", "display_name": "HubSpot", "category": "CRM", "auth": "apikey"},
    {"id": "airtable", "display_name": "Airtable", "category": "Productivity", "auth": "apikey"},
    {"id": "stripe", "display_name": "Stripe", "category": "Payments", "auth": "apikey"},
    {"id": "shopify", "display_name": "Shopify", "category": "Commerce", "auth": "apikey"},
    {"id": "telegram", "display_name": "Telegram", "category": "Messaging", "auth": "apikey"},
    {"id": "discord", "display_name": "Discord", "category": "Messaging", "auth": "apikey"},
    {"id": "teams", "display_name": "Microsoft Teams", "category": "Messaging", "auth": "apikey"},
    {"id": "zoom", "display_name": "Zoom", "category": "Messaging", "auth": "apikey"},
    {"id": "twitter", "display_name": "X / Twitter", "category": "Social", "auth": "apikey"},
    {"id": "sendgrid", "display_name": "SendGrid", "category": "Email", "auth": "apikey"},
    {"id": "mailchimp", "display_name": "Mailchimp", "category": "Marketing", "auth": "apikey"},
    {"id": "twilio", "display_name": "Twilio", "category": "Communications", "auth": "apikey"},
    {"id": "sentry", "display_name": "Sentry", "category": "DevOps", "auth": "apikey"},
    {"id": "gitee", "display_name": "Gitee", "category": "DevOps", "auth": "apikey"},
    {"id": "wecom", "display_name": "WeCom", "category": "Messaging", "auth": "apikey"},
    {"id": "weixin", "display_name": "WeChat Official Account", "category": "Messaging", "auth": "apikey"},
    {"id": "weibo", "display_name": "Weibo", "category": "Social", "auth": "apikey"},
    {"id": "yuque", "display_name": "Yuque", "category": "Productivity", "auth": "apikey"},
    {"id": "amap", "display_name": "Amap", "category": "Maps", "auth": "apikey"},
    {"id": "feishu", "display_name": "Feishu", "category": "Messaging", "auth": "apikey"},
    {"id": "dingtalk", "display_name": "DingTalk", "category": "Messaging", "auth": "apikey"},
]


class ConnectRequest(BaseModel):
    token: str = Field(..., min_length=1)


class ApiKeyRequest(ConnectRequest):
    pass


class OAuthStartRequest(BaseModel):
    pass


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _known(cid: str) -> bool:
    return any(item["id"] == cid for item in INTEGRATION_CATALOG)


def _vault_for(s: Any) -> Any:
    for name in ("vault", "secrets", "store"):
        candidate = getattr(s, name, None)
        if candidate is not None:
            return candidate
    for args in ((s,), (getattr(s, "db", None),), ()):
        try:
            if args and args[0] is None:
                continue
            return Vault(*args)
        except TypeError:
            continue
    return Vault(s)


def _call(store: Any, names: tuple[str, ...], *args: Any) -> Any:
    for name in names:
        method = getattr(store, name, None)
        if method is None:
            continue
        try:
            return method(*args)
        except TypeError:
            continue
    raise AttributeError("vault does not support the requested operation")


def _save_token(s: Any, cid: str, token: str) -> None:
    vault = _vault_for(s)
    payload = {"token": token}
    attempts = (
        ("set_integration", (cid, token)),
        ("set", (cid, token)),
        ("put", (cid, token)),
        ("save", (cid, token)),
        ("set", (f"integrations:{cid}", token)),
        ("put", (f"integrations:{cid}", token)),
        ("set", (cid, payload)),
        ("put", (cid, payload)),
    )
    for method_name, args in attempts:
        method = getattr(vault, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
            return
        except TypeError:
            continue
    raise RuntimeError("Unable to store integration credentials")


def _remove_token(s: Any, cid: str) -> None:
    vault = _vault_for(s)
    for method_name, args in (
        ("remove_integration", (cid,)),
        ("delete", (cid,)),
        ("remove", (cid,)),
        ("pop", (cid,)),
        ("delete", (f"integrations:{cid}",)),
        ("remove", (f"integrations:{cid}",)),
    ):
        method = getattr(vault, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
            return
        except TypeError:
            continue


def _callback_url(request: Request, cid: str) -> str:
    return f"{str(request.base_url).rstrip('/')}/oauth/{cid}/callback"


def _pending(cid: str, verifier: str | None = None) -> Any:
    for kwargs in (
        {"provider_id": cid, "verifier": verifier},
        {"provider_id": cid, "code_verifier": verifier},
        {"provider_id": cid},
    ):
        try:
            return PendingState(**{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            continue
    return PendingState(cid, verifier)


def _remember(key: str, pending: Any) -> None:
    for method_name in ("put", "set", "add"):
        method = getattr(OAuthFlow, method_name, None)
        if method is not None:
            method(key, pending)
            return
    try:
        OAuthFlow[key] = pending
    except Exception as exc:
        raise RuntimeError("OAuth state store is unavailable") from exc


def _broker_authorization_url(cid: str, callback_url: str, flow_id: str) -> str:
    candidates = (
        {"provider_id": cid, "callback_url": callback_url, "flow_id": flow_id},
        {"provider": cid, "callback_url": callback_url, "flow_id": flow_id},
        {"cid": cid, "callback_url": callback_url, "flow_id": flow_id},
    )
    for kwargs in candidates:
        try:
            return broker_authorize_url(**kwargs)
        except TypeError:
            continue
    base = broker_url()
    return f"{str(base).rstrip('/')}/authorize?{urlencode({'provider': cid, 'callback_url': callback_url, 'flow_id': flow_id})}"


def _pkce_authorization_url(spec: Any, callback_url: str, state_id: str, verifier: str) -> str:
    authorize_url = _value(spec, "authorize_url") or _value(spec, "authorization_url")
    client_id = _value(spec, "client_id")
    scopes = _value(spec, "scope") or _value(spec, "scopes") or ""
    if isinstance(scopes, (list, tuple, set)):
        scopes = " ".join(scopes)
    redirect_uri = _value(spec, "redirect_uri") or callback_url
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state_id,
        "code_challenge": make_challenge(verifier),
        "code_challenge_method": "S256",
    }
    separator = "&" if "?" in authorize_url else "?"
    return f"{authorize_url}{separator}{urlencode({k: v for k, v in params.items() if v is not None})}"


@router.get("")
@router.get("/")
def integrations():
    items = []
    for entry in INTEGRATION_CATALOG:
        item = dict(entry)
        help_info = TOKEN_HELP.get(item["id"])
        if help_info is not None:
            item["token_help"] = help_info
        items.append(item)
    return {"integrations": items}


@router.get("/catalog")
def catalog():
    return integrations()


@router.post("/{cid}/connect")
@router.post("/{cid}/apikey")
def connect(cid: str, body: ConnectRequest, s=Depends(state)):
    if not _known(cid):
        raise HTTPException(status_code=404, detail=f"Unknown integration: {cid}")
    _save_token(s, cid, body.token.strip())
    return {"ok": True, "id": cid}


@router.delete("/{cid}")
def disconnect(cid: str, s=Depends(state)):
    if not _known(cid):
        raise HTTPException(status_code=404, detail=f"Unknown integration: {cid}")
    _remove_token(s, cid)
    return {"ok": True, "id": cid}


@router.post("/{cid}/oauth")
@router.post("/{cid}/oauth/start")
def start_oauth(cid: str, request: Request, s=Depends(state)):
    spec = get_oauth_spec(cid)
    if not spec:
        raise HTTPException(status_code=404, detail=f"No OAuth configuration for '{cid}'.")

    flow = _value(spec, "flow", "pkce")
    callback_url = _callback_url(request, cid)
    flow_id = make_flow_id()

    if str(flow).lower() == "broker":
        _remember(f"flow:{flow_id}", _pending(cid))
        return {
            "url": _broker_authorization_url(cid, callback_url, flow_id),
            "flow_id": flow_id,
        }

    verifier = make_verifier()
    _remember(flow_id, _pending(cid, verifier))
    return {
        "url": _pkce_authorization_url(spec, callback_url, flow_id, verifier),
        "state": flow_id,
    }