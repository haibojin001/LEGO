from __future__ import annotations

import inspect
import os
from typing import Any, Callable, get_type_hints

from ..store import get_db
from .registry import Tool, ToolRegistry


CONNECTION_ENV_MAP: dict[str, list[str]] = {
    "github": ["GITHUB_TOKEN"],
    "gitlab": ["GITLAB_TOKEN"],
    "gitee": ["GITEE_TOKEN"],
    "notion": ["NOTION_API_KEY"],
    "slack": ["SLACK_BOT_TOKEN"],
    "telegram": ["TELEGRAM_BOT_TOKEN"],
    "discord": ["DISCORD_BOT_TOKEN"],
    "linear": ["LINEAR_API_KEY"],
    "jira": ["JIRA_API_TOKEN"],
    "confluence": ["CONFLUENCE_API_TOKEN"],
    "trello": ["TRELLO_API_KEY"],
    "asana": ["ASANA_PAT"],
    "monday": ["MONDAY_API_TOKEN"],
    "hubspot": ["HUBSPOT_TOKEN"],
    "stripe": ["STRIPE_API_KEY"],
    "shopify": ["SHOPIFY_TOKEN"],
    "sendgrid": ["SENDGRID_API_KEY"],
    "twilio": ["TWILIO_AUTH_TOKEN"],
    "mailchimp": ["MAILCHIMP_API_KEY"],
    "twitter": ["TWITTER_BEARER_TOKEN"],
    "sentry": ["SENTRY_AUTH_TOKEN"],
    "airtable": ["AIRTABLE_API_KEY"],
    "feishu": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    "dingtalk": ["DINGTALK_WEBHOOK"],
    "wecom": ["WECOM_WEBHOOK"],
    "weixin": ["WEIXIN_APP_ID", "WEIXIN_APP_SECRET"],
    "weibo": ["WEIBO_ACCESS_TOKEN"],
    "yuque": ["YUQUE_TOKEN"],
    "amap": ["AMAP_KEY"],
    "zoom": ["ZOOM_ACCOUNT_ID", "ZOOM_CLIENT_ID", "ZOOM_CLIENT_SECRET"],
    "teams": ["TEAMS_WEBHOOK"],
}


def _hydrate_env_from_connections() -> set[str]:
    database = get_db()
    available: set[str] = set()

    for connection in database.list_connections():
        if connection["status"] != "connected":
            continue

        stored = database.get_connection(connection["id"], decrypt=True)
        if not stored:
            continue

        credential = stored.get("token")
        if not credential:
            continue

        names = CONNECTION_ENV_MAP.get(connection["id"], [])
        if "=" in credential and "\n" in credential:
            for entry in credential.splitlines():
                if "=" not in entry:
                    continue
                key, value = entry.split("=", 1)
                os.environ[key.strip()] = value.strip()
        elif names:
            os.environ[names[0]] = credential

        available.add(connection["id"])

    return available


def _derive_schema(fn: Callable) -> dict:
    try:
        annotations = get_type_hints(fn)
    except Exception:
        annotations = {}

    signature = inspect.signature(fn)
    fields: dict[str, dict] = {}
    mandatory: list[str] = []
    json_types = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        annotation = annotations.get(name, str)
        field: dict[str, Any] = {"type": json_types.get(annotation, "string")}

        if parameter.default is inspect.Parameter.empty:
            mandatory.append(name)
        else:
            field["default"] = parameter.default

        fields[name] = field

    result: dict[str, Any] = {
        "type": "object",
        "properties": fields,
    }
    if mandatory:
        result["required"] = mandatory
    return result


class _MCPShim:
    def __init__(
        self,
        registry: ToolRegistry,
        connection_id: str,
        category: str,
    ):
        self.registry = registry
        self.connection_id = connection_id
        self.category = category

    def tool(self, *_a, **_k):
        def decorate(fn: Callable) -> Callable:
            documentation = inspect.getdoc(fn) or ""
            description = documentation.split("\n\nArgs:")[0].strip() or fn.__name__

            try:
                self.registry.register(
                    Tool(
                        name=fn.__name__,
                        description=description,
                        parameters=_derive_schema(fn),
                        handler=fn,
                        category=self.category,
                        requires=[self.connection_id],
                        danger="low",
                    )
                )
            except ValueError:
                pass

            return fn

        return decorate


def register(reg: ToolRegistry) -> None:
    loaded = _hydrate_env_from_connections()

    try:
        from ..integrations import ALL_INTEGRATIONS
    except ImportError:
        return

    for integration in ALL_INTEGRATIONS:
        if integration.name not in loaded and not integration.is_configured():
            continue

        category = f"integration:{integration.name}"
        try:
            integration.register(_MCPShim(reg, integration.name, category))
        except Exception:
            continue