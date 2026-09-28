from __future__ import annotations

from .feishu import FeishuIntegration
from .dingtalk import DingTalkIntegration
from .wecom import WeComIntegration
from .weixin import WeixinIntegration
from .weibo import WeiboIntegration
from .gitee import GiteeIntegration
from .yuque import YuqueIntegration
from .amap import AmapIntegration
from .slack import SlackIntegration
from .telegram import TelegramIntegration
from .discord import DiscordIntegration
from .teams import TeamsIntegration
from .zoom import ZoomIntegration
from .github_api import GitHubIntegration
from .gitlab import GitLabIntegration
from .sentry import SentryIntegration
from .notion import NotionIntegration
from .airtable import AirtableIntegration
from .linear import LinearIntegration
from .jira import JiraIntegration
from .trello import TrelloIntegration
from .hubspot import HubSpotIntegration
from .twitter import TwitterIntegration
from .sendgrid import SendGridIntegration
from .twilio import TwilioIntegration
from .mailchimp import MailchimpIntegration
from .stripe import StripeIntegration
from .shopify import ShopifyIntegration
from .confluence import ConfluenceIntegration
from .asana import AsanaIntegration
from .monday import MondayIntegration

ALL_INTEGRATIONS = [
    FeishuIntegration(),
    DingTalkIntegration(),
    WeComIntegration(),
    WeixinIntegration(),
    WeiboIntegration(),
    GiteeIntegration(),
    YuqueIntegration(),
    AmapIntegration(),
    SlackIntegration(),
    TelegramIntegration(),
    DiscordIntegration(),
    TeamsIntegration(),
    ZoomIntegration(),
    GitHubIntegration(),
    GitLabIntegration(),
    SentryIntegration(),
    NotionIntegration(),
    AirtableIntegration(),
    LinearIntegration(),
    JiraIntegration(),
    ConfluenceIntegration(),
    TrelloIntegration(),
    AsanaIntegration(),
    MondayIntegration(),
    HubSpotIntegration(),
    StripeIntegration(),
    ShopifyIntegration(),
    TwitterIntegration(),
    SendGridIntegration(),
    TwilioIntegration(),
    MailchimpIntegration(),
]


def register_all(mcp) -> list[str]:
    registered = []
    for integration in ALL_INTEGRATIONS:
        if integration.is_configured():
            integration.register(mcp)
            registered.append(integration.label)
    return registered


def get_configured_summary() -> str:
    configured = [integration for integration in ALL_INTEGRATIONS if integration.is_configured()]
    unconfigured = [
        integration for integration in ALL_INTEGRATIONS if not integration.is_configured()
    ]

    lines = []
    if configured:
        lines.append(
            f"Active ({len(configured)}): "
            + ", ".join(integration.label for integration in configured)
        )
    if unconfigured:
        lines.append(
            f"Inactive ({len(unconfigured)}): "
            + ", ".join(integration.label for integration in unconfigured)
        )

    return "\n".join(lines) if lines else "No integrations configured."