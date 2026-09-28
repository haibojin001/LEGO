from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse

from .base import BaseIntegration


class DingTalkIntegration(BaseIntegration):
    name = "dingtalk"
    label = "钉钉 (DingTalk)"
    env_vars = {
        "DINGTALK_WEBHOOK": "Group robot webhook URL (from DingTalk group settings)",
        "DINGTALK_SECRET": "Robot signing secret (optional but recommended)",
    }

    def _signed_url(self) -> str:
        webhook_url = self.env("DINGTALK_WEBHOOK")
        secret = self.env("DINGTALK_SECRET")

        if not secret:
            return webhook_url

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{secret}"
        signature_bytes = hmac.new(
            secret.encode(),
            string_to_sign.encode(),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(signature_bytes).decode()

        return (
            f"{webhook_url}&timestamp={timestamp}"
            f"&sign={urllib.parse.quote_plus(signature)}"
        )

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def dingtalk_send_text(content: str, at_all: bool = False) -> str:
            """
            Send a text message to a 钉钉 group via webhook robot.

            Args:
                content: Message text.
                at_all: Whether to @all members.
            """
            result = integration.post(
                integration._signed_url(),
                {
                    "msgtype": "text",
                    "text": {"content": content},
                    "at": {"isAtAll": at_all},
                },
            )
            return integration.ok(result)

        @mcp.tool()
        def dingtalk_send_markdown(title: str, text: str) -> str:
            """
            Send a Markdown message to a 钉钉 group via webhook robot.

            Args:
                title: Card title (shown in notification).
                text: Markdown body content.
            """
            result = integration.post(
                integration._signed_url(),
                {
                    "msgtype": "markdown",
                    "markdown": {"title": title, "text": text},
                },
            )
            return integration.ok(result)

        @mcp.tool()
        def dingtalk_send_link(
            title: str,
            text: str,
            url: str,
            pic_url: str = "",
        ) -> str:
            """
            Send a link card to a 钉钉 group.

            Args:
                title: Link title.
                text: Link description.
                url: Target URL.
                pic_url: Optional thumbnail image URL.
            """
            result = integration.post(
                integration._signed_url(),
                {
                    "msgtype": "link",
                    "link": {
                        "title": title,
                        "text": text,
                        "messageUrl": url,
                        "picUrl": pic_url,
                    },
                },
            )
            return integration.ok(result)