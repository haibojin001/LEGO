"""飞书 (Feishu / Lark) integration."""
from __future__ import annotations

import os

from .base import BaseIntegration

API = "https://open.feishu.cn/open-apis"


class FeishuIntegration(BaseIntegration):
    name = "feishu"
    label = "飞书 (Feishu)"
    env_vars = {
        "FEISHU_APP_ID": "App ID from open.feishu.cn",
        "FEISHU_APP_SECRET": "App Secret from open.feishu.cn",
    }

    def _token(self) -> str:
        result = self.post(
            f"{API}/auth/v3/tenant_access_token/internal",
            {
                "app_id": self.env("FEISHU_APP_ID"),
                "app_secret": self.env("FEISHU_APP_SECRET"),
            },
        )
        return result.get("tenant_access_token", "")

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def feishu_send_message(
            receive_id: str, content: str, receive_type: str = "chat_id"
        ) -> str:
            """
            Send a message via 飞书 (Feishu).

            Args:
                receive_id: chat_id or user_id of the recipient.
                content: Message text content.
                receive_type: "chat_id" (default) or "user_id".
            """
            result = integration.post(
                f"{API}/im/v1/messages?receive_id_type={receive_type}",
                {
                    "receive_id": receive_id,
                    "msg_type": "text",
                    "content": f'{{"text":"{content}"}}',
                },
                integration._auth(),
            )
            return integration.ok(result)

        @mcp.tool()
        def feishu_create_doc(title: str, content: str = "") -> str:
            """
            Create a 飞书 document.

            Args:
                title: Document title.
                content: Optional plain text content.
            """
            result = integration.post(
                f"{API}/docx/v1/documents",
                {"title": title},
                integration._auth(),
            )
            return integration.ok(result)

        @mcp.tool()
        def feishu_list_chats() -> str:
            """List all 飞书 group chats the bot is in."""
            result = integration.get(
                f"{API}/im/v1/chats",
                integration._auth(),
            )
            return integration.ok(result)

        @mcp.tool()
        def feishu_create_task(
            title: str, due_date: str = "", notes: str = ""
        ) -> str:
            """
            Create a task in 飞书 Tasks.

            Args:
                title: Task title.
                due_date: Due date in RFC3339 format, e.g. "2025-12-31T00:00:00+08:00".
                notes: Optional notes.
            """
            payload: dict = {"summary": title}
            if due_date:
                payload["due"] = {"time": due_date}
            if notes:
                payload["description"] = notes
            result = integration.post(
                f"{API}/task/v1/tasks",
                payload,
                integration._auth(),
            )
            return integration.ok(result)