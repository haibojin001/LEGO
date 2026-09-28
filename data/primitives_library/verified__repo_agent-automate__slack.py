"""Slack integration."""
from __future__ import annotations

from .base import BaseIntegration

API = "https://slack.com/api"


class SlackIntegration(BaseIntegration):
    name = "slack"
    label = "Slack"
    env_vars = {
        "SLACK_BOT_TOKEN": "Bot User OAuth Token (xoxb-...) from api.slack.com/apps",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.env('SLACK_BOT_TOKEN')}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def slack_send_message(channel: str, text: str) -> str:
            """
            Send a message to a Slack channel.

            Args:
                channel: Channel ID or name (e.g. "#general" or "C01234ABC").
                text: Message text (supports mrkdwn formatting).
            """
            response = integration.post(
                f"{API}/chat.postMessage",
                {"channel": channel, "text": text},
                integration._auth(),
            )
            if not response.get("ok"):
                return f"Error: {response.get('error')}"
            return f"Sent to {response['channel']} at ts={response['message']['ts']}"

        @mcp.tool()
        def slack_list_channels(limit: int = 50) -> str:
            """
            List public Slack channels.

            Args:
                limit: Maximum number of channels to return (default 50).
            """
            response = integration.get(
                f"{API}/conversations.list?limit={limit}&exclude_archived=true",
                integration._auth(),
            )
            channels = response.get("channels", [])
            lines = [f"Found {len(channels)} channel(s):"]
            for channel in channels:
                lines.append(
                    f"  #{channel['name']} — {channel['id']} "
                    f"({channel.get('num_members', 0)} members)"
                )
            return "\n".join(lines)

        @mcp.tool()
        def slack_get_messages(channel: str, limit: int = 20) -> str:
            """
            Get recent messages from a Slack channel.

            Args:
                channel: Channel ID.
                limit: Number of messages to retrieve (default 20).
            """
            response = integration.get(
                f"{API}/conversations.history?channel={channel}&limit={limit}",
                integration._auth(),
            )
            messages = response.get("messages", [])
            lines = [f"{len(messages)} message(s) from {channel}:"]
            for message in messages:
                sender = message.get("user", message.get("bot_id", "?"))
                lines.append(f"  [{sender}] {message.get('text', '')[:120]}")
            return "\n".join(lines)

        @mcp.tool()
        def slack_reply_thread(channel: str, thread_ts: str, text: str) -> str:
            """
            Reply to a Slack thread.

            Args:
                channel: Channel ID.
                thread_ts: Timestamp of the parent message (from slack_get_messages).
                text: Reply text.
            """
            response = integration.post(
                f"{API}/chat.postMessage",
                {"channel": channel, "thread_ts": thread_ts, "text": text},
                integration._auth(),
            )
            return integration.ok(
                {"ok": response.get("ok"), "error": response.get("error")}
            )