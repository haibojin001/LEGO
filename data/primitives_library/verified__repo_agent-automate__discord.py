"""Discord Bot integration."""
from __future__ import annotations

from .base import BaseIntegration

API = "https://discord.com/api/v10"


class DiscordIntegration(BaseIntegration):
    name = "discord"
    label = "Discord"
    env_vars = {
        "DISCORD_BOT_TOKEN": "Bot token from discord.com/developers/applications",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bot {self.env('DISCORD_BOT_TOKEN')}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def discord_send_message(channel_id: str, content: str) -> str:
            """
            Send a message to a Discord channel.

            Args:
                channel_id: Discord channel ID (enable Developer Mode to copy).
                content: Message content (supports Discord markdown).
            """
            response = integration.post(
                f"{API}/channels/{channel_id}/messages",
                {"content": content},
                integration._auth(),
            )
            return integration.ok(
                {
                    "id": response.get("id"),
                    "channel_id": response.get("channel_id"),
                }
            )

        @mcp.tool()
        def discord_get_messages(channel_id: str, limit: int = 20) -> str:
            """
            Get recent messages from a Discord channel.

            Args:
                channel_id: Discord channel ID.
                limit: Number of messages to retrieve (max 100, default 20).
            """
            import urllib.parse

            query = urllib.parse.urlencode({"limit": min(limit, 100)})
            response = integration.get(
                f"{API}/channels/{channel_id}/messages?{query}",
                integration._auth(),
            )
            messages = response if isinstance(response, list) else []
            output = [f"{len(messages)} message(s) from channel {channel_id}:"]
            for message in messages:
                author = message.get("author", {}).get("username", "?")
                output.append(f"  [{author}] {message.get('content', '')[:120]}")
            return "\n".join(output)

        @mcp.tool()
        def discord_get_guild_channels(guild_id: str) -> str:
            """
            List all channels in a Discord server (guild).

            Args:
                guild_id: Discord server (guild) ID.
            """
            response = integration.get(
                f"{API}/guilds/{guild_id}/channels",
                integration._auth(),
            )
            channels = response if isinstance(response, list) else []
            output = [f"Found {len(channels)} channel(s):"]
            for channel in channels:
                output.append(
                    f"  #{channel.get('name')} ({channel.get('type')}) — {channel.get('id')}"
                )
            return "\n".join(output)

        @mcp.tool()
        def discord_create_dm(user_id: str, content: str) -> str:
            """
            Send a direct message to a Discord user.

            Args:
                user_id: Target user's Discord ID.
                content: Message content.
            """
            dm_channel = integration.post(
                f"{API}/users/@me/channels",
                {"recipient_id": user_id},
                integration._auth(),
            )
            channel_id = dm_channel.get("id")
            if not channel_id:
                return integration.ok({"error": "Failed to create DM channel"})
            response = integration.post(
                f"{API}/channels/{channel_id}/messages",
                {"content": content},
                integration._auth(),
            )
            return integration.ok({"id": response.get("id"), "channel_id": channel_id})