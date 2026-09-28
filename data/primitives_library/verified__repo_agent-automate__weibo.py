"""微博 (Weibo) integration."""

from __future__ import annotations

from .base import BaseIntegration

API = "https://api.weibo.com/2"


class WeiboIntegration(BaseIntegration):
    name = "weibo"
    label = "微博 (Weibo)"
    env_vars = {
        "WEIBO_ACCESS_TOKEN": "OAuth2 access token from open.weibo.com",
    }

    def _auth(self) -> dict:
        return {
            "Authorization": f"OAuth2 {self.env('WEIBO_ACCESS_TOKEN')}",
        }

    def register(self, mcp) -> None:
        owner = self

        @mcp.tool()
        def weibo_post(text: str) -> str:
            """
            Post a new Weibo (微博).

            Args:
                text: Weibo content (max 140 Chinese characters).
            """
            import json
            import urllib.parse
            import urllib.request

            access_token = owner.env("WEIBO_ACCESS_TOKEN")
            payload = urllib.parse.urlencode(
                {"status": text, "access_token": access_token}
            ).encode()
            request = urllib.request.Request(
                f"{API}/statuses/update.json",
                data=payload,
            )

            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode())
            except Exception as exc:
                return owner.ok({"error": str(exc)})

        @mcp.tool()
        def weibo_get_timeline(count: int = 20) -> str:
            """
            Get your Weibo home timeline.

            Args:
                count: Number of posts to retrieve (max 100).
            """
            access_token = owner.env("WEIBO_ACCESS_TOKEN")
            result = owner.get(
                f"{API}/statuses/home_timeline.json"
                f"?access_token={access_token}&count={count}"
            )
            return owner.ok(result)

        @mcp.tool()
        def weibo_get_my_info() -> str:
            """Get current user's Weibo profile."""
            access_token = owner.env("WEIBO_ACCESS_TOKEN")
            account = owner.get(
                f"{API}/account/get_uid.json?access_token={access_token}"
            )
            user_id = account.get("uid", "")

            if user_id:
                profile = owner.get(
                    f"{API}/users/show.json?access_token={access_token}&uid={user_id}"
                )
                return owner.ok(profile)

            return owner.ok(account)