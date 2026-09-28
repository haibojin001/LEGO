"""Zoom meeting integration."""
from __future__ import annotations

from .base import BaseIntegration

API = "https://api.zoom.us/v2"


class ZoomIntegration(BaseIntegration):
    name = "zoom"
    label = "Zoom"
    env_vars = {
        "ZOOM_ACCOUNT_ID": "Account ID from marketplace.zoom.us app credentials",
        "ZOOM_CLIENT_ID": "Client ID from marketplace.zoom.us app credentials",
        "ZOOM_CLIENT_SECRET": "Client secret from marketplace.zoom.us app credentials",
    }

    def _get_token(self) -> str:
        import base64
        import json
        import urllib.parse
        import urllib.request

        client_id = self.env("ZOOM_CLIENT_ID")
        client_secret = self.env("ZOOM_CLIENT_SECRET")
        encoded_credentials = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()

        body = urllib.parse.urlencode(
            {
                "grant_type": "account_credentials",
                "account_id": self.env("ZOOM_ACCOUNT_ID"),
            }
        ).encode()

        request = urllib.request.Request(
            "https://zoom.us/oauth/token",
            data=body,
            headers={
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())

        return payload.get("access_token", "")

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def zoom_create_meeting(
            topic: str,
            duration: int = 60,
            start_time: str = "",
        ) -> str:
            """
            Create a Zoom meeting.

            Args:
                topic: Meeting topic/title.
                duration: Meeting duration in minutes (default 60).
                start_time: Meeting start time in ISO 8601 UTC (e.g. "2025-12-01T09:00:00Z"). Empty = instant meeting.
            """
            token = integration._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            data: dict = {
                "topic": topic,
                "type": 1 if not start_time else 2,
                "duration": duration,
            }

            if start_time:
                data["start_time"] = start_time
                data["timezone"] = "UTC"

            result = integration.post(
                f"{API}/users/me/meetings",
                data,
                headers,
            )

            return integration.ok(
                {
                    "id": result.get("id"),
                    "topic": result.get("topic"),
                    "join_url": result.get("join_url"),
                    "start_time": result.get("start_time"),
                    "duration": result.get("duration"),
                }
            )

        @mcp.tool()
        def zoom_list_meetings(limit: int = 10) -> str:
            """
            List upcoming scheduled Zoom meetings.

            Args:
                limit: Max meetings to return (default 10).
            """
            import urllib.parse

            token = integration._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            params = urllib.parse.urlencode(
                {"type": "scheduled", "page_size": limit}
            )
            result = integration.get(
                f"{API}/users/me/meetings?{params}",
                headers,
            )
            meetings = result.get("meetings", [])
            lines = [f"Found {len(meetings)} meeting(s):"]

            for meeting in meetings:
                lines.append(
                    f"  [{meeting['id']}] {meeting['topic']} — "
                    f"{meeting.get('start_time', 'instant')} "
                    f"({meeting.get('duration')} min)"
                )

            return "\n".join(lines)

        @mcp.tool()
        def zoom_get_meeting(meeting_id: str) -> str:
            """
            Get Zoom meeting details.

            Args:
                meeting_id: Zoom meeting ID.
            """
            token = integration._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            result = integration.get(
                f"{API}/meetings/{meeting_id}",
                headers,
            )

            return integration.ok(
                {
                    "id": result.get("id"),
                    "topic": result.get("topic"),
                    "status": result.get("status"),
                    "start_time": result.get("start_time"),
                    "duration": result.get("duration"),
                    "join_url": result.get("join_url"),
                    "password": result.get("password"),
                }
            )