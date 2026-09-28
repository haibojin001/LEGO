"""Twilio SMS/Voice integration."""
from __future__ import annotations

import base64

from .base import BaseIntegration


class TwilioIntegration(BaseIntegration):
    name = "twilio"
    label = "Twilio"
    env_vars = {
        "TWILIO_ACCOUNT_SID": "Account SID from console.twilio.com",
        "TWILIO_AUTH_TOKEN": "Auth token from console.twilio.com",
        "TWILIO_FROM_NUMBER": "Your Twilio phone number (e.g. +1xxxxxxxxxx)",
    }

    def _auth(self) -> dict:
        sid = self.env("TWILIO_ACCOUNT_SID")
        token = self.env("TWILIO_AUTH_TOKEN")
        encoded = base64.b64encode(f"{sid}:{token}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def _api(self, path: str) -> str:
        sid = self.env("TWILIO_ACCOUNT_SID")
        return f"https://api.twilio.com/2010-04-01/Accounts/{sid}{path}"

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def twilio_send_sms(to: str, body: str) -> str:
            """
            Send an SMS via Twilio.

            Args:
                to: Recipient phone number in E.164 format (e.g. "+8613800138000").
                body: Message text (max 1600 chars).
            """
            import json
            import urllib.parse
            import urllib.request

            data = urllib.parse.urlencode(
                {
                    "To": to,
                    "From": integration.env("TWILIO_FROM_NUMBER"),
                    "Body": body,
                }
            ).encode()
            req = urllib.request.Request(
                integration._api("/Messages.json"),
                data=data,
                headers={
                    **integration._auth(),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())
                    return integration.ok(
                        {
                            "sid": result.get("sid"),
                            "status": result.get("status"),
                            "to": result.get("to"),
                        }
                    )
            except Exception as error:
                return integration.ok({"error": str(error)})

        @mcp.tool()
        def twilio_list_messages(limit: int = 10) -> str:
            """
            List recent SMS messages sent via Twilio.

            Args:
                limit: Number of messages to return (default 10).
            """
            import urllib.parse

            params = urllib.parse.urlencode({"PageSize": limit})
            result = integration.get(
                integration._api(f"/Messages.json?{params}"),
                integration._auth(),
            )
            messages = result.get("messages", [])
            lines = [f"{len(messages)} message(s):"]
            for message in messages:
                lines.append(
                    f"  [{message.get('status')}] To:{message.get('to')} — "
                    f"{message.get('body', '')[:60]}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def twilio_get_account() -> str:
            """Get Twilio account info and balance."""
            result = integration.get(integration._api(".json"), integration._auth())
            return integration.ok(
                {
                    "sid": result.get("sid"),
                    "friendly_name": result.get("friendly_name"),
                    "status": result.get("status"),
                    "type": result.get("type"),
                }
            )