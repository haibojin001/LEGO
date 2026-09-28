from __future__ import annotations

from .base import BaseIntegration


class TeamsIntegration(BaseIntegration):
    name = "teams"
    label = "Microsoft Teams"
    env_vars = {
        "TEAMS_WEBHOOK_URL": "Incoming webhook URL from Teams channel connector settings",
    }

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def teams_send_message(text: str) -> str:
            """
            Send a message to a Microsoft Teams channel via webhook.

            Args:
                text: Message text (supports Markdown).
            """
            response = integration.post(
                integration.env("TEAMS_WEBHOOK_URL"),
                {"text": text},
            )
            return integration.ok({"sent": True, "response": str(response)})

        @mcp.tool()
        def teams_send_card(title: str, text: str, theme_color: str = "0076D7") -> str:
            """
            Send a rich message card to a Microsoft Teams channel.

            Args:
                title: Card title.
                text: Card body text (Markdown).
                theme_color: Hex color for the card border (default blue "0076D7").
            """
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": theme_color,
                "summary": title,
                "sections": [
                    {
                        "activityTitle": title,
                        "activityText": text,
                    }
                ],
            }
            integration.post(integration.env("TEAMS_WEBHOOK_URL"), payload)
            return integration.ok({"sent": True, "title": title})

        @mcp.tool()
        def teams_send_alert(
            title: str,
            message: str,
            severity: str = "warning",
        ) -> str:
            """
            Send a color-coded alert card to Microsoft Teams.

            Args:
                title: Alert title.
                message: Alert details.
                severity: "info" (blue), "warning" (yellow), "error" (red), "success" (green).
            """
            colors = {
                "info": "0076D7",
                "warning": "FFA500",
                "error": "FF0000",
                "success": "00B050",
            }
            color = colors.get(severity, "0076D7")
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": title,
                "sections": [
                    {
                        "activityTitle": f"[{severity.upper()}] {title}",
                        "activityText": message,
                    }
                ],
            }
            integration.post(integration.env("TEAMS_WEBHOOK_URL"), payload)
            return integration.ok(
                {"sent": True, "severity": severity, "title": title}
            )