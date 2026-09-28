"""SendGrid email integration."""
from __future__ import annotations

from .base import BaseIntegration

API = "https://api.sendgrid.com/v3"


class SendGridIntegration(BaseIntegration):
    name = "sendgrid"
    label = "SendGrid"
    env_vars = {
        "SENDGRID_API_KEY": "API key from app.sendgrid.com/settings/api_keys",
        "SENDGRID_FROM_EMAIL": "Verified sender email address",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.env('SENDGRID_API_KEY')}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def sendgrid_send_email(
            to_email: str,
            subject: str,
            body: str,
            is_html: bool = False,
        ) -> str:
            """
            Send an email via SendGrid.

            Args:
                to_email: Recipient email address.
                subject: Email subject.
                body: Email body (plain text or HTML).
                is_html: Set True if body contains HTML (default False).
            """
            content_type = "text/html" if is_html else "text/plain"
            payload = {
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": integration.env("SENDGRID_FROM_EMAIL")},
                "subject": subject,
                "content": [{"type": content_type, "value": body}],
            }
            r = integration.post(f"{API}/mail/send", payload, integration._auth())
            return integration.ok({"sent": True, "to": to_email, "subject": subject})

        @mcp.tool()
        def sendgrid_send_bulk(to_emails: str, subject: str, body: str) -> str:
            """
            Send an email to multiple recipients via SendGrid.

            Args:
                to_emails: Comma-separated list of recipient email addresses.
                subject: Email subject.
                body: Email body (plain text).
            """
            recipients = [{"email": e.strip()} for e in to_emails.split(",") if e.strip()]
            payload = {
                "personalizations": [{"to": recipients}],
                "from": {"email": integration.env("SENDGRID_FROM_EMAIL")},
                "subject": subject,
                "content": [{"type": "text/plain", "value": body}],
            }
            integration.post(f"{API}/mail/send", payload, integration._auth())
            return integration.ok(
                {"sent": True, "recipients": len(recipients), "subject": subject}
            )

        @mcp.tool()
        def sendgrid_get_stats(days: int = 7) -> str:
            """
            Get email delivery statistics for the last N days.

            Args:
                days: Number of days to look back (default 7).
            """
            from datetime import date, timedelta
            import urllib.parse

            start = (date.today() - timedelta(days=days)).isoformat()
            params = urllib.parse.urlencode(
                {"start_date": start, "aggregated_by": "day"}
            )
            r = integration.get(f"{API}/stats?{params}", integration._auth())
            stats = r if isinstance(r, list) else []
            lines = [f"SendGrid stats ({days} days):"]
            for s in stats:
                m = s.get("stats", [{}])[0].get("metrics", {})
                lines.append(
                    f"  {s['date']}: requests={m.get('requests', 0)} "
                    f"delivered={m.get('delivered', 0)} opens={m.get('opens', 0)} "
                    f"clicks={m.get('clicks', 0)} bounces={m.get('bounces', 0)}"
                )
            return "\n".join(lines)