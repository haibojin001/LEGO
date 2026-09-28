"""Mailchimp email marketing integration."""
from __future__ import annotations

import base64
import urllib.parse

from .base import BaseIntegration


class MailchimpIntegration(BaseIntegration):
    name = "mailchimp"
    label = "Mailchimp"
    env_vars = {
        "MAILCHIMP_API_KEY": (
            "API key from mailchimp.com/account/api/ (ends with -us1, -us6, etc.)"
        ),
    }

    def _api(self, path: str) -> str:
        api_key = self.env("MAILCHIMP_API_KEY")
        data_center = api_key.rsplit("-", 1)[-1] if "-" in api_key else "us1"
        return f"https://{data_center}.api.mailchimp.com/3.0{path}"

    def _auth(self) -> dict:
        credentials = f"anystring:{self.env('MAILCHIMP_API_KEY')}"
        token = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def mailchimp_list_audiences() -> str:
            """List all Mailchimp audiences (mailing lists)."""
            result = integration.get(
                integration._api("/lists?count=20"),
                integration._auth(),
            )
            audiences = result.get("lists", [])
            output = [f"Found {len(audiences)} audience(s):"]

            for audience in audiences:
                statistics = audience.get("stats", {})
                output.append(
                    f"  [{audience['id']}] {audience['name']} — "
                    f"{statistics.get('member_count', 0)} members, "
                    f"{statistics.get('open_rate', 0):.1%} open rate"
                )

            return "\n".join(output)

        @mcp.tool()
        def mailchimp_add_subscriber(
            list_id: str,
            email: str,
            first_name: str = "",
            last_name: str = "",
        ) -> str:
            """
            Add or update a subscriber in a Mailchimp audience.

            Args:
                list_id: Audience/list ID (from mailchimp_list_audiences).
                email: Subscriber email address.
                first_name: First name (optional).
                last_name: Last name (optional).
            """
            import hashlib

            subscriber_hash = hashlib.md5(email.lower().encode()).hexdigest()
            body: dict = {
                "email_address": email,
                "status": "subscribed",
            }

            if first_name or last_name:
                body["merge_fields"] = {}
                if first_name:
                    body["merge_fields"]["FNAME"] = first_name
                if last_name:
                    body["merge_fields"]["LNAME"] = last_name

            result = integration.put(
                integration._api(f"/lists/{list_id}/members/{subscriber_hash}"),
                body,
                integration._auth(),
            )
            return integration.ok(
                {
                    "id": result.get("id"),
                    "email": result.get("email_address"),
                    "status": result.get("status"),
                }
            )

        @mcp.tool()
        def mailchimp_list_campaigns(limit: int = 10) -> str:
            """
            List recent Mailchimp campaigns.

            Args:
                limit: Max campaigns to return (default 10).
            """
            query = urllib.parse.urlencode(
                {
                    "count": limit,
                    "sort_field": "create_time",
                    "sort_dir": "DESC",
                }
            )
            result = integration.get(
                integration._api(f"/campaigns?{query}"),
                integration._auth(),
            )
            campaigns = result.get("campaigns", [])
            output = [f"Found {len(campaigns)} campaign(s):"]

            for campaign in campaigns:
                settings = campaign.get("settings", {})
                output.append(
                    f"  [{campaign['id']}] [{campaign['status']}] "
                    f"{settings.get('subject_line', '(no subject)')} "
                    f"— sent: {campaign.get('emails_sent', 0)}"
                )

            return "\n".join(output)