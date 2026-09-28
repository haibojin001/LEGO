from __future__ import annotations

from .base import BaseIntegration

API = "https://api.hubapi.com"


class HubSpotIntegration(BaseIntegration):
    name = "hubspot"
    label = "HubSpot"
    env_vars = {
        "HUBSPOT_ACCESS_TOKEN": "Private app access token from app.hubspot.com/private-apps",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.env('HUBSPOT_ACCESS_TOKEN')}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def hubspot_create_contact(
            email: str,
            firstname: str = "",
            lastname: str = "",
            company: str = "",
            phone: str = "",
        ) -> str:
            """
            Create a HubSpot contact.

            Args:
                email: Contact email address.
                firstname: First name.
                lastname: Last name.
                company: Company name.
                phone: Phone number.
            """
            properties = {"email": email}
            if firstname:
                properties["firstname"] = firstname
            if lastname:
                properties["lastname"] = lastname
            if company:
                properties["company"] = company
            if phone:
                properties["phone"] = phone

            response = integration.post(
                f"{API}/crm/v3/objects/contacts",
                {"properties": properties},
                integration._auth(),
            )
            return integration.ok({"id": response.get("id"), "email": email})

        @mcp.tool()
        def hubspot_search_contacts(query: str, limit: int = 10) -> str:
            """
            Search HubSpot contacts by name, email, or company.

            Args:
                query: Search query string.
                limit: Max results (default 10).
            """
            response = integration.post(
                f"{API}/crm/v3/objects/contacts/search",
                {
                    "query": query,
                    "limit": limit,
                    "properties": ["email", "firstname", "lastname", "company"],
                },
                integration._auth(),
            )
            contacts = response.get("results", [])
            output = [f"Found {response.get('total', len(contacts))} contact(s):"]

            for contact in contacts:
                properties = contact.get("properties", {})
                full_name = (
                    f"{properties.get('firstname', '')} "
                    f"{properties.get('lastname', '')}"
                ).strip()
                output.append(
                    f"  [{contact['id']}] {full_name} — "
                    f"{properties.get('email', '')} ({properties.get('company', '')})"
                )

            return "\n".join(output)

        @mcp.tool()
        def hubspot_create_deal(
            name: str,
            stage: str = "appointmentscheduled",
            amount: str = "",
        ) -> str:
            """
            Create a HubSpot deal.

            Args:
                name: Deal name.
                stage: Pipeline stage ID (default "appointmentscheduled").
                amount: Deal amount (optional).
            """
            properties: dict = {
                "dealname": name,
                "dealstage": stage,
                "pipeline": "default",
            }
            if amount:
                properties["amount"] = amount

            response = integration.post(
                f"{API}/crm/v3/objects/deals",
                {"properties": properties},
                integration._auth(),
            )
            return integration.ok(
                {"id": response.get("id"), "name": name, "stage": stage}
            )

        @mcp.tool()
        def hubspot_list_deals(limit: int = 20) -> str:
            """
            List recent HubSpot deals.

            Args:
                limit: Max deals to return (default 20).
            """
            import urllib.parse

            params = urllib.parse.urlencode(
                {
                    "limit": limit,
                    "properties": "dealname,dealstage,amount,closedate",
                }
            )
            response = integration.get(
                f"{API}/crm/v3/objects/deals?{params}",
                integration._auth(),
            )
            deals = response.get("results", [])
            output = [f"Found {len(deals)} deal(s):"]

            for deal in deals:
                properties = deal.get("properties", {})
                output.append(
                    f"  [{deal['id']}] {properties.get('dealname')} — "
                    f"stage: {properties.get('dealstage')} "
                    f"amount: {properties.get('amount', '?')}"
                )

            return "\n".join(output)