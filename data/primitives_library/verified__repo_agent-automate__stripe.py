from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .base import BaseIntegration


class StripeIntegration(BaseIntegration):
    name = "stripe"
    label = "Stripe"
    env_vars = {
        "STRIPE_SECRET_KEY": "Stripe secret key (sk_live_... or sk_test_...)",
    }

    def _headers(self):
        key = self.env("STRIPE_SECRET_KEY")
        return {"Authorization": f"Bearer {key}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        address = "https://api.stripe.com/v1/" + path
        if params:
            address += "?" + urllib.parse.urlencode(params)

        request = urllib.request.Request(address, headers=self._headers())

        import urllib.error

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def _post_form(self, path: str, data: dict) -> dict:
        address = "https://api.stripe.com/v1/" + path
        encoded_data = urllib.parse.urlencode(data).encode()
        headers = self._headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            address,
            data=encoded_data,
            headers=headers,
        )

        import urllib.error

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def register(self, mcp) -> None:
        format_result = self.ok

        @mcp.tool()
        def stripe_get_balance() -> str:
            """Retrieve the current Stripe account balance."""
            return format_result(self._get("balance"))

        @mcp.tool()
        def stripe_list_customers(limit: int = 10, email: str = "") -> str:
            """List Stripe customers. Optionally filter by email."""
            query = {"limit": limit}
            if email:
                query["email"] = email
            return format_result(self._get("customers", query))

        @mcp.tool()
        def stripe_create_customer(
            email: str, name: str = "", phone: str = ""
        ) -> str:
            """Create a new Stripe customer."""
            customer_data = {"email": email}
            if name:
                customer_data["name"] = name
            if phone:
                customer_data["phone"] = phone
            return format_result(self._post_form("customers", customer_data))

        @mcp.tool()
        def stripe_list_payments(limit: int = 10) -> str:
            """List recent Stripe payment intents."""
            return format_result(self._get("payment_intents", {"limit": limit}))

        @mcp.tool()
        def stripe_list_products(limit: int = 10) -> str:
            """List Stripe products."""
            return format_result(
                self._get("products", {"limit": limit, "active": "true"})
            )

        @mcp.tool()
        def stripe_get_customer(customer_id: str) -> str:
            """Get a specific Stripe customer by ID."""
            return format_result(self._get(f"customers/{customer_id}"))

        @mcp.tool()
        def stripe_list_subscriptions(
            customer_id: str = "", limit: int = 10
        ) -> str:
            """List Stripe subscriptions. Optionally filter by customer_id."""
            query = {"limit": limit}
            if customer_id:
                query["customer"] = customer_id
            return format_result(self._get("subscriptions", query))