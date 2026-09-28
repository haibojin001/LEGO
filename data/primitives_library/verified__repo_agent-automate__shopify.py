from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .base import BaseIntegration


class ShopifyIntegration(BaseIntegration):
    name = "shopify"
    label = "Shopify"
    env_vars = {
        "SHOPIFY_STORE_DOMAIN": "Store domain (e.g. mystore.myshopify.com)",
        "SHOPIFY_ACCESS_TOKEN": "Admin API access token",
    }

    def _headers(self) -> dict:
        return {
            "X-Shopify-Access-Token": self.env("SHOPIFY_ACCESS_TOKEN"),
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        domain = self.env("SHOPIFY_STORE_DOMAIN").rstrip("/")
        return f"https://{domain}/admin/api/2024-01/{path}"

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self._url(path)
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    def _post(self, path: str, data: dict) -> dict:
        url = self._url(path)
        body = json.dumps(data).encode()
        request = urllib.request.Request(url, data=body, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    def register(self, mcp) -> None:
        ok = self.ok

        @mcp.tool()
        def shopify_list_products(limit: int = 10, status: str = "active") -> str:
            """List Shopify products. status: active, draft, archived."""
            return ok(self._get("products.json", {"limit": limit, "status": status}))

        @mcp.tool()
        def shopify_get_product(product_id: str) -> str:
            """Get a specific Shopify product by ID."""
            return ok(self._get(f"products/{product_id}.json"))

        @mcp.tool()
        def shopify_list_orders(limit: int = 10, status: str = "open") -> str:
            """List Shopify orders. status: open, closed, cancelled, any."""
            return ok(self._get("orders.json", {"limit": limit, "status": status}))

        @mcp.tool()
        def shopify_get_order(order_id: str) -> str:
            """Get a specific Shopify order by ID."""
            return ok(self._get(f"orders/{order_id}.json"))

        @mcp.tool()
        def shopify_list_customers(limit: int = 10, query: str = "") -> str:
            """List Shopify customers. Optionally search by name or email."""
            params: dict = {"limit": limit}
            if query:
                params["query"] = query
            return ok(
                self._get(
                    "customers/search.json" if query else "customers.json",
                    params,
                )
            )

        @mcp.tool()
        def shopify_get_shop_info() -> str:
            """Get basic Shopify store information."""
            return ok(self._get("shop.json"))

        @mcp.tool()
        def shopify_list_collections(limit: int = 10) -> str:
            """List Shopify custom collections."""
            return ok(self._get("custom_collections.json", {"limit": limit}))