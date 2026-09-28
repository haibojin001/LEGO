from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

from .base import BaseIntegration


class ConfluenceIntegration(BaseIntegration):
    name = "confluence"
    label = "Confluence (Atlassian)"
    env_vars = {
        "CONFLUENCE_EMAIL": "Atlassian account email",
        "CONFLUENCE_API_TOKEN": "Atlassian API token",
        "CONFLUENCE_BASE_URL": (
            "Confluence base URL (e.g. https://myorg.atlassian.net)"
        ),
    }

    def _headers(self) -> dict:
        username = self.env("CONFLUENCE_EMAIL")
        token = self.env("CONFLUENCE_API_TOKEN")
        credentials = base64.b64encode(f"{username}:{token}".encode()).decode()
        return {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        base_url = self.env("CONFLUENCE_BASE_URL").rstrip("/")
        url = f"{base_url}/wiki/rest/api/{path}"

        if params:
            url += "?" + urllib.parse.urlencode(params)

        request = urllib.request.Request(url, headers=self._headers())

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def _post(self, path: str, data: dict) -> dict:
        base_url = self.env("CONFLUENCE_BASE_URL").rstrip("/")
        url = f"{base_url}/wiki/rest/api/{path}"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers=headers,
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def register(self, mcp) -> None:
        success = self.ok

        @mcp.tool()
        def confluence_search_pages(query: str, limit: int = 10) -> str:
            """Search Confluence pages by text. Returns matching pages."""
            return success(
                self._get(
                    "search",
                    {
                        "cql": f'type=page AND text~"{query}"',
                        "limit": limit,
                    },
                )
            )

        @mcp.tool()
        def confluence_get_page(page_id: str) -> str:
            """Get a Confluence page by ID, including its body content."""
            return success(
                self._get(
                    f"content/{page_id}",
                    {"expand": "body.storage,version,ancestors"},
                )
            )

        @mcp.tool()
        def confluence_list_spaces(limit: int = 25) -> str:
            """List all Confluence spaces."""
            return success(self._get("space", {"limit": limit}))

        @mcp.tool()
        def confluence_list_pages_in_space(
            space_key: str, limit: int = 25
        ) -> str:
            """List pages in a specific Confluence space."""
            return success(
                self._get(
                    "content",
                    {
                        "spaceKey": space_key,
                        "type": "page",
                        "limit": limit,
                    },
                )
            )

        @mcp.tool()
        def confluence_create_page(
            space_key: str,
            title: str,
            content_html: str,
            parent_id: str = "",
        ) -> str:
            """Create a new Confluence page. content_html is the page body in HTML."""
            payload: dict = {
                "type": "page",
                "title": title,
                "space": {"key": space_key},
                "body": {
                    "storage": {
                        "value": content_html,
                        "representation": "storage",
                    }
                },
            }

            if parent_id:
                payload["ancestors"] = [{"id": parent_id}]

            return success(self._post("content", payload))