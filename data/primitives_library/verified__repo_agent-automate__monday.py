from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import BaseIntegration


class MondayIntegration(BaseIntegration):
    name = "monday"
    label = "Monday.com"
    env_vars = {"MONDAY_API_KEY": "Monday.com API key"}

    def _graphql(self, query: str, variables: dict | None = None) -> dict:
        endpoint = "https://api.monday.com/v2"
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": self.env("MONDAY_API_KEY"),
                "Content-Type": "application/json",
                "API-Version": "2024-01",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return {"error": error.code, "msg": error.read().decode()}

    def register(self, mcp) -> None:
        ok = self.ok

        @mcp.tool()
        def monday_list_boards(limit: int = 10) -> str:
            """List Monday.com boards."""
            query = f"{{ boards(limit: {limit}) {{ id name description state }} }}"
            return ok(self._graphql(query))

        @mcp.tool()
        def monday_list_items(board_id: str, limit: int = 20) -> str:
            """List items (rows) on a Monday.com board."""
            query = f"""{{
              boards(ids: [{board_id}]) {{
                items_page(limit: {limit}) {{
                  items {{
                    id name state
                    column_values {{ id text }}
                  }}
                }}
              }}
            }}"""
            return ok(self._graphql(query))

        @mcp.tool()
        def monday_create_item(
            board_id: str, item_name: str, group_id: str = ""
        ) -> str:
            """Create a new item (row) on a Monday.com board."""
            query = """mutation($board_id: ID!, $item_name: String!, $group_id: String) {
              create_item(board_id: $board_id, item_name: $item_name, group_id: $group_id) {
                id name
              }
            }"""
            variables: dict = {"board_id": board_id, "item_name": item_name}
            if group_id:
                variables["group_id"] = group_id
            return ok(self._graphql(query, variables))

        @mcp.tool()
        def monday_get_item(item_id: str) -> str:
            """Get a specific Monday.com item by ID."""
            query = f"""{{
              items(ids: [{item_id}]) {{
                id name state board {{ id name }}
                column_values {{ id title text }}
              }}
            }}"""
            return ok(self._graphql(query))

        @mcp.tool()
        def monday_list_groups(board_id: str) -> str:
            """List groups (sections) on a Monday.com board."""
            query = f"{{ boards(ids: [{board_id}]) {{ groups {{ id title color }} }} }}"
            return ok(self._graphql(query))