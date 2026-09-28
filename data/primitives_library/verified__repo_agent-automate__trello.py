from __future__ import annotations

import urllib.parse

from .base import BaseIntegration


API = "https://api.trello.com/1"


class TrelloIntegration(BaseIntegration):
    name = "trello"
    label = "Trello"
    env_vars = {
        "TRELLO_API_KEY": "API key from trello.com/power-ups/admin",
        "TRELLO_TOKEN": "Token generated from trello.com/1/authorize?...",
    }

    def _auth_params(self) -> str:
        return urllib.parse.urlencode(
            {
                "key": self.env("TRELLO_API_KEY"),
                "token": self.env("TRELLO_TOKEN"),
            }
        )

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def trello_list_boards() -> str:
            """List all Trello boards for the authenticated member."""
            response = integration.get(
                f"{API}/members/me/boards?fields=name,shortUrl&"
                f"{integration._auth_params()}"
            )
            boards = response if isinstance(response, list) else []
            lines = [f"Found {len(boards)} board(s):"]
            for board in boards:
                lines.append(
                    f"  [{board['id']}] {board['name']} — {board.get('shortUrl')}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def trello_list_cards(board_id: str) -> str:
            """
            List all cards on a Trello board.

            Args:
                board_id: Trello board ID (from trello_list_boards).
            """
            response = integration.get(
                f"{API}/boards/{board_id}/cards?{integration._auth_params()}"
            )
            cards = response if isinstance(response, list) else []
            lines = [f"Found {len(cards)} card(s):"]
            for card in cards:
                lines.append(
                    f"  [{card['id']}] {card['name']} — {card.get('shortUrl')}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def trello_create_card(
            list_id: str,
            name: str,
            desc: str = "",
            due: str = "",
        ) -> str:
            """
            Create a new Trello card.

            Args:
                list_id: Target list ID (get from trello_list_lists).
                name: Card name.
                desc: Card description (Markdown).
                due: Due date in ISO format (e.g. "2025-12-31T00:00:00.000Z").
            """
            data: dict = {
                "idList": list_id,
                "name": name,
                "desc": desc,
                "key": integration.env("TRELLO_API_KEY"),
                "token": integration.env("TRELLO_TOKEN"),
            }
            if due:
                data["due"] = due

            response = integration.post(f"{API}/cards", data)
            return integration.ok(
                {
                    "id": response.get("id"),
                    "name": response.get("name"),
                    "url": response.get("shortUrl"),
                }
            )

        @mcp.tool()
        def trello_list_lists(board_id: str) -> str:
            """
            List all lists (columns) on a Trello board.

            Args:
                board_id: Trello board ID.
            """
            response = integration.get(
                f"{API}/boards/{board_id}/lists?{integration._auth_params()}"
            )
            lists = response if isinstance(response, list) else []
            lines = [f"Found {len(lists)} list(s):"]
            for trello_list in lists:
                lines.append(f"  [{trello_list['id']}] {trello_list['name']}")
            return "\n".join(lines)