"""Notion integration."""

from __future__ import annotations

from .base import BaseIntegration

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"


class NotionIntegration(BaseIntegration):
    name = "notion"
    label = "Notion"
    env_vars = {
        "NOTION_API_KEY": "Integration token from notion.so/my-integrations",
    }

    def _auth(self) -> dict:
        return {
            "Authorization": f"Bearer {self.env('NOTION_API_KEY')}",
            "Notion-Version": VERSION,
        }

    def register(self, mcp) -> None:
        current = self

        @mcp.tool()
        def notion_search(query: str, filter_type: str = "") -> str:
            """
            Search Notion pages and databases.

            Args:
                query: Search query string.
                filter_type: Optional "page" or "database" to filter results.
            """
            payload: dict = {"query": query}
            if filter_type in ("page", "database"):
                payload["filter"] = {
                    "value": filter_type,
                    "property": "object",
                }

            response = current.post(
                f"{API}/search",
                payload,
                current._auth(),
            )
            entries = response.get("results", [])
            output = [f"Found {len(entries)} result(s):"]

            for entry in entries[:10]:
                title = ""
                properties = entry.get("properties", {})

                if "title" in properties:
                    title = "".join(
                        text.get("plain_text", "")
                        for text in properties["title"].get("title", [])
                    )
                elif entry.get("object") == "database":
                    title = "".join(
                        text.get("plain_text", "")
                        for text in entry.get("title", [])
                    )

                output.append(f"  [{entry['object']}] {title} — {entry['id']}")

            return "\n".join(output)

        @mcp.tool()
        def notion_create_page(parent_id: str, title: str, content: str = "") -> str:
            """
            Create a new Notion page.

            Args:
                parent_id: ID of the parent page or database.
                title: Page title.
                content: Optional plain text content for the first paragraph.
            """
            payload: dict = {
                "parent": {"page_id": parent_id},
                "properties": {
                    "title": {
                        "title": [
                            {
                                "text": {
                                    "content": title,
                                }
                            }
                        ]
                    }
                },
            }

            if content:
                payload["children"] = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "text": {
                                        "content": content,
                                    }
                                }
                            ]
                        },
                    }
                ]

            response = current.post(
                f"{API}/pages",
                payload,
                current._auth(),
            )
            return current.ok(
                {
                    "id": response.get("id"),
                    "url": response.get("url"),
                }
            )

        @mcp.tool()
        def notion_query_database(database_id: str, filter_json: str = "") -> str:
            """
            Query a Notion database.

            Args:
                database_id: Database ID.
                filter_json: Optional JSON filter string per Notion API spec.
            """
            import json

            payload: dict = {}
            if filter_json:
                payload["filter"] = json.loads(filter_json)

            response = current.post(
                f"{API}/databases/{database_id}/query",
                payload,
                current._auth(),
            )
            entries = response.get("results", [])
            output = [f"Found {len(entries)} row(s):"]

            for entry in entries[:20]:
                properties = entry.get("properties", {})
                name = next(
                    (
                        "".join(
                            text.get("plain_text", "")
                            for text in value.get("title", [])
                        )
                        for value in properties.values()
                        if value.get("type") == "title"
                    ),
                    entry["id"],
                )
                output.append(f"  {name} — {entry['id']}")

            return "\n".join(output)

        @mcp.tool()
        def notion_append_block(
            page_id: str,
            content: str,
            block_type: str = "paragraph",
        ) -> str:
            """
            Append a block to an existing Notion page.

            Args:
                page_id: Target page ID.
                content: Text content to append.
                block_type: "paragraph", "heading_1", "heading_2", "bulleted_list_item", "to_do".
            """
            block: dict = {
                "object": "block",
                "type": block_type,
                block_type: {
                    "rich_text": [
                        {
                            "text": {
                                "content": content,
                            }
                        }
                    ]
                },
            }

            if block_type == "to_do":
                block[block_type]["checked"] = False

            response = current.patch(
                f"{API}/blocks/{page_id}/children",
                {"children": [block]},
                current._auth(),
            )
            return current.ok(response)