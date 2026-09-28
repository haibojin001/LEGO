from __future__ import annotations

from .base import BaseIntegration

API = "https://www.yuque.com/api/v2"


class YuqueIntegration(BaseIntegration):
    name = "yuque"
    label = "语雀 (Yuque)"
    env_vars = {
        "YUQUE_TOKEN": "Personal access token from yuque.com/settings/tokens",
    }

    def _auth(self) -> dict:
        return {"X-Auth-Token": self.env("YUQUE_TOKEN")}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def yuque_get_user() -> str:
            """Get current authenticated Yuque user info."""
            response = integration.get(f"{API}/user", integration._auth())
            data = response.get("data", {})
            return integration.ok(
                {
                    "id": data.get("id"),
                    "login": data.get("login"),
                    "name": data.get("name"),
                    "description": data.get("description"),
                }
            )

        @mcp.tool()
        def yuque_list_repos(login: str = "") -> str:
            """
            List Yuque knowledge bases (repos) for a user or org.

            Args:
                login: User/org login name. Empty = current user's repos.
            """
            if login:
                response = integration.get(
                    f"{API}/users/{login}/repos", integration._auth()
                )
            else:
                response = integration.get(f"{API}/mine/repos", integration._auth())

            repos = response.get("data", [])
            lines = [f"Found {len(repos)} knowledge base(s):"]
            for repo in repos:
                lines.append(
                    f"  [{repo.get('namespace')}] {repo.get('name')} — "
                    f"{repo.get('description', '')}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def yuque_list_docs(namespace: str) -> str:
            """
            List documents in a Yuque knowledge base.

            Args:
                namespace: Knowledge base namespace (e.g. "username/repo-slug").
            """
            response = integration.get(
                f"{API}/repos/{namespace}/docs", integration._auth()
            )
            docs = response.get("data", [])
            lines = [f"Found {len(docs)} doc(s) in {namespace}:"]
            for doc in docs:
                lines.append(
                    f"  [{doc.get('slug')}] {doc.get('title')} — updated "
                    f"{doc.get('updated_at', '')[:10]}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def yuque_get_doc(namespace: str, slug: str) -> str:
            """
            Get a Yuque document's content.

            Args:
                namespace: Knowledge base namespace (e.g. "username/repo-slug").
                slug: Document slug.
            """
            response = integration.get(
                f"{API}/repos/{namespace}/docs/{slug}", integration._auth()
            )
            data = response.get("data", {})
            return integration.ok(
                {
                    "title": data.get("title"),
                    "slug": data.get("slug"),
                    "body": (data.get("body") or "")[:2000],
                    "updated_at": data.get("updated_at"),
                }
            )

        @mcp.tool()
        def yuque_create_doc(
            namespace: str, title: str, body: str, slug: str = ""
        ) -> str:
            """
            Create a new document in a Yuque knowledge base.

            Args:
                namespace: Knowledge base namespace.
                title: Document title.
                body: Document content (Markdown).
                slug: URL-friendly slug (auto-generated if empty).
            """
            payload: dict = {
                "title": title,
                "body": body,
                "format": "markdown",
                "status": 1,
            }
            if slug:
                payload["slug"] = slug

            response = integration.post(
                f"{API}/repos/{namespace}/docs", payload, integration._auth()
            )
            data = response.get("data", {})
            return integration.ok(
                {
                    "id": data.get("id"),
                    "slug": data.get("slug"),
                    "title": data.get("title"),
                }
            )