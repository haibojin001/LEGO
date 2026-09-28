"""Sentry error tracking integration."""
from __future__ import annotations

import urllib.parse

from .base import BaseIntegration

API = "https://sentry.io/api/0"


class SentryIntegration(BaseIntegration):
    name = "sentry"
    label = "Sentry"
    env_vars = {
        "SENTRY_AUTH_TOKEN": "Auth token from sentry.io/settings/account/api/auth-tokens/",
        "SENTRY_ORG_SLUG": "Your Sentry organization slug (from the URL: sentry.io/organizations/<slug>)",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.env('SENTRY_AUTH_TOKEN')}"}

    def _org(self) -> str:
        return self.env("SENTRY_ORG_SLUG")

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def sentry_list_issues(
            project_slug: str,
            limit: int = 20,
            query: str = "is:unresolved",
        ) -> str:
            """
            List Sentry issues (errors) for a project.

            Args:
                project_slug: Project slug (from sentry.io/organizations/.../projects/).
                limit: Max issues to return (default 20).
                query: Sentry search query (default "is:unresolved").
            """
            params = urllib.parse.urlencode({"limit": limit, "query": query})
            response = integration.get(
                f"{API}/projects/{integration._org()}/{project_slug}/issues/?{params}",
                integration._auth(),
            )
            issues = response if isinstance(response, list) else []
            lines = [f"Found {len(issues)} issue(s):"]
            for issue in issues:
                lines.append(
                    f"  [{issue['id']}] [{issue.get('level', '?').upper()}] "
                    f"{issue['title']} — {issue.get('count', '?')} events, "
                    f"last seen {issue.get('lastSeen', '')[:10]}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def sentry_get_issue(issue_id: str) -> str:
            """
            Get details of a specific Sentry issue.

            Args:
                issue_id: Sentry issue ID.
            """
            response = integration.get(
                f"{API}/issues/{issue_id}/",
                integration._auth(),
            )
            return integration.ok(
                {
                    "id": response.get("id"),
                    "title": response.get("title"),
                    "level": response.get("level"),
                    "status": response.get("status"),
                    "count": response.get("count"),
                    "first_seen": response.get("firstSeen"),
                    "last_seen": response.get("lastSeen"),
                    "permalink": response.get("permalink"),
                }
            )

        @mcp.tool()
        def sentry_list_projects() -> str:
            """List all Sentry projects in the organization."""
            response = integration.get(
                f"{API}/organizations/{integration._org()}/projects/",
                integration._auth(),
            )
            projects = response if isinstance(response, list) else []
            lines = [f"Found {len(projects)} project(s):"]
            for project in projects:
                lines.append(
                    f"  [{project['slug']}] {project['name']} — "
                    f"platform: {project.get('platform', '?')}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def sentry_resolve_issue(issue_id: str) -> str:
            """
            Mark a Sentry issue as resolved.

            Args:
                issue_id: Sentry issue ID to resolve.
            """
            response = integration.put(
                f"{API}/issues/{issue_id}/",
                {"status": "resolved"},
                integration._auth(),
            )
            return integration.ok(
                {"id": issue_id, "status": response.get("status", "resolved")}
            )