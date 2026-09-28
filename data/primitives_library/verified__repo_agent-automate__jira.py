"""Jira integration."""
from __future__ import annotations

import base64

from .base import BaseIntegration


class JiraIntegration(BaseIntegration):
    name = "jira"
    label = "Jira"
    env_vars = {
        "JIRA_BASE_URL": "Your Jira instance URL (e.g. https://yourcompany.atlassian.net)",
        "JIRA_EMAIL": "Your Atlassian account email",
        "JIRA_API_TOKEN": "API token from id.atlassian.com/manage-profile/security/api-tokens",
    }

    def _auth(self) -> dict:
        credentials = f"{self.env('JIRA_EMAIL')}:{self.env('JIRA_API_TOKEN')}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    def _api(self, path: str) -> str:
        base = self.env("JIRA_BASE_URL").rstrip("/")
        return f"{base}/rest/api/3{path}"

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def jira_create_issue(
            project_key: str,
            summary: str,
            description: str = "",
            issue_type: str = "Task",
        ) -> str:
            """
            Create a Jira issue.

            Args:
                project_key: Jira project key (e.g. "PROJ").
                summary: Issue summary/title.
                description: Issue description (plain text).
                issue_type: Issue type name (default "Task"; also "Bug", "Story", "Epic").
            """
            body = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": summary,
                    "issuetype": {"name": issue_type},
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": description}],
                            }
                        ],
                    }
                    if description
                    else None,
                }
            }
            if not description:
                del body["fields"]["description"]
            result = integration.post(
                integration._api("/issue"),
                body,
                integration._auth(),
            )
            return integration.ok(
                {
                    "id": result.get("id"),
                    "key": result.get("key"),
                    "url": f"{integration.env('JIRA_BASE_URL')}/browse/{result.get('key')}",
                }
            )

        @mcp.tool()
        def jira_search_issues(jql: str, max_results: int = 20) -> str:
            """
            Search Jira issues using JQL (Jira Query Language).

            Args:
                jql: JQL query string (e.g. 'project=PROJ AND status="In Progress"').
                max_results: Maximum results to return (default 20).
            """
            result = integration.post(
                integration._api("/search"),
                {
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": ["summary", "status", "assignee", "priority"],
                },
                integration._auth(),
            )
            issues = result.get("issues", [])
            lines = [f"Found {result.get('total', len(issues))} issue(s):"]
            for issue in issues:
                fields = issue.get("fields", {})
                status = fields.get("status", {}).get("name", "?")
                lines.append(
                    f"  [{issue['key']}] [{status}] {fields.get('summary', '')}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def jira_get_issue(issue_key: str) -> str:
            """
            Get details of a specific Jira issue.

            Args:
                issue_key: Issue key (e.g. "PROJ-123").
            """
            result = integration.get(
                integration._api(f"/issue/{issue_key}"),
                integration._auth(),
            )
            fields = result.get("fields", {})
            return integration.ok(
                {
                    "key": result.get("key"),
                    "summary": fields.get("summary"),
                    "status": fields.get("status", {}).get("name"),
                    "assignee": (fields.get("assignee") or {}).get("displayName"),
                    "priority": (fields.get("priority") or {}).get("name"),
                    "url": f"{integration.env('JIRA_BASE_URL')}/browse/{result.get('key')}",
                }
            )

        @mcp.tool()
        def jira_transition_issue(issue_key: str, transition_name: str) -> str:
            """
            Transition a Jira issue to a new status.

            Args:
                issue_key: Issue key (e.g. "PROJ-123").
                transition_name: Target status name (e.g. "In Progress", "Done").
            """
            transitions_result = integration.get(
                integration._api(f"/issue/{issue_key}/transitions"),
                integration._auth(),
            )
            transitions = transitions_result.get("transitions", [])
            transition_id = next(
                (
                    transition["id"]
                    for transition in transitions
                    if transition["name"].lower() == transition_name.lower()
                ),
                None,
            )
            if not transition_id:
                available = [transition["name"] for transition in transitions]
                return integration.ok(
                    {
                        "error": f"Transition '{transition_name}' not found",
                        "available": available,
                    }
                )
            integration.post(
                integration._api(f"/issue/{issue_key}/transitions"),
                {"transition": {"id": transition_id}},
                integration._auth(),
            )
            return integration.ok(
                {
                    "success": True,
                    "issue": issue_key,
                    "new_status": transition_name,
                }
            )