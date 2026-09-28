from __future__ import annotations

from .base import BaseIntegration

API = "https://api.linear.app/graphql"


class LinearIntegration(BaseIntegration):
    name = "linear"
    label = "Linear"
    env_vars = {
        "LINEAR_API_KEY": "API key from linear.app/settings/api",
    }

    def _auth(self) -> dict:
        return {"Authorization": self.env("LINEAR_API_KEY")}

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        body = {
            "query": query,
            "variables": variables or {},
        }
        return self.post(API, body, self._auth())

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def linear_create_issue(
            team_id: str,
            title: str,
            description: str = "",
            priority: int = 0,
        ) -> str:
            """
            Create a Linear issue.

            Args:
                team_id: Linear team ID (get from linear_list_teams).
                title: Issue title.
                description: Issue description (Markdown).
                priority: 0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low.
            """
            operation = """
            mutation CreateIssue($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue { id identifier title url }
              }
            }
            """
            response = integration._gql(
                operation,
                {
                    "input": {
                        "teamId": team_id,
                        "title": title,
                        "description": description,
                        "priority": priority,
                    }
                },
            )
            issue = response.get("data", {}).get("issueCreate", {}).get("issue", {})
            return integration.ok(
                {
                    "id": issue.get("id"),
                    "identifier": issue.get("identifier"),
                    "url": issue.get("url"),
                }
            )

        @mcp.tool()
        def linear_list_issues(team_id: str = "", state: str = "") -> str:
            """
            List Linear issues.

            Args:
                team_id: Optional team ID to filter by.
                state: Optional state filter ("Todo", "In Progress", "Done", etc.).
            """
            criteria = []
            if team_id:
                criteria.append(f'team: {{id: {{eq: "{team_id}"}}}}')
            if state:
                criteria.append(f'state: {{name: {{eq: "{state}"}}}}')

            filter_clause = "{" + ", ".join(criteria) + "}" if criteria else ""
            operation = f"""
            query {{
              issues(filter: {filter_clause} first: 20) {{
                nodes {{ id identifier title state {{ name }} priority url }}
              }}
            }}
            """
            response = integration._gql(operation)
            issues = response.get("data", {}).get("issues", {}).get("nodes", [])

            output = [f"Found {len(issues)} issue(s):"]
            for issue in issues:
                state_name = issue.get("state", {}).get("name", "?")
                output.append(
                    f"  [{issue.get('identifier')}] [{state_name}] "
                    f"{issue.get('title')} — {issue.get('url')}"
                )
            return "\n".join(output)

        @mcp.tool()
        def linear_list_teams() -> str:
            """List all Linear teams in the workspace."""
            operation = "{ teams { nodes { id name key } } }"
            response = integration._gql(operation)
            teams = response.get("data", {}).get("teams", {}).get("nodes", [])

            output = [f"Found {len(teams)} team(s):"]
            for team in teams:
                output.append(
                    f"  [{team.get('key')}] {team.get('name')} — {team.get('id')}"
                )
            return "\n".join(output)

        @mcp.tool()
        def linear_update_issue(
            issue_id: str,
            state_id: str = "",
            title: str = "",
            description: str = "",
        ) -> str:
            """
            Update a Linear issue.

            Args:
                issue_id: Linear issue ID.
                state_id: New workflow state ID (optional).
                title: New title (optional).
                description: New description (optional).
            """
            changes: dict = {}
            if state_id:
                changes["stateId"] = state_id
            if title:
                changes["title"] = title
            if description:
                changes["description"] = description

            operation = """
            mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
              issueUpdate(id: $id, input: $input) {
                success
                issue { id identifier title }
              }
            }
            """
            response = integration._gql(
                operation,
                {"id": issue_id, "input": changes},
            )
            issue = response.get("data", {}).get("issueUpdate", {}).get("issue", {})
            return integration.ok(
                {
                    "id": issue.get("id"),
                    "identifier": issue.get("identifier"),
                    "title": issue.get("title"),
                }
            )