from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .base import BaseIntegration


class AsanaIntegration(BaseIntegration):
    name = "asana"
    label = "Asana"
    env_vars = {"ASANA_ACCESS_TOKEN": "Asana personal access token"}

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.env('ASANA_ACCESS_TOKEN')}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"https://app.asana.com/api/1.0/{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        request = urllib.request.Request(url, headers=self._headers())

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def _post(self, path: str, data: dict) -> dict:
        url = f"https://app.asana.com/api/1.0/{path}"
        payload = json.dumps({"data": data}).encode()
        headers = {**self._headers(), "Content-Type": "application/json"}
        request = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def _put(self, path: str, data: dict) -> dict:
        url = f"https://app.asana.com/api/1.0/{path}"
        payload = json.dumps({"data": data}).encode()
        headers = {**self._headers(), "Content-Type": "application/json"}
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="PUT",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": exc.code, "msg": exc.read().decode()}

    def register(self, mcp) -> None:
        ok = self.ok

        @mcp.tool()
        def asana_list_workspaces() -> str:
            """List all Asana workspaces accessible by the current user."""
            return ok(self._get("workspaces"))

        @mcp.tool()
        def asana_list_projects(workspace_id: str) -> str:
            """List all projects in an Asana workspace."""
            return ok(
                self._get(
                    "projects",
                    {
                        "workspace": workspace_id,
                        "opt_fields": "name,color,status",
                    },
                )
            )

        @mcp.tool()
        def asana_list_tasks(project_id: str, completed: bool = False) -> str:
            """List tasks in an Asana project. Set completed=True to include finished tasks."""
            params: dict = {
                "project": project_id,
                "opt_fields": "name,assignee.name,due_on,completed,notes",
            }
            if not completed:
                params["completed_since"] = "now"
            return ok(self._get("tasks", params))

        @mcp.tool()
        def asana_get_task(task_id: str) -> str:
            """Get full details of an Asana task."""
            return ok(
                self._get(
                    f"tasks/{task_id}",
                    {
                        "opt_fields": (
                            "name,notes,due_on,assignee.name,"
                            "completed,subtasks.name"
                        )
                    },
                )
            )

        @mcp.tool()
        def asana_create_task(
            project_id: str,
            name: str,
            notes: str = "",
            due_on: str = "",
            assignee: str = "",
        ) -> str:
            """Create an Asana task. due_on format: YYYY-MM-DD. assignee is an email or GID."""
            data: dict = {"name": name, "projects": [project_id]}
            if notes:
                data["notes"] = notes
            if due_on:
                data["due_on"] = due_on
            if assignee:
                data["assignee"] = assignee
            return ok(self._post("tasks", data))

        @mcp.tool()
        def asana_update_task(
            task_id: str,
            name: str = "",
            notes: str = "",
            completed: bool | None = None,
            due_on: str = "",
        ) -> str:
            """Update an Asana task. Only provided fields are changed."""
            data: dict = {}
            if name:
                data["name"] = name
            if notes:
                data["notes"] = notes
            if completed is not None:
                data["completed"] = completed
            if due_on:
                data["due_on"] = due_on
            return ok(self._put(f"tasks/{task_id}", data))