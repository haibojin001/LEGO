from __future__ import annotations

from .base import BaseIntegration

API = "https://api.airtable.com/v0"


class AirtableIntegration(BaseIntegration):
    name = "airtable"
    label = "Airtable"
    env_vars = {
        "AIRTABLE_API_KEY": "Personal access token from airtable.com/account",
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.env('AIRTABLE_API_KEY')}"}

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def airtable_list_records(
            base_id: str,
            table_name: str,
            max_records: int = 20,
        ) -> str:
            """
            List records from an Airtable table.

            Args:
                base_id: Airtable base ID (starts with "app", from the URL).
                table_name: Table name or ID.
                max_records: Maximum records to return (default 20).
            """
            import urllib.parse

            params = urllib.parse.urlencode({"maxRecords": max_records})
            table = urllib.parse.quote(table_name)
            response = integration.get(
                f"{API}/{base_id}/{table}?{params}",
                integration._auth(),
            )
            records = response.get("records", [])
            lines = [f"Found {len(records)} record(s) in {table_name}:"]

            for record in records:
                fields = record.get("fields", {})
                preview = ", ".join(
                    f"{key}={str(value)[:30]}"
                    for key, value in list(fields.items())[:3]
                )
                lines.append(f"  [{record['id']}] {preview}")

            return "\n".join(lines)

        @mcp.tool()
        def airtable_create_record(
            base_id: str,
            table_name: str,
            fields_json: str,
        ) -> str:
            """
            Create a new record in an Airtable table.

            Args:
                base_id: Airtable base ID (starts with "app").
                table_name: Table name or ID.
                fields_json: JSON object of field name → value pairs.
            """
            import json
            import urllib.parse

            fields = json.loads(fields_json)
            table = urllib.parse.quote(table_name)
            response = integration.post(
                f"{API}/{base_id}/{table}",
                {"fields": fields},
                integration._auth(),
            )
            return integration.ok(
                {"id": response.get("id"), "fields": response.get("fields")}
            )

        @mcp.tool()
        def airtable_update_record(
            base_id: str,
            table_name: str,
            record_id: str,
            fields_json: str,
        ) -> str:
            """
            Update an existing Airtable record (PATCH — only specified fields updated).

            Args:
                base_id: Airtable base ID.
                table_name: Table name or ID.
                record_id: Record ID (starts with "rec").
                fields_json: JSON object of field name → new value pairs.
            """
            import json
            import urllib.parse

            fields = json.loads(fields_json)
            table = urllib.parse.quote(table_name)
            response = integration.patch(
                f"{API}/{base_id}/{table}/{record_id}",
                {"fields": fields},
                integration._auth(),
            )
            return integration.ok(
                {"id": response.get("id"), "fields": response.get("fields")}
            )

        @mcp.tool()
        def airtable_search_records(
            base_id: str,
            table_name: str,
            formula: str,
        ) -> str:
            """
            Search Airtable records using a filter formula.

            Args:
                base_id: Airtable base ID.
                table_name: Table name or ID.
                formula: Airtable formula string (e.g. "{Status}='Done'" or "FIND('hello',{Name})").
            """
            import urllib.parse

            table = urllib.parse.quote(table_name)
            params = urllib.parse.urlencode({"filterByFormula": formula})
            response = integration.get(
                f"{API}/{base_id}/{table}?{params}",
                integration._auth(),
            )
            records = response.get("records", [])
            lines = [f"Found {len(records)} matching record(s):"]

            for record in records:
                fields = record.get("fields", {})
                preview = ", ".join(
                    f"{key}={str(value)[:30]}"
                    for key, value in list(fields.items())[:3]
                )
                lines.append(f"  [{record['id']}] {preview}")

            return "\n".join(lines)