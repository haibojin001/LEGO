from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Type

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

from promptify.core.exceptions import TemplateMissingVariableError, TemplateNotFoundError

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class PromptBuilder:
    def __init__(self, template: Optional[str] = None) -> None:
        self._template_name = template
        self._jinja_template: Optional[Template] = None
        self._env: Optional[Environment] = None

        if template is not None:
            self._load_template(template)

    def _load_template(self, template: str) -> None:
        name = template if template.endswith(".jinja") else f"{template}.jinja"
        bundled_file = os.path.join(TEMPLATES_DIR, name)

        if os.path.isfile(bundled_file):
            environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            loaded_template = environment.get_template(name)
        elif os.path.isfile(template):
            directory, filename = os.path.split(template)
            environment = Environment(loader=FileSystemLoader(directory))
            loaded_template = environment.get_template(filename)
        else:
            raise TemplateNotFoundError(f"Template not found: {template}")

        self._env = environment
        self._jinja_template = loaded_template

    def _render_template(self, **kwargs: Any) -> str:
        if self._jinja_template is None:
            raise TemplateNotFoundError("No template loaded")
        return self._jinja_template.render(**kwargs).strip()

    def _build_schema_instruction(
        self, output_schema: Optional[Type[BaseModel]]
    ) -> str:
        if output_schema is None:
            return ""

        schema = output_schema.model_json_schema()
        fields: List[str] = []

        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "")
            suffix = f" ({description})" if description else ""
            fields.append(f"  - {field_name}: {field_type}{suffix}")

        return (
            "\n\nRespond with valid JSON matching this schema:\n{\n"
            + "\n".join(fields)
            + "\n}"
        )

    def build(
        self,
        instruction: str,
        text_input: str,
        domain: Optional[str] = None,
        labels: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        schema_instruction = self._build_schema_instruction(output_schema)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": instruction + schema_instruction}
        ]

        if self._jinja_template is not None:
            variables: Dict[str, Any] = {
                "text_input": text_input,
                "domain": domain,
                "labels": labels,
                "examples": examples,
                "description": kwargs.get("description"),
                **kwargs,
            }
            rendered = self._render_template(**variables)
            messages.append({"role": "user", "content": rendered})
            return messages

        if examples:
            for example_input, example_output in examples:
                messages.append({"role": "user", "content": example_input})
                messages.append({"role": "assistant", "content": example_output})

        parts: List[str] = []
        if domain:
            parts.append(f"Domain: {domain}")
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        parts.append(text_input)

        messages.append({"role": "user", "content": "\n".join(parts)})
        return messages