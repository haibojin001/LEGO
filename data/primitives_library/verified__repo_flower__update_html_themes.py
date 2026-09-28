import json
import re
from pathlib import Path
from typing import Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = REPO_ROOT
MERGE_THEME_VARIABLE_FIELDS = {"light_css_variables", "dark_css_variables"}

NEW_FIELDS: dict[str, Optional[Union[dict[str, str], str]]] = {
    "light_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff",
    },
}

with (REPO_ROOT / "dev" / "docs-ui-config.yml").open(encoding="utf-8") as file:
    announcement = yaml.safe_load(file)["announcement"]
    if announcement["enabled"]:
        NEW_FIELDS["announcement"] = announcement["html"]


def dict_to_fields_str(fields: dict[str, Optional[Union[dict[str, str], str]]]) -> str:
    """Format fields for use inside a Python dictionary literal."""
    if not fields:
        return ""

    rendered = json.dumps(fields, indent=4, ensure_ascii=False)
    lines = rendered.splitlines()
    if len(lines) >= 2 and lines[0].strip() == "{" and lines[-1].strip() == "}":
        return "\n".join(
            item[4:] if item.startswith("    ") else item for item in lines[1:-1]
        )
    return rendered


def find_conf_files(root_dir: Path) -> list[Path]:
    """Find documentation configuration files below root_dir."""
    return list(root_dir.rglob("conf.py"))


def _dict_entry_str(key: str, value: str) -> str:
    """Render a single dictionary member."""
    return f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)},"


def _brace_delta(line: str) -> int:
    """Calculate the net brace count change in line."""
    return line.count("{") - line.count("}")


def _copy_fields(
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> dict[str, Optional[Union[dict[str, str], str]]]:
    """Make a per-update copy of fields."""
    copied: dict[str, Optional[Union[dict[str, str], str]]] = {}
    for key, value in fields.items():
        copied[key] = value.copy() if isinstance(value, dict) else value
    return copied


def _merge_fields(
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> dict[str, dict[str, str]]:
    """Extract variable fields eligible for merging."""
    result: dict[str, dict[str, str]] = {}
    for key, value in fields.items():
        if key in MERGE_THEME_VARIABLE_FIELDS and isinstance(value, dict):
            result[key] = value.copy()
    return result


def _append_indented_fields(
    updated_content: list[str],
    line: str,
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> bool:
    """Place remaining new fields immediately before an options closing brace."""
    match = re.match(r"^(\s*)}", line)
    indentation = match.group(1) if match else ""

    field_text = dict_to_fields_str(fields)
    inserted = "\n".join(
        indentation + "    " + field_line for field_line in field_text.splitlines()
    )
    if inserted:
        updated_content.insert(-1, inserted + ",")
    return bool(inserted)


def _split_inline_comment(line: str) -> tuple[str, str]:
    """Separate a comment from a line without treating string contents as comments."""
    in_string = False
    quote = ""
    escaped = False

    for index, character in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
        elif character in {"'", '"'}:
            in_string = True
            quote = character
        elif character == "#":
            return line[:index], line[index:]

    return line, ""


def _ensure_previous_entry_has_comma(updated_content: list[str]) -> None:
    """Add a trailing comma to the preceding dictionary entry where necessary."""
    for index in range(len(updated_content) - 1, -1, -1):
        stripped = updated_content[index].strip()
        if not stripped or stripped.startswith("#"):
            continue

        code, comment = _split_inline_comment(updated_content[index])
        if code.rstrip().endswith((",", "{")):
            return

        updated_content[index] = code.rstrip() + (f", {comment}" if comment else ",")
        return


def _process_merge_field_line(
    updated_content: list[str],
    line: str,
    brace_depth: int,
    merge_field: str,
    merge_fields: dict[str, dict[str, str]],
) -> tuple[int, Optional[str], bool]:
    """Process a line belonging to an already present CSS variable mapping."""
    next_merge_field: Optional[str] = merge_field
    variable_match = re.match(r'^(\s*)"([^"]+)"\s*:', line)
    merge_variables = merge_fields[merge_field]

    if variable_match:
        variable_name = variable_match.group(2)
        if variable_name in merge_variables:
            updated_content.append(line)
            del merge_variables[variable_name]
            return brace_depth, next_merge_field, False

    brace_depth += _brace_delta(line)
    if brace_depth == 1:
        indent_match = re.match(r"^(\s*)}", line)
        indentation = indent_match.group(1) if indent_match else ""
        if merge_variables:
            _ensure_previous_entry_has_comma(updated_content)
        for key, value in merge_variables.items():
            updated_content.append(indentation + "    " + _dict_entry_str(key, value))
        next_merge_field = None

    updated_content.append(line)
    return brace_depth, next_merge_field, bool(merge_variables)


def update_conf_file(
    file_path: Path,
    new_fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> None:
    """
    Insert new_fields into the html_theme_options block of file_path.

    Existing light_css_variables and dark_css_variables dictionaries are merged
    rather than replaced.
    """
    if not dict_to_fields_str(new_fields).strip():
        print(f"Skipping {file_path} (no new fields to insert)")
        return

    updated_content: list[str] = []
    fields_to_append = _copy_fields(new_fields)
    merge_fields = _merge_fields(new_fields)
    inside_options = False
    brace_depth = 0
    merge_field: Optional[str] = None
    found_options = False
    modified = False

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if merge_field:
            brace_depth, merge_field, line_modified = _process_merge_field_line(
                updated_content,
                line,
                brace_depth,
                merge_field,
                merge_fields,
            )
            modified |= line_modified
            continue

        updated_content.append(line)

        if re.match(r"^\s*html_theme_options\s*=\s*{", line):
            inside_options = True
            found_options = True

        top_level_match = re.match(r'^\s*"(?P<key>[^"]+)"\s*:', line)
        if inside_options and brace_depth == 1 and top_level_match:
            theme_variable_key = top_level_match.group("key")
            fields_to_append.pop(theme_variable_key, None)

            if theme_variable_key in merge_fields:
                variable_match = re.match(
                    r'^\s*"(?P<key>light_css_variables|dark_css_variables)"\s*:\s*{',
                    line,
                )
                if variable_match:
                    merge_field = theme_variable_key

        if inside_options:
            brace_depth += _brace_delta(line)

        if inside_options and brace_depth == 0:
            modified |= _append_indented_fields(updated_content, line, fields_to_append)
            inside_options = False

    if modified:
        file_path.write_text("\n".join(updated_content) + "\n", encoding="utf-8")
        print(f"Updated: {file_path}")
    elif found_options:
        print(f"No changes needed in: {file_path}")
    else:
        print(f"No html_theme_options found in: {file_path}")


def main() -> None:
    """Update HTML theme options in all documentation configuration files."""
    for file_path in find_conf_files(ROOT_DIR):
        update_conf_file(file_path, NEW_FIELDS)


if __name__ == "__main__":
    main()