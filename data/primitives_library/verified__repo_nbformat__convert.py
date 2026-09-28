from __future__ import annotations

import json
import re

from traitlets.log import get_logger

from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode

from .nbbase import nbformat, nbformat_minor


def _warn_if_invalid(nb, version):
    from nbformat import ValidationError, validate

    try:
        validate(nb, version=version)
    except ValidationError as error:
        get_logger().error("Notebook JSON is not valid v%i: %s", version, error)


def upgrade(nb, from_version=None, from_minor=None):
    if not from_version:
        from_version = nb["nbformat"]

    if not from_minor:
        if "nbformat_minor" in nb:
            from_minor = nb["nbformat_minor"]
        elif from_version == 4:
            raise validator.ValidationError(
                "The v4 notebook does not include the nbformat minor, which is needed."
            )
        else:
            from_minor = 0

    if from_version == 3:
        _warn_if_invalid(nb, from_version)

        original_major = nb.pop("orig_nbformat", None)
        original_minor = nb.pop("orig_nbformat_minor", None)
        nb.metadata.orig_nbformat = original_major or 3
        nb.metadata.orig_nbformat_minor = original_minor or 0

        nb.nbformat = nbformat
        nb.nbformat_minor = nbformat_minor

        converted_cells = []
        nb["cells"] = converted_cells
        for worksheet in nb.pop("worksheets", []):
            for old_cell in worksheet["cells"]:
                converted_cells.append(upgrade_cell(old_cell))

        nb.metadata.pop("name", "")
        nb.metadata.pop("signature", "")

        _warn_if_invalid(nb, nbformat)
        return nb

    if from_version == 4:
        if from_minor == nbformat_minor:
            return nb

        if from_minor < 5:
            for cell in nb.cells:
                cell.id = random_cell_id()

        nb.metadata.orig_nbformat_minor = from_minor
        nb.nbformat_minor = nbformat_minor
        return nb

    raise ValueError(
        "Cannot convert a notebook directly from v%s to v4.  "
        "Try using the nbformat.convert module." % from_version
    )


def upgrade_cell(cell):
    cell.setdefault("metadata", NotebookNode())
    cell.id = random_cell_id()

    if cell.cell_type == "code":
        cell.pop("language", "")

        if "collapsed" in cell:
            cell.metadata["collapsed"] = cell.pop("collapsed")

        cell.source = cell.pop("input", "")
        cell.execution_count = cell.pop("prompt_number", None)
        cell.outputs = upgrade_outputs(cell.outputs)

    elif cell.cell_type == "heading":
        heading_level = cell.pop("level", 1)
        text = " ".join(cell.get("source", "").splitlines())
        cell.cell_type = "markdown"
        cell.source = "{} {}".format("#" * heading_level, text)

    elif cell.cell_type == "html":
        cell.cell_type = "markdown"

    return cell


def downgrade_cell(cell):
    if cell.cell_type == "code":
        cell.language = "python"
        cell.input = cell.pop("source", "")
        cell.prompt_number = cell.pop("execution_count", None)
        cell.collapsed = cell.metadata.pop("collapsed", False)
        cell.outputs = downgrade_outputs(cell.outputs)

    elif cell.cell_type == "markdown":
        source = cell.get("source", "")
        if "\n" not in source and source.startswith("#"):
            matched = re.match(r"(#+)\s*(.*)", source)
            assert matched is not None
            hashes, content = matched.groups()
            cell.cell_type = "heading"
            cell.source = content
            cell.level = len(hashes)

    cell.pop("id", None)
    cell.pop("attachments", None)
    return cell


_mime_map = {
    "text": "text/plain",
    "html": "text/html",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "latex": "text/latex",
    "json": "application/json",
    "javascript": "application/javascript",
}


def to_mime_key(d):
    for alias, mime_type in _mime_map.items():
        if alias in d:
            d[mime_type] = d.pop(alias)
    return d


def from_mime_key(d):
    aliases = {}
    for alias, mime_type in _mime_map.items():
        if mime_type in d:
            aliases[alias] = d[mime_type]
    return aliases


def upgrade_output(output):
    output_kind = output["output_type"]

    if output_kind in {"pyout", "display_data"}:
        output.setdefault("metadata", NotebookNode())

        if output_kind == "pyout":
            output["output_type"] = "execute_result"
            output["execution_count"] = output.pop("prompt_number", None)

        payload = {}
        protected = {"output_type", "execution_count", "metadata"}
        for field in list(output):
            if field not in protected:
                payload[field] = output.pop(field)

        to_mime_key(payload)
        output["data"] = payload
        to_mime_key(output.metadata)

        if "application/json" in payload:
            payload["application/json"] = json.loads(payload["application/json"])

        for image_type in ("image/png", "image/jpeg"):
            if image_type in payload and isinstance(payload[image_type], bytes):
                payload[image_type] = payload[image_type].decode("ascii")

    elif output_kind == "pyerr":
        output["output_type"] = "error"

    elif output_kind == "stream":
        output["name"] = output.pop("stream", "stdout")

    return output


def downgrade_output(output):
    output_kind = output["output_type"]

    if output_kind in {"execute_result", "display_data"}:
        if output_kind == "execute_result":
            output["output_type"] = "pyout"
            output["prompt_number"] = output.pop("execution_count", None)

        payload = output.pop("data", {})
        if "application/json" in payload:
            payload["application/json"] = json.dumps(payload["application/json"])

        output.update(from_mime_key(payload))
        from_mime_key(output.get("metadata", {}))

    elif output_kind == "error":
        output["output_type"] = "pyerr"

    elif output_kind == "stream":
        output["stream"] = output.pop("name")

    return output


def upgrade_outputs(outputs):
    return [upgrade_output(output) for output in outputs]


def downgrade_outputs(outputs):
    return [downgrade_output(output) for output in outputs]


def downgrade(nb):
    if nb.nbformat != nbformat:
        return nb

    _warn_if_invalid(nb, nbformat)

    original_major = nb.metadata.pop("orig_nbformat", None)
    original_minor = nb.metadata.pop("orig_nbformat_minor", None)
    nb.nbformat = original_major or 3
    nb.nbformat_minor = original_minor or 0

    cells = nb.pop("cells")
    nb.worksheets = [v3.new_worksheet(cells=cells)]

    for cell in cells:
        downgrade_cell(cell)

    nb.metadata.name = nb.metadata.pop("name", "")
    nb.metadata.signature = nb.metadata.pop("signature", "")

    _warn_if_invalid(nb, nb.nbformat)
    return nb