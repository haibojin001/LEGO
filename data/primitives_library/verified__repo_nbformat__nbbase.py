"""Basic dictionary-based helpers for version 3 notebooks."""

from __future__ import annotations

import warnings

from nbformat._struct import Struct

nbformat = 3
nbformat_minor = 0
nbformat_schema = {(3, 0): "nbformat.v3.schema.json"}


class NotebookNode(Struct):
    """A notebook node object."""


def from_dict(d):
    """Create notebook node(s) from an object."""
    if isinstance(d, dict):
        result = NotebookNode()
        for key, value in d.items():
            result[key] = from_dict(value)
        return result
    if isinstance(d, (tuple, list)):
        return [from_dict(value) for value in d]
    return d


def str_passthrough(obj):
    """Return a string unchanged, rejecting non-string values."""
    if not isinstance(obj, str):
        raise TypeError
    return obj


def cast_str(obj):
    """Cast an object as a string."""
    if isinstance(obj, bytes):
        warnings.warn(
            "A notebook got bytes instead of likely base64 encoded values."
            "The content will likely be corrupted.",
            UserWarning,
            stacklevel=3,
        )
        return obj.decode("ascii", "replace")
    if not isinstance(obj, str):
        raise TypeError
    return obj


def new_output(
    output_type,
    output_text=None,
    output_png=None,
    output_html=None,
    output_svg=None,
    output_latex=None,
    output_json=None,
    output_javascript=None,
    output_jpeg=None,
    prompt_number=None,
    ename=None,
    evalue=None,
    traceback=None,
    stream=None,
    metadata=None,
):
    """Create a new output for a code cell's outputs list."""
    output = NotebookNode()
    output.output_type = str(output_type)

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict")

    if output_type in {"pyout", "display_data"}:
        output.metadata = metadata

    if output_type != "pyerr":
        if output_text is not None:
            output.text = str_passthrough(output_text)
        if output_png is not None:
            output.png = cast_str(output_png)
        if output_jpeg is not None:
            output.jpeg = cast_str(output_jpeg)
        if output_html is not None:
            output.html = str_passthrough(output_html)
        if output_svg is not None:
            output.svg = str_passthrough(output_svg)
        if output_latex is not None:
            output.latex = str_passthrough(output_latex)
        if output_json is not None:
            output.json = str_passthrough(output_json)
        if output_javascript is not None:
            output.javascript = str_passthrough(output_javascript)

    if output_type == "pyout" and prompt_number is not None:
        output.prompt_number = int(prompt_number)

    if output_type == "pyerr":
        if ename is not None:
            output.ename = str_passthrough(ename)
        if evalue is not None:
            output.evalue = str_passthrough(evalue)
        if traceback is not None:
            output.traceback = [str_passthrough(frame) for frame in list(traceback)]

    if output_type == "stream":
        output.stream = "stdout" if stream is None else str_passthrough(stream)

    return output


def new_code_cell(
    input=None,
    prompt_number=None,
    outputs=None,
    language="python",
    collapsed=False,
    metadata=None,
):
    """Create a new code cell with input and output."""
    cell = NotebookNode()
    cell.cell_type = "code"

    if language is not None:
        cell.language = str_passthrough(language)
    if input is not None:
        cell.input = str_passthrough(input)
    if prompt_number is not None:
        cell.prompt_number = int(prompt_number)

    cell.outputs = [] if outputs is None else outputs

    if collapsed is not None:
        cell.collapsed = bool(collapsed)

    cell.metadata = NotebookNode(metadata or {})
    return cell


def new_text_cell(cell_type, source=None, rendered=None, metadata=None):
    """Create a new text cell."""
    cell = NotebookNode()

    if cell_type == "plaintext":
        cell_type = "raw"

    if source is not None:
        cell.source = str_passthrough(source)

    cell.metadata = NotebookNode(metadata or {})
    cell.cell_type = cell_type
    return cell


def new_heading_cell(source=None, level=1, rendered=None, metadata=None):
    """Create a new section cell with a given integer level."""
    cell = NotebookNode()
    cell.cell_type = "heading"

    if source is not None:
        cell.source = str_passthrough(source)

    cell.level = int(level)
    cell.metadata = NotebookNode(metadata or {})
    return cell


def new_worksheet(name=None, cells=None, metadata=None):
    """Create a worksheet by name with a list of cells."""
    worksheet = NotebookNode()
    worksheet.cells = [] if cells is None else list(cells)
    worksheet.metadata = NotebookNode(metadata or {})
    return worksheet


def new_notebook(name=None, metadata=None, worksheets=None):
    """Create a notebook by name, id and a list of worksheets."""
    notebook = NotebookNode()
    notebook.nbformat = nbformat
    notebook.nbformat_minor = nbformat_minor
    notebook.worksheets = [] if worksheets is None else list(worksheets)

    if metadata is None:
        notebook.metadata = new_metadata()
    else:
        notebook.metadata = NotebookNode(metadata)

    if name is not None:
        notebook.metadata.name = str_passthrough(name)

    return notebook


def new_metadata(
    name=None,
    authors=None,
    license=None,
    created=None,
    modified=None,
    gistid=None,
):
    """Create a new metadata node."""
    metadata = NotebookNode()

    if name is not None:
        metadata.name = str_passthrough(name)
    if authors is not None:
        metadata.authors = list(authors)
    if created is not None:
        metadata.created = str_passthrough(created)
    if modified is not None:
        metadata.modified = str_passthrough(modified)
    if license is not None:
        metadata.license = str_passthrough(license)
    if gistid is not None:
        metadata.gistid = str_passthrough(gistid)

    return metadata


def new_author(name=None, email=None, affiliation=None, url=None):
    """Create a new author."""
    author = NotebookNode()

    if name is not None:
        author.name = str_passthrough(name)
    if email is not None:
        author.email = str_passthrough(email)
    if affiliation is not None:
        author.affiliation = str_passthrough(affiliation)
    if url is not None:
        author.url = str_passthrough(url)

    return author