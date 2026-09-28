from __future__ import annotations

from base64 import decodebytes, encodebytes


_multiline_outputs = ["text", "html", "svg", "latex", "javascript", "json"]


def restore_bytes(nb):
    """Convert stored image strings back to ASCII bytes."""
    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.outputs:
                if "png" in output:
                    output.png = output.png.encode("ascii", "replace")
                if "jpeg" in output:
                    output.jpeg = output.jpeg.encode("ascii", "replace")
    return nb


def _join_lines(lines):
    """Turn a sequence produced by splitlines back into a string."""
    if lines and lines[0].endswith(("\n", "\r")):
        return "".join(lines)
    return "\n".join(lines)


def rejoin_lines(nb):
    """Restore multiline notebook fields from lists of lines."""
    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type == "code":
                if "input" in cell and isinstance(cell.input, list):
                    cell.input = _join_lines(cell.input)
                for output in cell.outputs:
                    for key in _multiline_outputs:
                        value = output.get(key, None)
                        if isinstance(value, list):
                            output[key] = _join_lines(value)
            else:
                for key in ("source", "rendered"):
                    value = cell.get(key, None)
                    if isinstance(value, list):
                        cell[key] = _join_lines(value)
    return nb


def split_lines(nb):
    """Represent multiline notebook fields as lists of lines."""
    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type == "code":
                if "input" in cell and isinstance(cell.input, str):
                    cell.input = cell.input.splitlines(True)
                for output in cell.outputs:
                    for key in _multiline_outputs:
                        value = output.get(key, None)
                        if isinstance(value, str):
                            output[key] = value.splitlines(True)
            else:
                for key in ("source", "rendered"):
                    value = cell.get(key, None)
                    if isinstance(value, str):
                        cell[key] = value.splitlines(True)
    return nb


def base64_decode(nb):
    """Decode base64 image fields in a notebook."""
    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.outputs:
                if "png" in output:
                    if isinstance(output.png, str):
                        output.png = output.png.encode("ascii")
                    output.png = decodebytes(output.png)
                if "jpeg" in output:
                    if isinstance(output.jpeg, str):
                        output.jpeg = output.jpeg.encode("ascii")
                    output.jpeg = decodebytes(output.jpeg)
    return nb


def base64_encode(nb):
    """Encode binary image fields as base64 strings."""
    for worksheet in nb.worksheets:
        for cell in worksheet.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.outputs:
                if "png" in output:
                    output.png = encodebytes(output.png).decode("ascii")
                if "jpeg" in output:
                    output.jpeg = encodebytes(output.jpeg).decode("ascii")
    return nb


def strip_transient(nb):
    """Remove notebook data that is not intended for persistence."""
    nb.pop("orig_nbformat", None)
    nb.pop("orig_nbformat_minor", None)
    for worksheet in nb["worksheets"]:
        for cell in worksheet["cells"]:
            cell.get("metadata", {}).pop("trusted", None)
            cell.pop("trusted", None)
    return nb


class NotebookReader:
    """Base class for notebook readers."""

    def reads(self, s, **kwargs):
        """Read a notebook from a string."""
        raise NotImplementedError("loads must be implemented in a subclass")

    def read(self, fp, **kwargs):
        """Read a notebook from a file-like object."""
        return self.reads(fp.read(), **kwargs)


class NotebookWriter:
    """Base class for notebook writers."""

    def writes(self, nb, **kwargs):
        """Write a notebook to a string."""
        raise NotImplementedError("loads must be implemented in a subclass")

    def write(self, nb, fp, **kwargs):
        """Write a notebook to a file-like object."""
        return fp.write(self.writes(nb, **kwargs))