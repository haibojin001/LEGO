from __future__ import annotations

import ast
import re

from .nbbase import (
    nbformat,
    nbformat_minor,
    new_code_cell,
    new_heading_cell,
    new_notebook,
    new_text_cell,
    new_worksheet,
)
from .rwbase import NotebookReader, NotebookWriter


_encoding_declaration_re = re.compile(r"^#.*coding[:=]\s*([-\w.]+)")


class PyReaderError(Exception):
    """Exception raised when Python notebook input cannot be handled."""


class PyReader(NotebookReader):
    """Reader for the legacy Python notebook representation."""

    def reads(self, s, **kwargs):
        """Convert Python notebook text into a notebook object."""
        return self.to_notebook(s, **kwargs)

    def to_notebook(self, s, **kwargs):
        """Convert Python notebook text into a notebook object."""
        cells = []
        buffered = []
        state = "codecell"
        cell_kwargs = {}

        for line in s.splitlines():
            if line.startswith("# <nbformat>") or _encoding_declaration_re.match(line):
                continue

            if line.startswith("# <codecell>"):
                cell = self.new_cell(state, buffered, **cell_kwargs)
                if cell is not None:
                    cells.append(cell)
                state = "codecell"
                buffered = []
                cell_kwargs = {}
                continue

            if line.startswith("# <htmlcell>"):
                cell = self.new_cell(state, buffered, **cell_kwargs)
                if cell is not None:
                    cells.append(cell)
                state = "htmlcell"
                buffered = []
                cell_kwargs = {}
                continue

            if line.startswith("# <markdowncell>"):
                cell = self.new_cell(state, buffered, **cell_kwargs)
                if cell is not None:
                    cells.append(cell)
                state = "markdowncell"
                buffered = []
                cell_kwargs = {}
                continue

            if line.startswith(("# <rawcell>", "# <plaintextcell>")):
                cell = self.new_cell(state, buffered, **cell_kwargs)
                if cell is not None:
                    cells.append(cell)
                state = "rawcell"
                buffered = []
                cell_kwargs = {}
                continue

            if line.startswith("# <headingcell"):
                cell = self.new_cell(state, buffered, **cell_kwargs)
                if cell is not None:
                    cells.append(cell)
                    buffered = []

                match = re.match(r"# <headingcell level=(?P<level>\d)>", line)
                if match is None:
                    state = "codecell"
                    cell_kwargs = {}
                    buffered = []
                else:
                    state = "headingcell"
                    cell_kwargs = {"level": int(match.group("level"))}
                continue

            buffered.append(line)

        if buffered and state == "codecell":
            cell = self.new_cell(state, buffered)
            if cell is not None:
                cells.append(cell)

        worksheet = new_worksheet(cells=cells)
        return new_notebook(worksheets=[worksheet])

    def new_cell(self, state, lines, **kwargs):
        """Build a notebook cell from the currently accumulated lines."""
        if state == "codecell":
            value = "\n".join(lines).strip("\n")
            if value:
                return new_code_cell(input=value)
            return None

        if state in ("htmlcell", "markdowncell", "rawcell"):
            value = self._remove_comments(lines)
            if value:
                kind = {
                    "htmlcell": "html",
                    "markdowncell": "markdown",
                    "rawcell": "raw",
                }[state]
                return new_text_cell(kind, source=value)
            return None

        if state == "headingcell":
            value = self._remove_comments(lines)
            if value:
                return new_heading_cell(
                    source=value,
                    level=kwargs.get("level", 1),
                )

        return None

    def _remove_comments(self, lines):
        result = []
        for line in lines:
            if line.startswith("#"):
                result.append(line[2:])
            else:
                result.append(line)
        return "\n".join(result).strip("\n")

    def split_lines_into_blocks(self, lines):
        """Yield source blocks corresponding to top-level Python statements."""
        if len(lines) == 1:
            yield lines[0]
            raise StopIteration()

        tree = ast.parse("\n".join(lines))
        positions = [node.lineno - 1 for node in tree.body]

        for index in range(len(positions) - 1):
            yield "\n".join(lines[positions[index] : positions[index + 1]]).strip("\n")

        yield "\n".join(lines[positions[-1] :]).strip("\n")


class PyWriter(NotebookWriter):
    """Writer for the legacy Python notebook representation."""

    def writes(self, nb, **kwargs):
        """Convert a notebook object to Python notebook text."""
        output = [
            "# -*- coding: utf-8 -*-",
            "# <nbformat>%i.%i</nbformat>" % (nbformat, nbformat_minor),
            "",
        ]

        for worksheet in nb.worksheets:
            for cell in worksheet.cells:
                cell_type = cell.cell_type

                if cell_type == "code":
                    source = cell.get("input")
                    if source is not None:
                        output.extend(("# <codecell>", ""))
                        output.extend(source.splitlines())
                        output.append("")

                elif cell_type in ("html", "markdown", "raw"):
                    source = cell.get("source")
                    if source is not None:
                        output.extend(("# <%scell>" % cell_type, ""))
                        output.extend("# " + line for line in source.splitlines())
                        output.append("")

                elif cell_type == "heading":
                    source = cell.get("source")
                    if source is not None:
                        output.extend(
                            (
                                "# <headingcell level=%s>" % cell.get("level", 1),
                                "",
                            )
                        )
                        output.extend("# " + line for line in source.splitlines())
                        output.append("")

        output.append("")
        return "\n".join(output)


_reader = PyReader()
_writer = PyWriter()

reads = _reader.reads
read = _reader.read
to_notebook = _reader.to_notebook
write = _writer.write
writes = _writer.writes