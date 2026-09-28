from __future__ import annotations

import ast
import io
import operator
import os
import tokenize
from collections import UserList, UserString
from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ContextManager, Protocol, SupportsIndex, TypeVar, Union, cast

from refactor import common


DEFAULT_ENCODING = "utf-8"

AnyStringType = Union[str, "SourceSegment"]
StringType = TypeVar("StringType", bound=AnyStringType)


@dataclass
class Lines(UserList[StringType]):
    lines: list[StringType]

    def __post_init__(self) -> None:
        super().__init__(self.lines)
        self.lines = self.data

    def join(self) -> str:
        return "".join(str(line) for line in self.lines)

    def apply_indentation(
        self,
        indentation: StringType,
        *,
        start_prefix: AnyStringType = "",
        end_suffix: AnyStringType = "",
    ) -> None:
        for position, line in enumerate(self.data):
            if position == 0:
                self.data[position] = indentation + str(start_prefix) + str(line)  # type: ignore[operator]
            else:
                self.data[position] = indentation + line  # type: ignore[operator]

        if self.data:
            self.data[-1] += str(end_suffix)  # type: ignore[operator]

    @cached_property
    def _newline_type(self) -> str:
        if self.lines[-1].endswith(os.linesep):
            return os.linesep
        return "\n"


@dataclass
class SourceSegment(UserString):
    data: str
    encoding: str = DEFAULT_ENCODING

    def __getitem__(self, index: SupportsIndex | slice) -> SourceSegment:
        encoded = self.encode(encoding=self.encoding)

        if isinstance(index, slice):
            result = encoded[index].decode(encoding=self.encoding)
        else:
            offset = operator.index(index)
            result = encoded[offset : offset + 1].decode(encoding=self.encoding)
            if not result:
                raise IndexError("index out of range")

        return SourceSegment(result, encoding=self.encoding)


def split_lines(source: str, *, encoding: str | None = None) -> Lines:
    parts = source.splitlines(keepends=True)

    if encoding is not None:
        parts = [SourceSegment(part, encoding=encoding) for part in parts]  # type: ignore[assignment]

    return Lines(parts)


class Unparser(Protocol):
    def __init__(self, source: str, *args: Any, **kwargs: Any) -> None:
        ...

    def unparse(self, node: ast.AST) -> str:
        ...


class BaseUnparser(ast._Unparser):  # type: ignore[attr-defined]
    source: str | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.source = kwargs.pop("source", None)
        super().__init__(*args, **kwargs)

    def unparse(self, node: ast.AST) -> str:
        return self.visit(node)

    @cached_property
    def tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        stream = io.StringIO(self.source)
        return tuple(tokenize.generate_tokens(stream.readline))

    @contextmanager
    def indented(self) -> Generator[None, None, None]:
        self._indent += 1
        yield
        self._indent -= 1


class PreciseUnparser(BaseUnparser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._visited_comment_lines: set[int] = set()
        super().__init__(*args, **kwargs)

    def traverse(self, node: list[ast.AST] | ast.AST) -> None:
        if isinstance(node, list) or self.source is None:
            return super().traverse(node)

        if not self.maybe_retrieve(node):
            super().traverse(node)

    def maybe_retrieve(self, node: ast.AST) -> bool:
        if self.source is None:
            return False

        if not isinstance(node, (ast.expr, ast.stmt)):
            return False

        segment = common.get_source_segment(self.source, node)
        if segment is None:
            return False

        candidate = common.wrap_with_parens(segment) if isinstance(node, ast.expr) else segment

        try:
            parsed = ast.parse(candidate)
        except SyntaxError:
            return False

        if len(parsed.body) != 1:
            return False

        recovered = parsed.body[0]

        if isinstance(node, ast.expr) and isinstance(recovered, ast.Expr):
            recovered = recovered.value

        same = common.compare_ast(recovered, node)
        if same:
            self.retrieve_segment(node, segment)

        return same

    @contextmanager
    def _collect_stmt_comments(self, node: ast.AST) -> Iterator[None]:
        def emit_comment(line_number: int, line: str, column: int) -> None:
            if line_number in self._visited_comment_lines:
                return

            self.fill()
            self.write(line[column:])
            self._visited_comment_lines.add(line_number)

        assert self.source is not None

        source_lines = self.source.splitlines()
        first_line = node.lineno - 1
        last_line = cast(int, node.end_lineno)

        prior: list[tuple[int, str, int]] = []
        for offset, line in enumerate(reversed(source_lines[:first_line])):
            column = line.find("#")
            if column == -1 or column != node.col_offset:
                break
            prior.append((first_line - offset, line, column))

        for comment in reversed(prior):
            emit_comment(*comment)

        yield

        for offset, line in enumerate(source_lines[last_line:], 1):
            column = line.find("#")
            if column == -1 or column != node.col_offset:
                break
            emit_comment(last_line + offset, line, column)

    def collect_comments(self, node: ast.AST) -> ContextManager[None]:
        if isinstance(node, ast.stmt):
            return self._collect_stmt_comments(node)
        return nullcontext()

    def retrieve_segment(self, node: ast.AST, segment: str) -> None:
        with self.collect_comments(node):
            if isinstance(node, ast.stmt):
                self.fill()
            self.write(segment)


UNPARSER_BACKENDS = {
    "fast": BaseUnparser,
    "precise": PreciseUnparser,
}