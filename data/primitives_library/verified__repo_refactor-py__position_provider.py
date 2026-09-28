from __future__ import annotations

import ast
import tokenize
from collections.abc import Iterator
from functools import singledispatch
from itertools import chain
from typing import Callable

from refactor import common
from refactor.ast import Lines, split_lines
from refactor.context import Context


def _line_wrapper(lines: Lines) -> Callable[[], str]:
    iterator = iter(lines)

    def readline() -> str:
        return next(iterator, "")

    return readline


_SPACE_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
)


def _ignore_space(
    token_iterator: Iterator[tokenize.TokenInfo],
) -> Iterator[tokenize.TokenInfo]:
    for token in token_iterator:
        if token.type not in _SPACE_TOKENS:
            yield token


@singledispatch
def infer_identifier_position(
    node: ast.AST,
    identifier_value: str,
    context: Context,
) -> common.PositionType | None:
    return None


EXPECTED_KEYWORDS = {
    ast.FunctionDef: ["def"],
    ast.AsyncFunctionDef: ["async", "def"],
    ast.ClassDef: ["class"],
}


@infer_identifier_position.register(ast.ClassDef)
@infer_identifier_position.register(ast.FunctionDef)
@infer_identifier_position.register(ast.AsyncFunctionDef)
def infer_definition_name(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    identifier_value: str,
    context: Context,
) -> common.PositionType | None:
    segment = common.get_source_segment(context.source, node)
    if segment is None:
        return None

    token_stream = _ignore_space(
        tokenize.generate_tokens(_line_wrapper(split_lines(segment)))
    )

    def get_token() -> tokenize.TokenInfo | None:
        try:
            return next(token_stream, None)
        except (SyntaxError, tokenize.TokenError):
            return None

    def matches(
        token_type: int, value: str
    ) -> tokenize.TokenInfo | None:
        token = get_token()
        if (
            token is not None
            and token.exact_type == token_type
            and token.string == value
        ):
            return token
        return None

    found: tokenize.TokenInfo | None = None
    for expected in chain(EXPECTED_KEYWORDS[type(node)], (identifier_value,)):
        found = matches(tokenize.NAME, expected)
        if found is None:
            return None

    start_line = node.lineno - 1
    start_column = node.col_offset
    return (
        start_line + found.start[0],
        start_column + found.start[1],
        start_line + found.end[0],
        start_column + found.end[1],
    )