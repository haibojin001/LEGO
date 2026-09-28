from __future__ import annotations

import ast
from collections.abc import Iterable

from tokenize_rt import Offset, Token

from add_trailing_comma._ast_helpers import ast_to_offset
from add_trailing_comma._data import State, TokenFunc, register
from add_trailing_comma._token_helpers import Fix, find_simple, fix_brace


def _find_import(i: int, tokens: list[Token]) -> Fix | None:
    position = i
    while position < len(tokens):
        current = tokens[position]
        if current.name == 'NEWLINE':
            return None
        if current.name == 'OP' and current.src == '(':
            return find_simple(position, tokens)
        position += 1
    raise AssertionError('Past end?')


def _fix_import(i: int, tokens: list[Token]) -> None:
    fix = _find_import(i, tokens)
    fix_brace(tokens, fix, add_comma=True, remove_comma=True)


@register(ast.ImportFrom)
def visit_ImportFrom(
        state: State,
        node: ast.ImportFrom,
) -> Iterable[tuple[Offset, TokenFunc]]:
    yield ast_to_offset(node), _fix_import