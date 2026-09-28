from __future__ import annotations

import ast
from collections.abc import Iterable

from tokenize_rt import Offset
from tokenize_rt import Token

from add_trailing_comma._ast_helpers import ast_to_offset
from add_trailing_comma._data import State
from add_trailing_comma._data import TokenFunc
from add_trailing_comma._data import register
from add_trailing_comma._token_helpers import find_simple
from add_trailing_comma._token_helpers import fix_brace


def _fix_with(i: int, tokens: list[Token]) -> None:
    token_index = i + 1

    if tokens[token_index].name == 'UNIMPORTANT_WS':
        token_index += 1

    if tokens[token_index].src != '(':
        return

    result = find_simple(token_index, tokens)
    if result is None:
        return

    if tokens[result.braces[-1] + 1].src == ':':
        fix_brace(tokens, result, add_comma=True, remove_comma=True)


@register(ast.With)
def visit_With(
    state: State,
    node: ast.With,
) -> Iterable[tuple[Offset, TokenFunc]]:
    yield ast_to_offset(node), _fix_with