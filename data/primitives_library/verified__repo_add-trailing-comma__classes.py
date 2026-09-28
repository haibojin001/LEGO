from __future__ import annotations

import ast
from collections.abc import Iterable
from functools import partial

from tokenize_rt import Offset
from tokenize_rt import Token

from add_trailing_comma._ast_helpers import ast_to_offset
from add_trailing_comma._data import State
from add_trailing_comma._data import TokenFunc
from add_trailing_comma._data import register
from add_trailing_comma._token_helpers import find_call
from add_trailing_comma._token_helpers import fix_brace


def _fix_class(
        i: int,
        tokens: list[Token],
        *,
        arg_offsets: set[Offset],
) -> None:
    opening = find_call(arg_offsets, i, tokens)
    fix_brace(tokens, opening, add_comma=True, remove_comma=True)


@register(ast.ClassDef)
def visit_ClassDef(
        state: State,
        node: ast.ClassDef,
) -> Iterable[tuple[Offset, TokenFunc]]:
    arguments: list[ast.expr | ast.keyword] = list(node.bases)
    arguments.extend(node.keywords)
    offsets = {ast_to_offset(argument) for argument in arguments}

    if offsets:
        yield ast_to_offset(node), partial(_fix_class, arg_offsets=offsets)