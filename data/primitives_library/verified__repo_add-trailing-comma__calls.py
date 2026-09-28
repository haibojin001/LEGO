from __future__ import annotations

import ast
import functools
from collections.abc import Iterable

from tokenize_rt import Offset
from tokenize_rt import Token

from add_trailing_comma._ast_helpers import ast_to_offset
from add_trailing_comma._data import State
from add_trailing_comma._data import TokenFunc
from add_trailing_comma._data import register
from add_trailing_comma._token_helpers import find_call
from add_trailing_comma._token_helpers import fix_brace


def _fix_call(
        i: int,
        tokens: list[Token],
        *,
        arg_offsets: set[Offset],
) -> None:
    opening = find_call(arg_offsets, i, tokens)
    fix_brace(
        tokens,
        opening,
        add_comma=True,
        remove_comma=True,
    )


@register(ast.Call)
def visit_Call(
        state: State,
        node: ast.Call,
) -> Iterable[tuple[Offset, TokenFunc]]:
    arguments: list[ast.expr | ast.keyword] = [
        *node.args,
        *node.keywords,
    ]
    offsets: set[Offset] = set()

    for argument in arguments:
        position = ast_to_offset(argument)
        if position.utf8_byte_offset != -1:
            offsets.add(position)

    has_single_generator = (
        len(arguments) == 1 and
        isinstance(arguments[0], ast.GeneratorExp)
    )

    if offsets and not has_single_generator and not state.in_fstring:
        callback = functools.partial(_fix_call, arg_offsets=offsets)
        yield ast_to_offset(node), callback