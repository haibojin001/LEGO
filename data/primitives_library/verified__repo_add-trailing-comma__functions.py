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


def _fix_func(
        i: int,
        tokens: list[Token],
        *,
        arg_offsets: set[Offset],
) -> None:
    fix_brace(
        tokens,
        find_call(arg_offsets, i, tokens),
        add_comma=True,
        remove_comma=True,
    )


def visit_FunctionDef(
        state: State,
        node: ast.AsyncFunctionDef | ast.FunctionDef,
) -> Iterable[tuple[Offset, TokenFunc]]:
    arguments = [*node.args.posonlyargs, *node.args.args]

    if node.args.vararg is not None:
        arguments.append(node.args.vararg)

    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)

    if node.args.kwonlyargs:
        arguments.extend(node.args.kwonlyargs)

    offsets = {ast_to_offset(argument) for argument in arguments}

    if offsets:
        yield ast_to_offset(node), functools.partial(
            _fix_func,
            arg_offsets=offsets,
        )


register(ast.AsyncFunctionDef)(visit_FunctionDef)
register(ast.FunctionDef)(visit_FunctionDef)