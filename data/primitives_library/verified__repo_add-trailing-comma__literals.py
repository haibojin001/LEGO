from __future__ import annotations

import ast
import functools
from collections.abc import Iterable

from tokenize_rt import NON_CODING_TOKENS
from tokenize_rt import Offset
from tokenize_rt import Token

from add_trailing_comma._ast_helpers import ast_to_offset
from add_trailing_comma._data import State
from add_trailing_comma._data import TokenFunc
from add_trailing_comma._data import register
from add_trailing_comma._token_helpers import Fix
from add_trailing_comma._token_helpers import find_simple
from add_trailing_comma._token_helpers import fix_brace


def _fix_literal(
        i: int,
        tokens: list[Token],
        *,
        one_el_tuple: bool,
) -> None:
    brace = find_simple(i, tokens)
    fix_brace(
        tokens,
        brace,
        add_comma=True,
        remove_comma=not one_el_tuple,
    )


@register(ast.Set)
def visit_Set(
        state: State,
        node: ast.Set,
) -> Iterable[tuple[Offset, TokenFunc]]:
    callback = functools.partial(_fix_literal, one_el_tuple=False)
    yield ast_to_offset(node), callback


@register(ast.List)
def visit_List(
        state: State,
        node: ast.List,
) -> Iterable[tuple[Offset, TokenFunc]]:
    if not node.elts:
        return

    callback = functools.partial(_fix_literal, one_el_tuple=False)
    yield ast_to_offset(node), callback


@register(ast.Dict)
def visit_Dict(
        state: State,
        node: ast.Dict,
) -> Iterable[tuple[Offset, TokenFunc]]:
    if not node.values:
        return

    callback = functools.partial(_fix_literal, one_el_tuple=False)
    yield ast_to_offset(node), callback


def _find_tuple(i: int, tokens: list[Token]) -> Fix | None:
    previous = i - 1
    while tokens[previous].name in NON_CODING_TOKENS:
        previous -= 1

    if tokens[previous].src not in ('(', '['):
        return None

    return find_simple(previous, tokens)


def _fix_tuple(
        i: int,
        tokens: list[Token],
        *,
        one_el_tuple: bool,
) -> None:
    brace = _find_tuple(i, tokens)
    fix_brace(
        tokens,
        brace,
        add_comma=True,
        remove_comma=not one_el_tuple,
    )


def _fix_tuple_py38(
        i: int,
        tokens: list[Token],
        *,
        one_el_tuple: bool,
) -> None:
    brace = find_simple(i, tokens)
    if brace is None or not brace.multi_arg:
        return

    fix_brace(
        tokens,
        brace,
        add_comma=True,
        remove_comma=not one_el_tuple,
    )


@register(ast.Tuple)
def visit_Tuple(
        state: State,
        node: ast.Tuple,
) -> Iterable[tuple[Offset, TokenFunc]]:
    if not node.elts:
        return

    single_element = len(node.elts) == 1
    node_offset = ast_to_offset(node)
    if node_offset == ast_to_offset(node.elts[0]):
        callback = functools.partial(
            _fix_tuple,
            one_el_tuple=single_element,
        )
    else:
        callback = functools.partial(
            _fix_tuple_py38,
            one_el_tuple=single_element,
        )

    yield node_offset, callback