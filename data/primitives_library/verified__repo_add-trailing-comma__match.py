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
from add_trailing_comma._token_helpers import find_simple
from add_trailing_comma._token_helpers import fix_brace


def _fix_match_class(
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


@register(ast.MatchClass)
def visit_MatchClass(
        state: State,
        node: ast.MatchClass,
) -> Iterable[tuple[Offset, TokenFunc]]:
    offsets = {ast_to_offset(pattern) for pattern in node.patterns}
    offsets.update(ast_to_offset(pattern) for pattern in node.kwd_patterns)

    if offsets:
        yield (
            ast_to_offset(node),
            functools.partial(_fix_match_class, arg_offsets=offsets),
        )


def _fix_mapping(i: int, tokens: list[Token]) -> None:
    fix_brace(
        tokens,
        find_simple(i, tokens),
        add_comma=True,
        remove_comma=True,
    )


def _fix_sequence(i: int, tokens: list[Token], *, n: int) -> None:
    if tokens[i].src not in ('[', '('):
        return

    fix_brace(
        tokens,
        find_simple(i, tokens),
        add_comma=True,
        remove_comma=tokens[i].src == '[' or n > 1,
    )


@register(ast.MatchMapping)
def visit_MatchMapping(
        state: State,
        node: ast.MatchMapping,
) -> Iterable[tuple[Offset, TokenFunc]]:
    yield ast_to_offset(node), _fix_mapping


@register(ast.MatchSequence)
def visit_MatchSequence(
        state: State,
        node: ast.MatchSequence,
) -> Iterable[tuple[Offset, TokenFunc]]:
    yield (
        ast_to_offset(node),
        functools.partial(_fix_sequence, n=len(node.patterns)),
    )