from __future__ import annotations

import ast
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, partial
from typing import Any, cast

_MISSING = object()

_CONSTANT_FIELDS: dict[type[ast.AST], list[str]] = {
    ast.Constant: ["value"],
}

if hasattr(ast, "MatchSingleton"):
    _CONSTANT_FIELDS[ast.MatchSingleton] = ["value"]


class _Continue(Exception):
    pass


class IncompleteASTError(Exception):
    pass


class ChangeType(Enum):
    FULL = auto()
    FIELD_ADDITION = auto()
    FIELD_REMOVAL = auto()
    ITEM_VALUE = auto()
    FIELD_VALUE = auto()
    FIELD_SIZE = auto()
    UNINFERRABLE = auto()


@dataclass
class ChangeSet:
    change_type: ChangeType
    original_node: Any
    new_node: Any
    on_field: str | None = None
    on_index: int | None = None


@cache
def _is_constant(node_type: type[ast.AST], field: str) -> bool:
    return field in _CONSTANT_FIELDS.get(node_type, [])


def _incomplete_if(condition: bool) -> None:
    if condition:
        raise IncompleteASTError


def _change_if(condition: bool, *args: Any, **kwargs: Any) -> Iterator[ChangeSet]:
    if condition:
        yield ChangeSet(*args, **kwargs)
        raise _Continue


def ast_delta(baseline: ast.AST, new_node: ast.AST) -> Iterator[ChangeSet]:
    node_type = type(baseline)

    if node_type is not type(new_node):
        yield ChangeSet(ChangeType.FULL, baseline, new_node)
        return

    for field_name in node_type._fields:
        with suppress(_Continue):
            old_value = getattr(baseline, field_name)
            replacement_value = getattr(new_node, field_name, _MISSING)

            _incomplete_if(replacement_value is _MISSING)

            report = partial(
                _change_if,
                original_node=baseline,
                new_node=new_node,
                on_field=field_name,
            )

            if not _is_constant(node_type, field_name):
                if old_value is None:
                    yield from report(
                        replacement_value is not None,
                        ChangeType.FIELD_ADDITION,
                    )
                else:
                    yield from report(
                        replacement_value is None,
                        ChangeType.FIELD_REMOVAL,
                    )

            if isinstance(old_value, ast.AST):
                _incomplete_if(not isinstance(replacement_value, ast.AST))
                yield from ast_delta(old_value, cast(ast.AST, replacement_value))
            elif isinstance(old_value, list):
                _incomplete_if(not isinstance(replacement_value, list))
                yield from _ast_sequence_delta(baseline, new_node, field_name)
            else:
                yield from report(
                    old_value != replacement_value,
                    ChangeType.FIELD_VALUE,
                )


def _ast_sequence_delta(
    baseline: ast.AST,
    new_node: ast.AST,
    field: str,
) -> Iterator[ChangeSet]:
    baseline_items: list[Any] = getattr(baseline, field)
    new_items: list[Any] = getattr(new_node, field)

    if len(baseline_items) != len(new_items):
        yield ChangeSet(
            ChangeType.FIELD_SIZE,
            baseline,
            new_node,
            on_field=field,
        )
        return

    for index, (old_item, replacement_item) in enumerate(
        zip(baseline_items, new_items)
    ):
        report = partial(
            _change_if,
            original_node=baseline,
            new_node=new_node,
            on_field=field,
            on_index=index,
        )

        with suppress(_Continue):
            if isinstance(old_item, ast.AST) or old_item is None:
                if old_item is None or replacement_item is None:
                    yield from report(
                        replacement_item is not old_item,
                        ChangeType.FULL,
                    )
                else:
                    _incomplete_if(not isinstance(replacement_item, ast.AST))
                    yield from ast_delta(old_item, replacement_item)
            else:
                yield from report(
                    replacement_item != old_item,
                    ChangeType.ITEM_VALUE,
                )