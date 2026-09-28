from __future__ import annotations

import ast
import copy
import warnings
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from refactor.ast import split_lines
from refactor.common import PositionType

if TYPE_CHECKING:
    from refactor.context import Context

K = TypeVar("K")
T = TypeVar("T")

__all__ = [
    "BaseAction",
    "InsertAfter",
    "InsertBefore",
    "LazyInsertAfter",
    "LazyInsertBefore",
    "LazyReplace",
    "Replace",
    "Erase",
    "EraseOrReplace",
    "InvalidActionError",
]


def _hint(name: str, value: str):
    def decorator(obj):
        setattr(obj, name, value)
        return obj

    return decorator


def clone(node: K) -> K:
    return copy.deepcopy(node)


def position_for(node: ast.AST) -> PositionType:
    try:
        return (
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )
    except AttributeError as error:
        raise ValueError("Node does not have source position information.") from error


def find_indent(source: str) -> tuple[str, str]:
    whitespace_length = len(source) - len(source.lstrip(" \t"))
    indentation = source[:whitespace_length]
    return indentation, source[whitespace_length:]


class InvalidActionError(ValueError):
    """An improper usage of an action."""


class BaseAction:
    """A source code transformation action."""

    def apply(self, context: Context, source: str) -> str:
        raise NotImplementedError

    def _stack_effect(self) -> tuple[ast.AST, int]:
        raise NotImplementedError("This action can't be chained, yet.")

    def _replace_input(self, node: ast.AST) -> BaseAction:
        raise NotImplementedError("This action can't be chained, yet.")


class _DeprecatedAliasMixin:
    def __post_init__(self, *args, **kwargs):
        warnings.warn(
            f"{type(self).__name__!r} is deprecated, use "
            f"{type(self).__base__.__name__!r} instead",
            DeprecationWarning,
            stacklevel=3,
        )
        with suppress(AttributeError):
            super().__post_init__(*args, **kwargs)


@dataclass
class _LazyActionMixin(Generic[K, T], BaseAction):
    node: K

    def build(self) -> T:
        raise NotImplementedError

    def branch(self) -> K:
        return clone(self.node)

    def _replace_input(self, node: ast.AST) -> _LazyActionMixin[K, T]:
        return replace(self, node=node)


class _ReplaceCodeSegmentAction(BaseAction):
    def apply(self, context: Context, source: str) -> str:
        lines = split_lines(source, encoding=context.file_info.get_encoding())
        lineno, col_offset, end_lineno, end_col_offset = self._get_segment_span(
            context
        )

        view = slice(lineno - 1, end_lineno)
        source_lines = lines[view]

        indentation, start_prefix = find_indent(source_lines[0][:col_offset])
        end_suffix = source_lines[-1][end_col_offset:]

        replacement = split_lines(self._resynthesize(context))
        replacement.apply_indentation(
            indentation,
            start_prefix=start_prefix,
            end_suffix=end_suffix,
        )

        lines[view] = replacement
        return lines.join()

    def _get_segment_span(self, context: Context) -> PositionType:
        raise NotImplementedError

    def _resynthesize(self, context: Context) -> str:
        raise NotImplementedError


@_hint("deprecated_alias", "Action")
@dataclass
class LazyReplace(_ReplaceCodeSegmentAction, _LazyActionMixin[ast.AST, ast.AST]):
    """Replace a node with the node returned by :meth:`build`."""

    def _get_segment_span(self, context: Context) -> PositionType:
        return position_for(self.node)

    def _resynthesize(self, context: Context) -> str:
        return context.unparse(self.build())

    def _stack_effect(self) -> tuple[ast.AST, int]:
        return self.node, 0


@dataclass
class Action(LazyReplace, _DeprecatedAliasMixin):
    pass


@_hint("deprecated_alias", "ReplacementAction")
@dataclass
class Replace(LazyReplace):
    """Replace ``node`` with ``target``."""

    target: ast.AST

    def build(self) -> ast.AST:
        return self.target


@dataclass
class ReplacementAction(Replace, _DeprecatedAliasMixin):
    pass


@_hint("deprecated_alias", "NewStatementAction")
@dataclass
class LazyInsertAfter(_LazyActionMixin[ast.stmt, ast.stmt]):
    """Insert the statement returned by :meth:`build` after ``node``."""

    def apply(self, context: Context, source: str) -> str:
        lines = split_lines(source, encoding=context.file_info.get_encoding())
        indentation, start_prefix = find_indent(
            lines[self.node.lineno - 1][: self.node.col_offset]
        )

        replacement = split_lines(context.unparse(self.build()))
        replacement.apply_indentation(indentation, start_prefix=start_prefix)

        original_node_end = cast(int, self.node.end_lineno) - 1
        if lines[original_node_end].endswith(lines._newline_type):
            replacement[-1] += lines._newline_type
        else:
            replacement[0] = lines._newline_type + replacement[0]

        for line in reversed(replacement):
            lines.insert(original_node_end + 1, line)

        return lines.join()

    def _stack_effect(self) -> tuple[ast.AST, int]:
        return self.node, 1


@dataclass
class LazyInsertBefore(_LazyActionMixin[ast.stmt, ast.stmt]):
    """Insert the statement returned by :meth:`build` before ``node``."""

    def apply(self, context: Context, source: str) -> str:
        lines = split_lines(source, encoding=context.file_info.get_encoding())
        indentation, start_prefix = find_indent(
            lines[self.node.lineno - 1][: self.node.col_offset]
        )

        replacement = split_lines(context.unparse(self.build()))
        replacement.apply_indentation(indentation, start_prefix=start_prefix)
        replacement[-1] += lines._newline_type

        original_node_start = cast(int, self.node.lineno)
        for line in reversed(replacement):
            lines.insert(original_node_start - 1, line)

        return lines.join()

    def _stack_effect(self) -> tuple[ast.AST, int]:
        return self.node, -1


@dataclass
class NewStatementAction(LazyInsertAfter, _DeprecatedAliasMixin):
    pass


@dataclass
class NewStatementBeforeAction(LazyInsertBefore, _DeprecatedAliasMixin):
    pass


@_hint("deprecated_alias", "TargetedNewStatementAction")
@dataclass
class InsertAfter(LazyInsertAfter):
    """Insert ``target`` after ``node``."""

    target: ast.stmt

    def build(self) -> ast.stmt:
        return self.target


@dataclass
class InsertBefore(LazyInsertBefore):
    """Insert ``target`` before ``node``."""

    target: ast.stmt

    def build(self) -> ast.stmt:
        return self.target


@dataclass
class TargetedNewStatementAction(InsertAfter, _DeprecatedAliasMixin):
    pass


@dataclass
class TargetedNewStatementBeforeAction(InsertBefore, _DeprecatedAliasMixin):
    pass


@dataclass
class _Rename(Replace):
    identifier_span: PositionType

    def _get_segment_span(self, context: Context) -> PositionType:
        return self.identifier_span

    def _resynthesize(self, context: Context) -> str:
        return self.target.name


@dataclass
class Erase(_ReplaceCodeSegmentAction):
    """Remove the source segment occupied by a statement."""

    node: ast.stmt

    def _get_segment_span(self, context: Context) -> PositionType:
        return position_for(self.node)

    def _resynthesize(self, context: Context) -> str:
        return ""

    def _stack_effect(self) -> tuple[ast.AST, int]:
        return self.node, -1

    def _replace_input(self, node: ast.AST) -> Erase:
        return replace(self, node=node)


@dataclass
class EraseOrReplace(Erase):
    """Erase a statement, replacing it when an empty suite would result."""

    target: ast.stmt = field(default_factory=ast.Pass)

    @staticmethod
    def _same_node(left: ast.AST, right: ast.AST) -> bool:
        return (
            type(left) is type(right)
            and getattr(left, "lineno", None) == getattr(right, "lineno", None)
            and getattr(left, "col_offset", None)
            == getattr(right, "col_offset", None)
            and getattr(left, "end_lineno", None)
            == getattr(right, "end_lineno", None)
            and getattr(left, "end_col_offset", None)
            == getattr(right, "end_col_offset", None)
        )

    def _requires_replacement(self, source: str) -> bool:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            return False

        for parent in ast.walk(tree):
            for _, value in ast.iter_fields(parent):
                if not isinstance(value, list):
                    continue

                matching = [
                    item
                    for item in value
                    if isinstance(item, ast.stmt) and self._same_node(item, self.node)
                ]
                if matching and len(value) == 1:
                    return True

        return False

    def apply(self, context: Context, source: str) -> str:
        if self._requires_replacement(source):
            return Replace(self.node, self.target).apply(context, source)
        return super().apply(context, source)