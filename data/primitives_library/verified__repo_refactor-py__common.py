from __future__ import annotations

import ast
import copy
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache, singledispatch, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

if TYPE_CHECKING:
    from refactor.context import Context


T = TypeVar("T")
C = TypeVar("C")
PositionType = tuple[int, int, int, int]


@dataclass
class _FileInfo:
    """Represents information regarding source code."""

    path: Path | None = None
    encoding: str | None = None

    def get_encoding(self) -> str:
        from refactor.ast import DEFAULT_ENCODING

        return self.encoding or DEFAULT_ENCODING


def clone(node: T) -> T:
    """Clone the given node."""
    return copy.deepcopy(node)


def negate(node: ast.expr) -> ast.UnaryOp:
    """Negate the given expression."""
    return ast.UnaryOp(op=ast.Not(), operand=node)


def apply_condition(condition: bool, node: ast.expr) -> ast.expr:
    """Return node when condition is true, otherwise its negation."""
    if condition:
        return node
    return negate(node)


def wrap_with_parens(text: str) -> str:
    """Wrap text in parentheses."""
    return f"({text})"


_OPERATOR_MAP = {
    ast.Eq: True,
    ast.In: True,
    ast.Is: True,
    ast.NotEq: False,
    ast.NotIn: False,
    ast.IsNot: False,
}


def is_truthy(op: ast.cmpop) -> bool | None:
    """Return the truthiness direction of a comparison operator."""
    return _OPERATOR_MAP.get(type(op))


def _type_checker(
    *types: type,
    binders: Iterable[Callable[[type], bool]] = (),
) -> Callable[[Any], bool]:
    fast_binders = [getattr(binder, "fast_checker", binder) for binder in binders]

    @cache
    def check_type(node_type: type) -> bool:
        if issubclass(node_type, types):
            return True
        return any(binder(node_type) for binder in fast_binders)

    def check(node: Any) -> bool:
        return check_type(type(node))

    check.fast_checker = check_type
    return check


is_comprehension = _type_checker(
    ast.SetComp,
    ast.ListComp,
    ast.DictComp,
    ast.GeneratorExp,
)
is_function = _type_checker(ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
is_contextful = _type_checker(
    ast.Module,
    ast.ClassDef,
    binders=(is_function, is_comprehension),
)


def compare_ast(left: ast.AST, right: ast.AST, /) -> bool:
    """Compare two AST nodes."""
    return ast.dump(left) == ast.dump(right)


def _guarded(exc_type: type[BaseException], /, default: Any = None) -> Any:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            except exc_type:
                return default

        return wrapped

    return decorate


def _get_known_location_from_source(
    source: str, location: PositionType
) -> str | None:
    start_line, start_column, end_line, end_column = location
    start_line -= 1
    end_line -= 1

    lines = source.splitlines()
    if len(lines) < end_line + 1:
        return None

    if start_line == end_line:
        return lines[start_line][start_column:end_column]

    first, *between, last = lines[start_line : end_line + 1]
    return "\n".join((first[start_column:], *between, last[:end_column]))


@_guarded(Exception)
def get_source_segment(source: str, node: ast.AST) -> str | None:
    """Return the source occupied by node, if its location is known."""
    try:
        location = position_for(node)
    except AttributeError:
        return None
    return _get_known_location_from_source(source, location)


def pascal_to_snake(name: str) -> str:
    """Convert PascalCase text to snake_case text."""
    result = ""
    for index, character in enumerate(name):
        if index and character.isupper():
            result += "_"
        result += character
    return result.lower()


def find_indent(source: str) -> tuple[str, str]:
    """Split a line into indentation and the remaining text."""
    index = 0
    for index, character in enumerate(source, 1):
        if not character.isspace():
            index -= 1
            break
    return source[:index], source[index:]


def find_closest(node: ast.AST, *targets: ast.AST) -> ast.AST:
    """Find the target with the closest source position to node."""
    if not len(targets) >= 1:
        raise ValueError("condition failed: len(targets) >= 1")

    origin = position_for(node)

    def distance(target: ast.AST) -> tuple[int, ...]:
        target_position = position_for(target)
        return tuple(
            abs(target_value - origin_value)
            for target_value, origin_value in zip(target_position, origin)
        )

    return sorted(targets, key=distance)[0]


def extract_from_text(text: str) -> ast.AST:
    """Extract the first AST node from text."""
    return ast.parse(text).body[0]


_POSITIONAL_ATTRIBUTES = (
    "lineno",
    "col_offset",
    "end_lineno",
    "end_col_offset",
)
_POSITIONAL_ATTRIBUTES_SET = frozenset(_POSITIONAL_ATTRIBUTES)


@cache
def has_positions(node_type: type[ast.AST]) -> bool:
    """Return whether an AST node type tracks source positions."""
    return _POSITIONAL_ATTRIBUTES_SET.issubset(node_type._attributes)


def position_for(node: ast.AST) -> PositionType:
    """Return the source position of node."""
    values = tuple(getattr(node, name) for name in _POSITIONAL_ATTRIBUTES)
    return cast(PositionType, values)


def _hint(node: T, position: PositionType) -> T:
    """Attach source-position information to an AST node."""
    for attribute, value in zip(_POSITIONAL_ATTRIBUTES, position):
        setattr(node, attribute, value)
    return node


def unpack_lhs(node: ast.AST) -> Iterator[str]:
    """Yield names represented by an assignment target."""
    if isinstance(node, (ast.List, ast.Tuple)):
        for child in node.elts:
            yield from unpack_lhs(child)
    else:
        yield ast.unparse(node)


def next_statement_of(node: ast.stmt, context: Context) -> ast.stmt | None:
    """Return the statement immediately following node in its parent."""
    parent_field, parent = context.ancestry.infer(node)
    if not parent_field is not None:
        raise ValueError("condition failed: parent_field is not None")
    if not parent is not None:
        raise ValueError("condition failed: parent is not None")

    siblings = getattr(parent, parent_field)
    if not isinstance(siblings, list):
        return None

    position = siblings.index(node)
    try:
        return siblings[position + 1]
    except IndexError:
        return None


def walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Walk nodes visible from within a scope."""
    pending = deque(_walker(node))
    while pending:
        current = pending.popleft()
        pending.extend(_walker(current, top_level=True))
        yield current


def _walk_optional(node: ast.AST | None) -> Iterator[ast.AST]:
    if node is not None:
        yield node


def _walk_optional_list(nodes: Iterable[ast.AST | None]) -> Iterator[ast.AST]:
    for node in nodes:
        yield from _walk_optional(node)


def _walk_args(arguments: ast.arguments) -> Iterator[ast.AST]:
    yield from arguments.posonlyargs
    yield from arguments.args
    yield from arguments.kwonlyargs
    yield from _walk_optional(arguments.vararg)
    yield from _walk_optional(arguments.kwarg)


@singledispatch
def _walker(node: ast.AST, top_level: bool = False) -> Iterator[ast.AST]:
    yield from ast.iter_child_nodes(node)


@_walker.register(ast.Module)
@_walker.register(ast.SetComp)
@_walker.register(ast.ListComp)
@_walker.register(ast.DictComp)
@_walker.register(ast.GeneratorExp)
def _walk_ignore(node: ast.AST, top_level: bool = False) -> Iterator[ast.AST]:
    if not top_level:
        yield from ast.iter_child_nodes(node)


@_walker.register(ast.FunctionDef)
@_walker.register(ast.AsyncFunctionDef)
def _walk_func(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    top_level: bool = False,
) -> Iterator[ast.AST]:
    if top_level:
        yield from node.decorator_list
        yield from node.args.defaults
        yield from _walk_optional_list(node.args.kw_defaults)
        yield from _walk_optional(node.returns)
    else:
        yield from _walk_args(node.args)
        yield from node.body


@_walker.register(ast.Lambda)
def _walk_lambda(node: ast.Lambda, top_level: bool = False) -> Iterator[ast.AST]:
    if top_level:
        yield from node.args.defaults
        yield from _walk_optional_list(node.args.kw_defaults)
    else:
        yield from _walk_args(node.args)
        yield node.body


@_walker.register(ast.ClassDef)
def _walk_class(node: ast.ClassDef, top_level: bool = False) -> Iterator[ast.AST]:
    if top_level:
        yield from node.decorator_list
        yield from node.bases
        yield from node.keywords
    else:
        yield