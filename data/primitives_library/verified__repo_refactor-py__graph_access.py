from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar, Union

from refactor import common
from refactor.context import Context


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class AccessFailure(Exception):
    pass


@dataclass
class Access(Generic[InputType, OutputType]):
    expected_type: type[Any]

    def __repr__(self) -> str:
        raise NotImplementedError

    def _check(self, condition: bool) -> None:
        if not condition:
            raise AccessFailure

    def execute(self, input: InputType) -> OutputType:
        raise NotImplementedError

    replace = replace


@dataclass
class FieldAccess(Access[ast.AST, Union[ast.AST, list[ast.AST]]]):
    field: str

    def __repr__(self) -> str:
        return "." + self.field

    def execute(self, input: ast.AST) -> ast.AST | list[ast.AST]:
        value = getattr(input, self.field)
        self._check(type(value) is self.expected_type)
        return value


@dataclass
class IndexAccess(Access[list[ast.AST], ast.AST]):
    index: int

    def __repr__(self) -> str:
        return f"[{self.index}]"

    def execute(self, input: list[ast.AST]) -> ast.AST:
        self._check(isinstance(input, list))
        self._check(len(input) > self.index)
        value = input[self.index]
        self._check(type(value) is self.expected_type)
        return value


@dataclass
class GraphPath:
    parts: list[Access] = field(default_factory=list)

    @classmethod
    def backtrack_from(cls, context: Context, node: ast.AST) -> GraphPath:
        path_parts: list[Access] = []
        current = node

        for field_name, parent in context.ancestry.traverse(node):
            parent_value = getattr(parent, field_name)

            if isinstance(parent_value, list):
                path_parts.append(
                    IndexAccess(type(current), parent_value.index(current))
                )
                path_parts.append(FieldAccess(list, field_name))
            elif isinstance(parent_value, ast.AST):
                path_parts.append(FieldAccess(type(current), field_name))
            else:
                raise TypeError(
                    "Unexpeced ancestor field type:"
                    f" {type(parent_value).__name__}"
                )

            current = parent

        path_parts.reverse()
        return cls(path_parts)

    @common._allow_asserts
    def shift(self, shifts: list[tuple[GraphPath, int]]) -> GraphPath:
        updated_parts = self.parts.copy()

        for shift_path, offset in shifts:
            *shift_parent, shifter = shift_path.parts
            *matching_prefix, target = updated_parts[: len(shift_path.parts)]

            if shift_parent != matching_prefix or not offset:
                continue

            assert isinstance(shifter, IndexAccess)
            assert isinstance(target, IndexAccess)

            if shifter.index + offset >= target.index:
                continue

            updated_parts[updated_parts.index(target)] = target.replace(
                index=target.index + abs(offset)
            )

        return GraphPath(updated_parts)

    def execute(self, node: ast.AST) -> ast.AST:
        current = node
        for part in self.parts:
            current = part.execute(current)
        return current

    def __repr__(self) -> str:
        return "".join(repr(part) for part in self.parts)