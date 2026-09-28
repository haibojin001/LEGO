from __future__ import annotations

import typing

from marshmallow.exceptions import RegistryError

if typing.TYPE_CHECKING:
    from marshmallow import Schema

    SchemaType = type[Schema]

_registry: dict[str, list[SchemaType]] = {}


def register(classname: str, cls: SchemaType) -> None:
    """Add a schema class to the registry."""
    module_name = cls.__module__
    qualified_name = f"{module_name}.{classname}"

    if classname not in _registry:
        _registry[classname] = [cls]
    elif not any(item.__module__ == module_name for item in _registry[classname]):
        _registry[classname].append(cls)

    if qualified_name in _registry:
        _registry[qualified_name] = [cls]
    else:
        _registry.setdefault(qualified_name, []).append(cls)


@typing.overload
def get_class(classname: str, *, all: typing.Literal[False] = ...) -> SchemaType: ...


@typing.overload
def get_class(
    classname: str, *, all: typing.Literal[True] = ...
) -> list[SchemaType]: ...


def get_class(classname: str, *, all: bool = False) -> list[SchemaType] | SchemaType:
    """Retrieve a schema class from the registry."""
    try:
        matches = _registry[classname]
    except KeyError as exc:
        raise RegistryError(
            f"Class with name {classname!r} was not found. You may need "
            "to import the class."
        ) from exc

    if len(matches) > 1:
        if all:
            return matches
        raise RegistryError(
            f"Multiple classes with name {classname!r} "
            "were found. Please use the full, "
            "module-qualified path."
        )

    return matches[0]