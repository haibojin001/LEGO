from __future__ import annotations

from typing import TYPE_CHECKING

from ._parser import parse_requirement as _parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import InvalidSpecifier, SpecifierSet
from .utils import canonicalize_name

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["InvalidRequirement", "Requirement"]


def __dir__() -> list[str]:
    return __all__


class InvalidRequirement(ValueError):
    """Raised when a requirement does not conform to PEP 508."""


class Requirement:
    """Representation of a parsed dependency requirement."""

    __slots__ = ("extras", "marker", "name", "specifier", "url")

    def __init__(self, requirement_string: str) -> None:
        try:
            result = _parse_requirement(requirement_string)
        except ParserSyntaxError as exc:
            raise InvalidRequirement(str(exc)) from exc

        self.name = result.name
        self.url = result.url or None
        self.extras = set(result.extras)

        try:
            self.specifier = SpecifierSet(result.specifier)
        except InvalidSpecifier as exc:
            raise InvalidRequirement(str(exc)) from exc

        self.marker = None
        if result.marker is not None:
            marker = object.__new__(Marker)
            marker._markers = _normalize_extra_values(result.marker)
            self.marker = marker

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name

        if self.extras:
            yield "[" + ",".join(sorted(self.extras)) + "]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield " @ " + self.url
            if self.marker:
                yield " "

        if self.marker:
            yield "; " + str(self.marker)

    def __str__(self) -> str:
        return "".join(self._iter_parts(self.name))

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(
            (
                canonicalize_name(self.name),
                frozenset(canonicalize_name(extra) for extra in self.extras),
                self.specifier,
                self.url,
                self.marker,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Requirement):
            return NotImplemented

        return (
            canonicalize_name(self.name) == canonicalize_name(other.name)
            and frozenset(canonicalize_name(extra) for extra in self.extras)
            == frozenset(canonicalize_name(extra) for extra in other.extras)
            and self.specifier == other.specifier
            and self.url == other.url
            and self.marker == other.marker
        )

    def __getstate__(self) -> tuple[str, bool | None]:
        return str(self), self.specifier._prereleases

    def __setstate__(self, state: object) -> None:
        prerelease_override: bool | None

        if isinstance(state, str):
            text = state
            prerelease_override = None
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], str)
            and (state[1] is None or isinstance(state[1], bool))
        ):
            text = state[0]
            prerelease_override = state[1]
        elif isinstance(state, dict) and state.keys() >= set(self.__slots__):
            for attribute in self.__slots__:
                setattr(self, attribute, state[attribute])
            return
        else:
            raise TypeError(f"Cannot restore Requirement from {state!r}")

        try:
            restored = Requirement(text)
        except InvalidRequirement as exc:
            raise TypeError(f"Cannot restore Requirement from {state!r}") from exc

        self.name = restored.name
        self.url = restored.url
        self.extras = restored.extras
        self.specifier = restored.specifier
        self.specifier._prereleases = prerelease_override
        self.marker = restored.marker