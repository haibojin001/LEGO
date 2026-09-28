from __future__ import annotations

import functools
import operator
import os
import platform
import sys
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Literal, TypedDict, cast

# Some reconstructed versions of packaging.version omit this private helper,
# although packaging.utils imports it.
try:
    from .version import _TrimmedRelease as _TrimmedRelease
except ImportError:
    from . import version as _version

    class _TrimmedRelease(_version.Version):
        @property
        def release(self) -> tuple[int, ...]:
            release = super().release
            while len(release) > 1 and release[-1] == 0:
                release = release[:-1]
            return release

    _version._TrimmedRelease = _TrimmedRelease

from .specifiers import InvalidSpecifier, Specifier

# Newer parser/tokenizer code uses this combined expression while some
# reconstructed Specifier implementations only expose its component pieces.
if not hasattr(Specifier, "_specifier_regex_str"):
    try:
        Specifier._specifier_regex_str = (
            Specifier._operator_regex_str + Specifier._version_regex_str
        )
    except AttributeError:
        Specifier._specifier_regex_str = (
            r"(?P<operator>~=|==|!=|<=|>=|<|>|===)"
            r"\s*"
            r"(?P<version>[^\s,;()]+)"
        )

from ._parser import MarkerAtom, MarkerList, Op, Value, Variable
from ._tokenizer import ParserSyntaxError
from .utils import canonicalize_name

try:
    from ._parser import parse_marker as _parse_marker
except ImportError:
    from ._parser import _parse_full_marker
    from ._tokenizer import DEFAULT_RULES, Tokenizer

    def _parse_marker(source: str) -> MarkerList:
        return _parse_full_marker(Tokenizer(source, rules=DEFAULT_RULES))

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "Environment",
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]


def __dir__() -> list[str]:
    return __all__


Operator = Callable[[str, str | AbstractSet[str]], bool]
EvaluateContext = Literal["metadata", "lock_file", "requirement"]

MARKERS_ALLOWING_SET = {"extras", "dependency_groups"}
MARKERS_REQUIRING_VERSION = {
    "implementation_version",
    "platform_release",
    "python_full_version",
    "python_version",
}


class InvalidMarker(ValueError):
    """Raised when a marker expression cannot be parsed."""


class UndefinedComparison(ValueError):
    """Raised for marker comparisons that have no defined meaning."""


class UndefinedEnvironmentName(KeyError):
    """Raised when a marker refers to an unavailable environment value."""


class Environment(TypedDict):
    implementation_name: str
    implementation_version: str
    os_name: str
    platform_machine: str
    platform_release: str
    platform_system: str
    platform_version: str
    python_full_version: str
    platform_python_implementation: str
    python_version: str
    sys_platform: str


def _format_full_version(info: object) -> str:
    major = getattr(info, "major")
    minor = getattr(info, "minor")
    micro = getattr(info, "micro")
    result = f"{major}.{minor}.{micro}"

    releaselevel = getattr(info, "releaselevel")
    if releaselevel != "final":
        result += f"{releaselevel[0]}{getattr(info, 'serial')}"

    return result


def _get_impl_version() -> str:
    implementation = getattr(sys, "implementation", None)
    if implementation is None:
        return "0"
    return _format_full_version(implementation.version)


def default_environment() -> Environment:
    python_version = platform.python_version()

    return {
        "implementation_name": platform.python_implementation().lower(),
        "implementation_version": _get_impl_version(),
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": python_version,
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


@functools.lru_cache
def _cached_default_environment() -> Environment:
    return default_environment()


def _normalize_extras(
    item: MarkerList | MarkerAtom | str,
) -> MarkerList | MarkerAtom | str:
    if isinstance(item, list):
        return [_normalize_extras(child) for child in item]

    if not isinstance(item, tuple):
        return item

    left, comparison, right = item

    if isinstance(left, Variable) and left.value == "extra" and isinstance(right, Value):
        right = Value(canonicalize_name(right.value))
    elif isinstance(right, Variable) and right.value == "extra" and isinstance(left, Value):
        left = Value(canonicalize_name(left.value))
    elif (
        isinstance(right, Variable)
        and right.value in MARKERS_ALLOWING_SET
        and isinstance(left, Value)
    ):
        left = Value(canonicalize_name(left.value))

    return left, comparison, right


def _normalize_extra_values(markers: MarkerList) -> MarkerList:
    return [_normalize_extras(marker) for marker in markers]


def _format_marker(
    marker: list[str] | MarkerAtom | str, first: bool | None = True
) -> str:
    if (
        isinstance(marker, list)
        and len(marker) == 1
        and isinstance(marker[0], (list, tuple))
    ):
        return _format_marker(marker[0], first=first)

    if isinstance(marker, str):
        return marker

    if isinstance(marker, tuple):
        return " ".join(part.serialize() for part in marker)

    text = " ".join(_format_marker(part, first=False) for part in marker)
    if first:
        return text
    return f"({text})"


_operators: dict[str, Operator] = {
    "in": lambda left, right: left in right,
    "not in": lambda left, right: left not in right,
    "<": lambda _left, _right: False,
    "<=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.eq,
    ">": lambda _left, _right: False,
}


def _eval_op(
    left: str, comparison: Op, right: str | AbstractSet[str], *, key: str
) -> bool:
    operation = comparison.serialize()

    if key in MARKERS_REQUIRING_VERSION:
        try:
            specifier = Specifier(f"{operation}{right}")
        except InvalidSpecifier:
            pass
        else:
            return specifier.contains(left, prereleases=True)

    evaluator = _operators.get(operation)
    if evaluator is None:
        raise UndefinedComparison(
            f"Undefined {comparison!r} on {left!r} and {right!r}."
        )

    return evaluator(left, right)


def _normalize(
    left: str, right: str | AbstractSet[str], key: str
) -> tuple[str, str | AbstractSet[str]]:
    if key == "extra":
        assert isinstance(right, str), "extra value must be a string"
        return canonicalize_name(left), canonicalize_name(right)

    if key in MARKERS_ALLOWING_SET:
        normalized_left = canonicalize_name(left)
        if isinstance(right, str):
            return normalized_left, canonicalize_name(right)
        return normalized_left, {canonicalize_name(value) for value in right}

    return left, right


def _lookup_environment(
    environment: Mapping[str, str | AbstractSet[str]], key: str
) -> str | AbstractSet[str]:
    try:
        return environment[key]
    except KeyError:
        raise UndefinedEnvironmentName(key) from None


def _evaluate_markers(
    markers: MarkerList, environment: Mapping[str, str | AbstractSet[str]]
) -> bool:
    alternatives: list[list[bool]] = [[]]

    for marker in markers:
        if isinstance(marker, list):
            alternatives[-1].append(_evaluate_markers(marker, environment))
            continue

        if isinstance(marker, tuple):
            left, comparison, right = marker

            if isinstance(left, Variable):
                key = left.value
                left_value = _lookup_environment(environment, key)
                right_value = right.value
            else:
                key = right.value
                left_value = left.value
                right_value = _lookup_environment(environment, key)

            normalized_left, normalized_right = _normalize(
                cast(str, left_value),
                cast(str | AbstractSet[str], right_value),
                key,
            )
            alternatives[-1].append(
                _eval_op(normalized_left, comparison, normalized_right, key=key)
            )
            continue

        assert marker in {"and", "or"}
        if marker == "or":
            alternatives.append([])

    return any(all(group) for group in alternatives)


class Marker:
    def __init__(self, marker: str) -> None:
        try:
            parsed = _parse_marker(marker)
        except ParserSyntaxError as exc:
            raise InvalidMarker(str(exc)) from exc

        self._markers = _normalize_extra_values(parsed)

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<Marker('{self}')>"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented
        return str(self) == str(other)

    def evaluate(
        self,
        environment: Mapping[str, str | AbstractSet[str]] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        if context == "lock_file":
            current_environment: dict[str, str | AbstractSet[str]] = {
                "extras": frozenset(),
                "dependency_groups": frozenset(),
            }
        elif context == "metadata":
            current_environment = {"extra": ""}
        elif context == "requirement":
            current_environment = {}
        else:
            raise ValueError(
                "Invalid context. Expected one of: 'metadata', 'lock_file', "
                f"or 'requirement', got {context!r}"
            )

        current_environment.update(_cached_default_environment())

        if environment is not None:
            current_environment.update(environment)

        return _evaluate_markers(self._markers, current_environment)