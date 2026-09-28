from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from ._ranges import (
    FULL_RANGE,
    MIN_VERSION,
    NEG_INF,
    POS_INF,
    BoundaryKind,
    BoundaryVersion,
    LowerBound,
    UpperBound,
    coerce_version,
    filter_by_ranges,
    intersect_ranges,
    least_version_above,
    matches_bounds_only,
    range_is_empty,
    ranges_are_prerelease_only,
    trim_release,
)
from .specifiers import SpecifierSet
from .version import InvalidVersion, Version

__all__ = ["VersionRange"]

T = TypeVar("T")
UnparsedVersion = Version | str
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)
Interval = tuple[LowerBound, UpperBound]

_MAX_EXCLUSION_RUN = 128


class _SetOp(enum.Enum):
    INTERSECTION = enum.auto()
    UNION = enum.auto()
    DIFFERENCE = enum.auto()


def __dir__() -> list[str]:
    return __all__


def _union_ranges(
    left: Sequence[Interval], right: Sequence[Interval]
) -> list[Interval]:
    ordered = sorted([*left, *right], key=lambda item: item[0])
    if not ordered:
        return []

    result: list[Interval] = [ordered[0]]
    for lower, upper in ordered[1:]:
        old_lower, old_upper = result[-1]
        joined = False

        if old_upper.version is None or lower.version is None:
            joined = True
        elif old_upper.version > lower.version:
            joined = True
        elif old_upper.version == lower.version:
            joined = old_upper.inclusive or lower.inclusive
        else:
            gap_lower = LowerBound(old_upper.version, not old_upper.inclusive)
            gap_upper = UpperBound(lower.version, not lower.inclusive)
            joined = range_is_empty(gap_lower, gap_upper)

        if joined:
            result[-1] = (old_lower, max(old_upper, upper))
        else:
            result.append((lower, upper))

    return result


def _complement_ranges(ranges: Sequence[Interval]) -> list[Interval]:
    if not ranges:
        return list(FULL_RANGE)

    answer: list[Interval] = []
    previous: UpperBound | None = None

    for lower, upper in ranges:
        if previous is None:
            if lower.version is not None:
                answer.append((NEG_INF, UpperBound(lower.version, not lower.inclusive)))
        else:
            answer.append(
                (
                    LowerBound(previous.version, not previous.inclusive),
                    UpperBound(lower.version, not lower.inclusive),
                )
            )
        previous = upper

    if previous is not None and previous.version is not None:
        answer.append((LowerBound(previous.version, not previous.inclusive), POS_INF))

    return answer


def _canonical_floor(bounds: tuple[Interval, ...]) -> tuple[Interval, ...]:
    if not bounds:
        return bounds

    lower, upper = bounds[0]
    if range_is_empty(NEG_INF, upper):
        return bounds[1:]

    if (
        lower.inclusive
        and isinstance(lower.version, Version)
        and lower.version <= MIN_VERSION
    ):
        return ((NEG_INF, upper), *bounds[1:])

    return bounds


def _predecessor_boundary(version: Version) -> BoundaryVersion | None:
    if version.dev is None:
        return None

    candidate: BoundaryVersion | None = None

    if version.pre is not None and version.dev == 0 and version.post is None:
        label, number = version.pre
        if number:
            candidate = BoundaryVersion(
                version.__replace__(pre=(label, number - 1), dev=None),
                BoundaryKind.AFTER_POSTS,
            )
    elif version.dev >= 1:
        candidate = BoundaryVersion(
            version.__replace__(dev=version.dev - 1),
            BoundaryKind.AFTER_LOCALS,
        )
    elif version.dev == 0 and version.post is not None:
        if version.post == 0:
            base = version.__replace__(post=None, dev=None)
        else:
            base = version.__replace__(post=version.post - 1, dev=None)
        candidate = BoundaryVersion(base, BoundaryKind.AFTER_LOCALS)

    if candidate is not None and least_version_above(candidate) == version:
        return candidate
    return None


def _canonicalize(bounds: tuple[Interval, ...]) -> tuple[Interval, ...]:
    changed: list[Interval] = []

    for lower, upper in bounds:
        new_lower = lower
        new_upper = upper

        if lower.inclusive and isinstance(lower.version, Version):
            predecessor = _predecessor_boundary(lower.version)
            if predecessor is not None:
                new_lower = LowerBound(predecessor, False)

        changed.append((new_lower, new_upper))

    return _canonical_floor(tuple(_union_ranges((), changed)))


def _release_version(release: Sequence[int]) -> Version:
    return Version(".".join(str(part) for part in release))


def _next_release_prefix(version: Version, width: int) -> Version:
    release = list(version.release[:width])
    if not release:
        release = [1]
    release[-1] += 1
    return _release_version(release)


def _prefix_bounds(text: str) -> Interval:
    prefix = text[:-2]
    if prefix.endswith("."):
        prefix = prefix[:-1]
    if not prefix:
        return (NEG_INF, POS_INF)

    version = Version(prefix)
    lower_version = version.__replace__(pre=None, post=None, dev=0, local=None)
    upper_version = _next_release_prefix(version, len(version.release))
    return (
        LowerBound(lower_version, True),
        UpperBound(upper_version, False),
    )


def _exact_bounds(version: Version) -> Interval:
    if version.local is None:
        end: Version | BoundaryVersion = BoundaryVersion(
            version, BoundaryKind.AFTER_LOCALS
        )
        return (LowerBound(version, True), UpperBound(end, False))
    return (LowerBound(version, True), UpperBound(version, True))


def _compatible_upper(version: Version) -> Version:
    release = trim_release(version.release)
    if len(release) < 2:
        raise ValueError("Compatible release clauses require at least two release segments")
    return _next_release_prefix(version, len(release) - 1)


def _interval_for_specifier(operator: str, text: str) -> list[Interval]:
    if operator == "===":
        try:
            return [_exact_bounds(Version(text))]
        except InvalidVersion:
            return []

    if operator in ("==", "!=") and text.endswith(".*"):
        interval = _prefix_bounds(text)
        if operator == "==":
            return [interval]
        return _complement_ranges([interval])

    version = Version(text)

    if operator == "==":
        return [_exact_bounds(version)]

    if operator == "!=":
        return _complement_ranges([_exact_bounds(version)])

    if operator == ">=":
        return [(LowerBound(version, True), POS_INF)]

    if operator == "<":
        return [(NEG_INF, UpperBound(version, False))]

    if operator == "<=":
        if version.local is None:
            upper: Version | BoundaryVersion = BoundaryVersion(
                version, BoundaryKind.AFTER_LOCALS
            )
            return [(NEG_INF, UpperBound(upper, False))]
        return [(NEG_INF, UpperBound(version, True))]

    if operator == ">":
        boundary = BoundaryVersion(version, BoundaryKind.AFTER_POSTS)
        return [(LowerBound(boundary, False), POS_INF)]

    if operator == "~=":
        return [
            (
                LowerBound(version, True),
                UpperBound(_compatible_upper(version), False),
            )
        ]

    raise ValueError(f"Unsupported version specifier operator: {operator!r}")


def _difference_ranges(left: Sequence[Interval], right: Sequence[Interval]) -> list[Interval]:
    return intersect_ranges(left, _complement_ranges(right))


def _bounds_to_specifiers(bounds: Sequence[Interval]) -> str | None:
    if not bounds:
        return "<0.dev0"

    if len(bounds) != 1:
        return None

    lower, upper = bounds[0]
    pieces: list[str] = []

    if lower.version is not None:
        if isinstance(lower.version, BoundaryVersion):
            return None
        pieces.append((">=" if lower.inclusive else ">") + str(lower.version))

    if upper.version is not None:
        if isinstance(upper.version, BoundaryVersion):
            return None
        pieces.append(("<=" if upper.inclusive else "<") + str(upper.version))

    return ",".join(pieces)


class VersionRange:
    def __init__(self, specifier_set: SpecifierSet | str = "") -> None:
        if isinstance(specifier_set, str):
            specifier_set = SpecifierSet(specifier_set)
        elif not isinstance(specifier_set, SpecifierSet):
            raise TypeError("specifier_set must be a SpecifierSet or string")

        self._specifier_set: SpecifierSet | None = specifier_set
        self._literals: frozenset[str] = frozenset()
        intervals: list[Interval] = list(FULL_RANGE)

        literal_values: set[str] = set()
        for specifier in specifier_set:
            operator = specifier.operator
            text = specifier.version

            if operator == "===":
                try:
                    parsed = Version(text)
                except InvalidVersion:
                    literal_values.add(text)
                    intervals = []
                    continue
                part = [_exact_bounds(parsed)]
            else:
                part = _interval_for_specifier(operator, text)

            intervals = intersect_ranges(intervals, part)

        self._bounds = _canonicalize(tuple(intervals))
        self._literals = frozenset(literal_values)
        self._prereleases = specifier_set.prereleases

    @classmethod
    def _from_parts(
        cls,
        bounds: Sequence[Interval],
        prereleases: bool | None = None,
        literals: Iterable[str] = (),
    ) -> VersionRange:
        result = cls.__new__(cls)
        result._bounds = _canonicalize(tuple(_union_ranges((), bounds)))
        result._literals = frozenset(literals)
        result._prereleases = prereleases
        result._specifier_set = None
        return result

    @property
    def specifier_set(self) -> SpecifierSet | None:
        return self.to_specifier_set()

    @property
    def prereleases(self) -> bool | None:
        return self._prereleases

    @property
    def is_empty(self) -> bool:
        return not self._bounds and not self._literals

    @property
    def is_prerelease_only(self) -> bool:
        return bool(self._bounds) and ranges_are_prerelease_only(self._bounds)

    def __repr__(self) -> str:
        text = self.to_specifier_set()
        if text is not None:
            return f"{type(self).__name__}({str(text)!r})"
        return f"{type(self).__name__}({self._bounds!r})"

    def __hash__(self) -> int:
        return hash((self._bounds, self._literals, self._prereleases))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionRange):
            return NotImplemented
        return (
            self._bounds == other._bounds
            and self._literals == other._literals
            and self._prereleases == other._prereleases
        )

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, (Version, str)):
            return False

        if isinstance(item, str) and item in self._literals:
            return True

        try:
            version = coerce_version(item)
        except (InvalidVersion, TypeError):
            return False

        if not matches_bounds_only(version, self._bounds):
            return False

        if version.is_prerelease and self._prereleases is not True:
            return ranges_are_prerelease_only(self._bounds)

        return True

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
    ) -> bool:
        if prereleases is None:
            return item in self

        if not isinstance(item, (Version, str)):
            return False

        if isinstance(item, str) and item in self._literals:
            return True

        try:
            version = coerce_version(item)
        except (InvalidVersion, TypeError):
            return False

        if not matches_bounds_only(version, self._bounds):
            return False
        return prereleases or not version.is_prerelease

    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] | None = None,
    ) -> Iterator[T]:
        if key is None:
            key = lambda value: value  # type: ignore[return-value]

        effective_prereleases = self._prereleases if prereleases is None else prereleases

        values = list(iterable)
        normal: list[T] = []
        prerelease_values: list[T] = []

        for value in values:
            candidate = key(value)
            if isinstance(candidate, str) and candidate in self._literals:
                normal.append(value)
                continue

            try:
                parsed = coerce_version(candidate)
            except (InvalidVersion, TypeError):
                continue

            if not matches_bounds_only(parsed, self._bounds):
                continue

            if parsed.is_prerelease:
                prerelease_values.append(value)
            else:
                normal.append(value)

        if effective_prereleases is True:
            yield from normal
            yield from prerelease_values
        elif normal:
            yield from normal
        elif effective_prereleases is None:
            yield from prerelease_values

    def intersection(self, other: VersionRange) -> VersionRange:
        if not isinstance(other, VersionRange):
            return NotImplemented  # type: ignore[return-value]

        literals = self._literals & other._literals
        return self._from_parts(
            intersect_ranges(self._bounds, other._bounds),
            self._combine_prereleases(other, _SetOp.INTERSECTION),
            literals,
        )

    def union(self, other: VersionRange) -> VersionRange:
        if not isinstance(other, VersionRange):
            return NotImplemented  # type: ignore[return-value]

        return self._from_parts(
            _union_ranges(self._bounds, other._bounds),
            self._combine_prereleases(other, _SetOp.UNION),
            self._literals | other._literals,
        )

    def difference(self, other: VersionRange) -> VersionRange:
        if not isinstance(other, VersionRange):
            return NotImplemented  # type: ignore[return-value]

        return self._from_parts(
            _difference_ranges(self._bounds, other._bounds),
            self._combine_prereleases(other, _SetOp.DIFFERENCE),
            self._literals - other._literals,
        )

    def complement(self) -> VersionRange:
        return self._from_parts(
            _complement_ranges(self._bounds),
            self._prereleases,
        )

    def __and__(self, other: VersionRange) -> VersionRange:
        return self.intersection(other)

    def __or__(self, other: VersionRange) -> VersionRange:
        return self.union(other)

    def __sub__(self, other: VersionRange) -> VersionRange:
        return self.difference(other)

    def __invert__(self) -> VersionRange:
        return self.complement()

    def _combine_prereleases(
        self, other: VersionRange, operation: _SetOp
    ) -> bool | None:
        left = self._prereleases
        right = other._prereleases

        if operation is _SetOp.INTERSECTION:
            if left is False or right is False:
                return False
            if left is True and right is True:
                return True
            return None

        if operation is _SetOp.UNION:
            if left is True or right is True:
                return True
            if left is False and right is False:
                return False
            return None

        return left

    def to_specifier_set(self) -> SpecifierSet | None:
        if self._specifier_set is not None:
            return self._specifier_set

        if self._literals:
            if self._bounds:
                return None
            if len(self._literals) == 1:
                return SpecifierSet("===" + next(iter(self._literals)))
            return None

        text = _bounds_to_specifiers(self._bounds)
        if text is None:
            return None

        try:
            result = SpecifierSet(text)
        except ValueError:
            return None

        result.prereleases = self._prereleases
        return result