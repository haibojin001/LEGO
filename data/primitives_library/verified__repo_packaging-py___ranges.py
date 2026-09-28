from __future__ import annotations

import enum
import functools
from typing import TYPE_CHECKING, Any, Final

from .version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    _BoundaryOrderSuffix = tuple[int, int, int, int | float, int, int]
    _BoundaryOrderKey = tuple[int, tuple[int, ...], _BoundaryOrderSuffix, float]
    _VersionOrBoundary = Version | BoundaryVersion | None
    Range = tuple[LowerBound, UpperBound]

__all__ = [
    "FULL_RANGE",
    "bounds_for_spec",
    "coerce_version",
    "filter_by_ranges",
    "intersect_ranges",
    "intersect_specifier_bounds",
    "least_version_above",
    "matches_bounds_only",
    "range_is_empty",
    "ranges_are_prerelease_only",
    "resolve_prereleases",
    "standard_ranges",
    "wildcard_ranges",
]

MIN_VERSION: Final[Version] = Version("0.dev0")
MIN_RELEASE: Final[Version] = Version("0")
_BOUNDARY_INF: Final[float] = float("inf")


class BoundaryKind(enum.Enum):
    AFTER_LOCALS = enum.auto()
    AFTER_POSTS = enum.auto()


@functools.total_ordering
class BoundaryVersion:
    __slots__ = (
        "_cached_dev",
        "_cached_epoch",
        "_cached_post",
        "_cached_pre",
        "_cached_trimmed_release",
        "kind",
        "version",
    )

    def __init__(self, version: Version, kind: BoundaryKind) -> None:
        self.version = version
        self.kind = kind
        self._cached_trimmed_release = trim_release(version.release)
        self._cached_epoch = version.epoch
        self._cached_pre = version.pre
        self._cached_post = version.post
        self._cached_dev = version.dev

    def _is_family(self, other: Version) -> bool:
        if other.epoch != self._cached_epoch:
            return False

        release = other.release
        expected = self._cached_trimmed_release
        length = len(expected)

        if len(release) < length or release[:length] != expected:
            return False

        if any(item != 0 for item in release[length:]):
            return False

        if other.pre != self._cached_pre:
            return False

        if self.kind == BoundaryKind.AFTER_LOCALS:
            return other.post == self._cached_post and other.dev == self._cached_dev

        return other.dev == self._cached_dev or other.post is not None

    def _order_key(self) -> _BoundaryOrderKey:
        key = self.version._key
        suffix: _BoundaryOrderSuffix = key[2]

        if self.kind == BoundaryKind.AFTER_POSTS:
            suffix = (suffix[0], suffix[1], 1, _BOUNDARY_INF, 1, 0)

        return key[0], key[1], suffix, _BOUNDARY_INF

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoundaryVersion):
            return self._order_key() == other._order_key()
        return NotImplemented

    def __lt__(self, other: BoundaryVersion | Version) -> bool:
        if isinstance(other, BoundaryVersion):
            return self._order_key() < other._order_key()

        if not (self.version < other):
            return False

        return not self._is_family(other)

    def __gt__(self, other: BoundaryVersion | Version) -> bool:
        if isinstance(other, BoundaryVersion):
            return self._order_key() > other._order_key()

        if self.version >= other:
            return True

        return self._is_family(other)

    def __hash__(self) -> int:
        return hash(self._order_key())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.version!r}, {self.kind.name})"


def trim_release(release: tuple[int, ...]) -> tuple[int, ...]:
    index = len(release)
    while index > 1 and release[index - 1] == 0:
        index -= 1
    return release[:index]


def _make_above_after_locals(version: Version) -> Callable[[Version], bool]:
    boundary = BoundaryVersion(version, BoundaryKind.AFTER_LOCALS)

    def above(candidate: Version) -> bool:
        return boundary < candidate

    return above


def _make_above_after_posts(version: Version) -> Callable[[Version], bool]:
    boundary = BoundaryVersion(version, BoundaryKind.AFTER_POSTS)

    def above(candidate: Version) -> bool:
        return boundary < candidate

    return above


def _make_below_boundary(boundary: BoundaryVersion) -> Callable[[Version], bool]:
    def below(candidate: Version) -> bool:
        return boundary > candidate

    return below


@functools.total_ordering
class LowerBound:
    __slots__ = ("_above", "inclusive", "version")

    def __init__(
        self, version: Version | BoundaryVersion | None, inclusive: bool
    ) -> None:
        if version is None:
            inclusive = False

        self.version = version
        self.inclusive = inclusive

        if version is None:
            self._above: Callable[[Version], bool] | None = None
        elif isinstance(version, BoundaryVersion):
            if version.kind == BoundaryKind.AFTER_POSTS:
                self._above = _make_above_after_posts(version.version)
            else:
                self._above = _make_above_after_locals(version.version)
        elif inclusive:
            self._above = version.__le__
        else:
            self._above = version.__lt__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LowerBound):
            return NotImplemented
        return self.version == other.version and self.inclusive == other.inclusive

    def __lt__(self, other: LowerBound) -> bool:
        if not isinstance(other, LowerBound):
            return NotImplemented

        if self.version is None:
            return other.version is not None
        if other.version is None:
            return False

        if self.version != other.version:
            return self.version < other.version

        return self.inclusive and not other.inclusive

    def __hash__(self) -> int:
        return hash((self.version, self.inclusive))

    def __repr__(self) -> str:
        bracket = "[" if self.inclusive else "("
        return f"<{type(self).__name__} {bracket}{self.version!r}>"


@functools.total_ordering
class UpperBound:
    __slots__ = ("_below", "inclusive", "version")

    def __init__(
        self, version: Version | BoundaryVersion | None, inclusive: bool
    ) -> None:
        if version is None:
            inclusive = False

        self.version = version
        self.inclusive = inclusive

        if version is None:
            self._below: Callable[[Version], bool] | None = None
        elif isinstance(version, BoundaryVersion):
            self._below = _make_below_boundary(version)
        elif inclusive:
            self._below = version.__ge__
        else:
            self._below = version.__gt__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UpperBound):
            return NotImplemented
        return self.version == other.version and self.inclusive == other.inclusive

    def __lt__(self, other: UpperBound) -> bool:
        if not isinstance(other, UpperBound):
            return NotImplemented

        if self.version is None:
            return False
        if other.version is None:
            return True

        if self.version != other.version:
            return self.version < other.version

        return not self.inclusive and other.inclusive

    def __hash__(self) -> int:
        return hash((self.version, self.inclusive))

    def __repr__(self) -> str:
        bracket = "]" if self.inclusive else ")"
        return f"<{type(self).__name__} {self.version!r}{bracket}>"


FULL_RANGE: Final[tuple[tuple[LowerBound, UpperBound], ...]] = (
    (LowerBound(None, False), UpperBound(None, False)),
)


@functools.lru_cache(maxsize=4096)
def _parse_version(value: str) -> Version:
    return Version(value)


def coerce_version(version: Version | str) -> Version:
    if isinstance(version, Version):
        return version
    return _parse_version(version)


def _as_ranges(
    value: Any,
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    if value is None:
        return ()

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], LowerBound)
        and isinstance(value[1], UpperBound)
    ):
        return (value,)

    return tuple(value)


def _point_compare(
    left: Version | BoundaryVersion, right: Version | BoundaryVersion
) -> int:
    if left == right:
        return 0
    if left < right:
        return -1
    return 1


def range_is_empty(bounds: tuple[LowerBound, UpperBound]) -> bool:
    lower, upper = bounds

    if lower.version is None or upper.version is None:
        return False

    comparison = _point_compare(lower.version, upper.version)

    if comparison > 0:
        return True
    if comparison < 0:
        return False

    if isinstance(lower.version, BoundaryVersion):
        return True

    return not (lower.inclusive and upper.inclusive)


def _range_sort_key(bounds: tuple[LowerBound, UpperBound]) -> LowerBound:
    return bounds[0]


def _normalise_ranges(
    ranges: Iterable[tuple[LowerBound, UpperBound]],
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    ordered = sorted(
        (item for item in ranges if not range_is_empty(item)),
        key=_range_sort_key,
    )

    if not ordered:
        return ()

    result: list[tuple[LowerBound, UpperBound]] = []

    for lower, upper in ordered:
        if not result:
            result.append((lower, upper))
            continue

        previous_lower, previous_upper = result[-1]

        if previous_upper.version is None:
            continue

        overlaps = False
        if lower.version is None:
            overlaps = True
        else:
            comparison = _point_compare(previous_upper.version, lower.version)
            if comparison > 0:
                overlaps = True
            elif comparison == 0:
                overlaps = previous_upper.inclusive or lower.inclusive
                if isinstance(previous_upper.version, BoundaryVersion):
                    overlaps = True

        if not overlaps:
            result.append((lower, upper))
            continue

        if upper > previous_upper:
            result[-1] = (previous_lower, upper)

    return tuple(result)


def intersect_ranges(
    left: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
    right: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    left_ranges = _as_ranges(left)
    right_ranges = _as_ranges(right)
    intersections: list[tuple[LowerBound, UpperBound]] = []

    for left_lower, left_upper in left_ranges:
        for right_lower, right_upper in right_ranges:
            lower = max(left_lower, right_lower)
            upper = min(left_upper, right_upper)
            candidate = (lower, upper)
            if not range_is_empty(candidate):
                intersections.append(candidate)

    return _normalise_ranges(intersections)


def matches_bounds_only(
    version: Version | str,
    ranges: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
) -> bool:
    candidate = coerce_version(version)

    for lower, upper in _as_ranges(ranges):
        if lower._above is not None and not lower._above(candidate):
            continue
        if upper._below is not None and not upper._below(candidate):
            continue
        return True

    return False


def filter_by_ranges(
    versions: Iterable[Any],
    ranges: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
    prereleases: bool | None = None,
) -> Iterator[Any]:
    allowed_prereleases = (
        resolve_prereleases(ranges, prereleases)
        if prereleases is not None
        else True
    )

    for value in versions:
        try:
            parsed = coerce_version(value)
        except (InvalidVersion, TypeError):
            continue

        if not allowed_prereleases and parsed.is_prerelease:
            continue

        if matches_bounds_only(parsed, ranges):
            yield value


def _version_string(epoch: int, release: tuple[int, ...], suffix: str = "") -> str:
    prefix = f"{epoch}!" if epoch else ""
    return f"{prefix}{'.'.join(str(part) for part in release)}{suffix}"


def _release_ceiling(version: Version, components: int | None = None) -> Version:
    release = version.release
    if components is not None:
        release = release[:components]

    release = tuple(release)
    bumped = release[:-1] + (release[-1] + 1,)
    return Version(_version_string(version.epoch, bumped, ".dev0"))


def _wildcard_bounds(
    version_text: str,
) -> tuple[LowerBound, UpperBound]:
    prefix = version_text[:-2]
    parsed = coerce_version(prefix)

    release = parsed.release
    lower_version = Version(_version_string(parsed.epoch, release, ".dev0"))
    upper_version = _release_ceiling(parsed)

    return LowerBound(lower_version, True), UpperBound(upper_version, False)


def wildcard_ranges(
    specifier: Any,
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    operator = getattr(specifier, "operator", None)
    version = getattr(specifier, "version", None)

    if version is None and isinstance(specifier, tuple):
        operator, version = specifier

    if not isinstance(version, str) or not version.endswith(".*"):
        return standard_ranges(specifier)

    lower, upper = _wildcard_bounds(version)

    if operator in ("==", "==="):
        return ((lower, upper),)

    if operator == "!=":
        return _normalise_ranges(
            (
                (LowerBound(None, False), UpperBound(lower.version, False)),
                (LowerBound(upper.version, True), UpperBound(None, False)),
            )
        )

    return ()


def standard_ranges(
    specifier: Any,
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    operator = getattr(specifier, "operator", None)
    version_text = getattr(specifier, "version", None)

    if version_text is None and isinstance(specifier, tuple):
        operator, version_text = specifier

    if isinstance(version_text, str) and version_text.endswith(".*"):
        return wildcard_ranges(specifier)

    try:
        version = coerce_version(version_text)
    except (InvalidVersion, TypeError):
        return ()

    if operator in ("==", "==="):
        if version.local is None:
            endpoint = BoundaryVersion(version, BoundaryKind.AFTER_LOCALS)
            return ((LowerBound(version, True), UpperBound(endpoint, True)),)
        return ((LowerBound(version, True), UpperBound(version, True)),)

    if operator == "!=":
        if version.local is None:
            endpoint = BoundaryVersion(version, BoundaryKind.AFTER_LOCALS)
            return (
                (LowerBound(None, False), UpperBound(version, False)),
                (LowerBound(endpoint, True), UpperBound(None, False)),
            )
        return (
            (LowerBound(None, False), UpperBound(version, False)),
            (LowerBound(version, False), UpperBound(None, False)),
        )

    if operator == "<":
        return ((LowerBound(None, False), UpperBound(version, False)),)

    if operator == "<=":
        endpoint = (
            BoundaryVersion(version, BoundaryKind.AFTER_LOCALS)
            if version.local is None
            else version
        )
        return ((LowerBound(None, False), UpperBound(endpoint, True)),)

    if operator == ">":
        endpoint = BoundaryVersion(version, BoundaryKind.AFTER_POSTS)
        return ((LowerBound(endpoint, True), UpperBound(None, False)),)

    if operator == ">=":
        return ((LowerBound(version, True), UpperBound(None, False)),)

    if operator == "~=":
        upper = _release_ceiling(version, len(version.release) - 1)
        return ((LowerBound(version, True), UpperBound(upper, False)),)

    return ()


def bounds_for_spec(
    specifier: Any,
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    version = getattr(specifier, "version", None)
    if isinstance(version, str) and version.endswith(".*"):
        return wildcard_ranges(specifier)
    return standard_ranges(specifier)


def intersect_specifier_bounds(
    specifiers: Iterable[Any],
    ranges: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound] = FULL_RANGE,
) -> tuple[tuple[LowerBound, UpperBound], ...]:
    result = _as_ranges(ranges)

    for specifier in specifiers:
        result = intersect_ranges(result, bounds_for_spec(specifier))
        if not result:
            break

    return result


def _next_release_after(version: Version) -> Version:
    release = version.release
    return Version(_version_string(version.epoch, release + (1,)))


def least_version_above(lower: LowerBound) -> Version:
    version = lower.version

    if version is None:
        return MIN_RELEASE

    if isinstance(version, BoundaryVersion):
        base = version.version

        if version.kind == BoundaryKind.AFTER_LOCALS:
            if base.post is None:
                return Version(
                    _version_string(
                        base.epoch,
                        base.release,
                        ".post0",
                    )
                )
            return Version(
                _version_string(
                    base.epoch,
                    base.release,
                    f".post{base.post + 1}",
                )
            )

        return _next_release_after(base)

    if version.is_prerelease or version.dev is not None:
        return Version(_version_string(version.epoch, version.release))

    if not lower.inclusive:
        return _next_release_after(version)

    return version


def ranges_are_prerelease_only(
    ranges: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
) -> bool:
    checked = False

    for lower, upper in _as_ranges(ranges):
        if range_is_empty((lower, upper)):
            continue

        checked = True
        candidate = least_version_above(lower)

        if upper._below is None or upper._below(candidate):
            return False

    return checked


def resolve_prereleases(
    ranges: Sequence[tuple[LowerBound, UpperBound]]
    | tuple[LowerBound, UpperBound],
    prereleases: bool | None,
) -> bool:
    if prereleases is not None:
        return prereleases

    return ranges_are_prerelease_only(ranges)