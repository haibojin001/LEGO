from __future__ import annotations

import re
from typing import NewType, cast

from .tags import InvalidTag, Tag, UnsortedTagsError, parse_tag
from .version import InvalidVersion, Version, _TrimmedRelease

__all__ = [
    "BuildTag",
    "InvalidName",
    "InvalidSdistFilename",
    "InvalidWheelFilename",
    "NormalizedName",
    "canonicalize_name",
    "canonicalize_version",
    "is_normalized_name",
    "parse_sdist_filename",
    "parse_wheel_filename",
]


def __dir__() -> list[str]:
    return __all__


BuildTag = tuple[()] | tuple[int, str]

NormalizedName = NewType("NormalizedName", str)


class InvalidName(ValueError):
    pass


class InvalidWheelFilename(ValueError):
    pass


class InvalidSdistFilename(ValueError):
    pass


_validate_regex = re.compile(
    r"[a-z0-9]|[a-z0-9][a-z0-9._-]*[a-z0-9]",
    re.IGNORECASE | re.ASCII,
)
_normalized_regex = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)
_build_tag_regex = re.compile(r"(\d+)(.*)", re.ASCII)
_wheel_name_regex = re.compile(r"^[\w._]+\Z", re.UNICODE)


def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    if validate and _validate_regex.fullmatch(name) is None:
        raise InvalidName(f"name is invalid: {name!r}")

    normalized = name.lower().replace("_", "-").replace(".", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return cast(NormalizedName, normalized)


def is_normalized_name(name: str) -> bool:
    return _normalized_regex.fullmatch(name) is not None


def canonicalize_version(
    version: Version | str, *, strip_trailing_zero: bool = True
) -> str:
    if isinstance(version, str):
        try:
            version = Version(version)
        except InvalidVersion:
            return str(version)

    if strip_trailing_zero:
        return str(_TrimmedRelease(version))
    return str(version)


def parse_wheel_filename(
    filename: str,
    *,
    validate_order: bool = False,
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Tag]]:
    if not filename.endswith(".whl"):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (extension must be '.whl'): {filename!r}"
        )

    filename = filename[:-4]
    dash_count = filename.count("-")
    if dash_count not in (4, 5):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (wrong number of parts): {filename!r}"
        )

    parts = filename.split("-", dash_count - 2)
    name_part = parts[0]
    if "__" in name_part or _wheel_name_regex.match(name_part) is None:
        raise InvalidWheelFilename(f"Invalid project name: {filename!r}")

    name = canonicalize_name(name_part)

    try:
        version = Version(parts[1])
    except InvalidVersion as exc:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid version): {filename!r}"
        ) from exc

    if dash_count == 5:
        build_part = parts[2]
        match = _build_tag_regex.match(build_part)
        if match is None:
            raise InvalidWheelFilename(
                f"Invalid build number: {filename!r}"
            )
        build: BuildTag = (int(match.group(1)), match.group(2))
    else:
        build = ()

    try:
        tags = parse_tag(
            parts[-3],
            parts[-2],
            parts[-1],
            validate_order=validate_order,
        )
    except (InvalidTag, UnsortedTagsError) as exc:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid tag set): {filename!r}"
        ) from exc

    return name, version, build, tags


def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    if filename.endswith(".tar.gz"):
        stem = filename[:-7]
    elif filename.endswith(".zip"):
        stem = filename[:-4]
    else:
        raise InvalidSdistFilename(
            "Invalid sdist filename (extension must be '.tar.gz' or '.zip'): "
            f"{filename!r}"
        )

    if "-" not in stem:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (invalid version): {filename!r}"
        )

    name_part, version_part = stem.rsplit("-", 1)
    name = canonicalize_name(name_part)

    try:
        version = Version(version_part)
    except InvalidVersion as exc:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (invalid version): {filename!r}"
        ) from exc

    return name, version