from __future__ import annotations

import dataclasses
import logging
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from urllib.parse import unquote, urlparse

from .markers import Environment, Marker, _pep440_python_full_version, default_environment
from .specifiers import SpecifierSet
from .tags import create_compatible_tags_selector, sys_tags
from .utils import (
    NormalizedName,
    is_normalized_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from .version import Version

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator
    from typing_extensions import Self

    from .tags import Tag

_logger = logging.getLogger(__name__)

__all__ = [
    "Package",
    "PackageArchive",
    "PackageDirectory",
    "PackageSdist",
    "PackageVcs",
    "PackageWheel",
    "Pylock",
    "PylockSelectError",
    "PylockUnsupportedVersionError",
    "PylockValidationError",
    "is_valid_pylock_path",
]

_T = TypeVar("_T")
_T2 = TypeVar("_T2")

_PYLOCK_FILE_NAME_RE = re.compile(r"^pylock\.([^.]+)\.toml$")


def __dir__() -> list[str]:
    return __all__


class _FromMappingProtocol(Protocol):
    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self: ...


_FromMappingProtocolT = TypeVar("_FromMappingProtocolT", bound=_FromMappingProtocol)


def is_valid_pylock_path(path: Path) -> bool:
    return path.name == "pylock.toml" or bool(_PYLOCK_FILE_NAME_RE.match(path.name))


def _toml_key(key: str) -> str:
    return key.replace("_", "-")


def _toml_value(key: str, value: Any) -> Any:
    if isinstance(value, (Version, Marker, SpecifierSet)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if key == "environments":
            return [str(item) for item in value]
    return value


def _toml_dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
    return {
        _toml_key(key): _toml_value(key, value)
        for key, value in data
        if value is not None
    }


def _get(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, expected_type) or (
        expected_type is int and isinstance(value, bool)
    ):
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} "
            f"(expected {expected_type.__name__})",
            context=key,
        )
    return value


def _get_required(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T:
    value = _get(d, expected_type, key)
    if value is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence(
    d: Mapping[str, Any], expected_item_type: type[_T], key: str
) -> Sequence[_T] | None:
    value = _get(d, Sequence, key)  # type: ignore[type-abstract]
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} (expected Sequence)",
            context=key,
        )
    for index, item in enumerate(value):
        if not isinstance(item, expected_item_type):
            raise PylockValidationError(
                f"Unexpected type {type(item).__name__} "
                f"(expected {expected_item_type.__name__})",
                context=f"{key}[{index}]",
            )
    return cast(Sequence[_T], value)


def _get_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2 | None:
    value = _get(d, expected_type, key)
    if value is None:
        return None
    try:
        return target_type(value)
    except Exception as exc:
        raise PylockValidationError(exc, context=key) from exc


def _get_required_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2:
    value = _get_as(d, expected_type, target_type, key)
    if value is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence_as(
    d: Mapping[str, Any],
    expected_item_type: type[_T],
    target_item_type: Callable[[_T], _T2],
    key: str,
) -> list[_T2] | None:
    value = _get_sequence(d, expected_item_type, key)
    if value is None:
        return None
    result: list[_T2] = []
    try:
        for item in value:
            result.append(target_item_type(item))
    except Exception as exc:
        raise PylockValidationError(exc, context=f"{key}[{len(result)}]") from exc
    return result


def _get_object(
    d: Mapping[str, Any], target_type: type[_FromMappingProtocolT], key: str
) -> _FromMappingProtocolT | None:
    value = _get(d, Mapping, key)  # type: ignore[type-abstract]
    if value is None:
        return None
    try:
        return target_type._from_dict(value)
    except Exception as exc:
        raise PylockValidationError(exc, context=key) from exc


def _get_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> list[_FromMappingProtocolT] | None:
    value = _get_sequence(d, Mapping, key)  # type: ignore[type-abstract]
    if value is None:
        return None
    result: list[_FromMappingProtocolT] = []
    try:
        for item in value:
            result.append(target_item_type._from_dict(item))
    except Exception as exc:
        raise PylockValidationError(exc, context=f"{key}[{len(result)}]") from exc
    return result


def _get_required_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> Sequence[_FromMappingProtocolT]:
    value = _get_sequence_of_objects(d, target_item_type, key)
    if value is None:
        raise _PylockRequiredKeyError(key)
    return value


def _validate_normalized_name(name: str) -> NormalizedName:
    if not is_normalized_name(name):
        raise PylockValidationError(f"Name {name!r} is not normalized")
    return NormalizedName(name)


def _validate_path_url(path: str | None, url: str | None) -> None:
    if not path and not url:
        raise PylockValidationError("path or url must be provided")


def _path_name(path: str | None) -> str | None:
    if not path:
        return None
    if "/" in path:
        return path.rsplit("/", 1)[-1]
    if "\\" in path:
        return path.rsplit("\\", 1)[-1]
    return path


def _url_name(url: str | None) -> str | None:
    if not url:
        return None
    return unquote(urlparse(url).path.rsplit("/", 1)[-1])


def _validate_hashes(hashes: Mapping[str, Any]) -> Mapping[str, Any]:
    if not hashes:
        raise PylockValidationError("At least one hash must be provided")
    if not all(isinstance(value, str) for value in hashes.values()):
        raise PylockValidationError("Hash values must be strings")
    return hashes


def _ensure_only_keys(d: Mapping[str, Any], keys: set[str]) -> None:
    unknown = set(d).difference(keys)
    if unknown:
        names = ", ".join(repr(key) for key in sorted(unknown))
        raise PylockValidationError(f"Unexpected key(s): {names}")


class PylockValidationError(Exception):
    context: str | None = None
    message: str

    def __init__(self, cause: str | Exception, *, context: str | None = None) -> None:
        if isinstance(cause, PylockValidationError):
            self.context = (
                f"{context}.{cause.context}"
                if context and cause.context
                else context or cause.context
            )
            self.message = cause.message
        else:
            self.context = context
            self.message = str(cause)
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.context:
            return f"{self.context}: {self.message}"
        return self.message


class _PylockRequiredKeyError(PylockValidationError):
    def __init__(self, key: str) -> None:
        super().__init__(f"Missing required key {key!r}")


class PylockUnsupportedVersionError(PylockValidationError):
    pass


class PylockSelectError(Exception):
    pass


@dataclass(frozen=True)
class PackageWheel:
    name: str
    hashes: Mapping[str, str]
    url: str | None = None
    path: str | None = None
    size: int | None = None
    upload_time: datetime | None = None

    def __post_init__(self) -> None:
        _validate_path_url(self.path, self.url)
        _validate_hashes(self.hashes)
        if self.size is not None and self.size < 0:
            raise PylockValidationError("size must not be negative")
        if self.path and self.url:
            raise PylockValidationError("Only one of path or url may be provided")
        source_name = _path_name(self.path) or _url_name(self.url)
        if source_name and source_name != self.name:
            raise PylockValidationError(
                f"Wheel name {self.name!r} does not match source filename {source_name!r}"
            )
        try:
            parse_wheel_filename(self.name)
        except Exception as exc:
            raise PylockValidationError(exc, context="name") from exc

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(d, {"name", "hashes", "url", "path", "size", "upload-time"})
        return cls(
            name=_get_required(d, str, "name"),
            hashes=cast(
                Mapping[str, str],
                _validate_hashes(_get_required(d, Mapping, "hashes")),
            ),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            upload_time=_get(d, datetime, "upload-time"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)


@dataclass(frozen=True)
class PackageSdist:
    name: str
    hashes: Mapping[str, str]
    url: str | None = None
    path: str | None = None
    size: int | None = None
    upload_time: datetime | None = None

    def __post_init__(self) -> None:
        _validate_path_url(self.path, self.url)
        _validate_hashes(self.hashes)
        if self.size is not None and self.size < 0:
            raise PylockValidationError("size must not be negative")
        if self.path and self.url:
            raise PylockValidationError("Only one of path or url may be provided")
        source_name = _path_name(self.path) or _url_name(self.url)
        if source_name and source_name != self.name:
            raise PylockValidationError(
                f"Sdist name {self.name!r} does not match source filename {source_name!r}"
            )
        try:
            parse_sdist_filename(self.name)
        except Exception as exc:
            raise PylockValidationError(exc, context="name") from exc

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(d, {"name", "hashes", "url", "path", "size", "upload-time"})
        return cls(
            name=_get_required(d, str, "name"),
            hashes=cast(
                Mapping[str, str],
                _validate_hashes(_get_required(d, Mapping, "hashes")),
            ),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            upload_time=_get(d, datetime, "upload-time"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)


@dataclass(frozen=True)
class PackageVcs:
    type: str
    url: str
    commit_id: str
    requested_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.type:
            raise PylockValidationError("type must not be empty")
        if not self.url:
            raise PylockValidationError("url must not be empty")
        if not self.commit_id:
            raise PylockValidationError("commit_id must not be empty")

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(d, {"type", "url", "commit-id", "requested-revision"})
        return cls(
            type=_get_required(d, str, "type"),
            url=_get_required(d, str, "url"),
            commit_id=_get_required(d, str, "commit-id"),
            requested_revision=_get(d, str, "requested-revision"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)


@dataclass(frozen=True)
class PackageDirectory:
    path: str
    editable: bool | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise PylockValidationError("path must not be empty")

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(d, {"path", "editable"})
        return cls(
            path=_get_required(d, str, "path"),
            editable=_get(d, bool, "editable"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)


@dataclass(frozen=True)
class PackageArchive:
    hashes: Mapping[str, str]
    url: str | None = None
    path: str | None = None
    size: int | None = None
    upload_time: datetime | None = None

    def __post_init__(self) -> None:
        _validate_path_url(self.path, self.url)
        _validate_hashes(self.hashes)
        if self.path and self.url:
            raise PylockValidationError("Only one of path or url may be provided")
        if self.size is not None and self.size < 0:
            raise PylockValidationError("size must not be negative")

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(d, {"hashes", "url", "path", "size", "upload-time"})
        return cls(
            hashes=cast(
                Mapping[str, str],
                _validate_hashes(_get_required(d, Mapping, "hashes")),
            ),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            upload_time=_get(d, datetime, "upload-time"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)


@dataclass(frozen=True)
class Package:
    name: NormalizedName
    version: Version
    marker: Marker | None = None
    requires_python: SpecifierSet | None = None
    dependencies: Sequence[str] | None = None
    wheels: Sequence[PackageWheel] | None = None
    sdist: PackageSdist | None = None
    vcs: PackageVcs | None = None
    directory: PackageDirectory | None = None
    archive: PackageArchive | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_normalized_name(str(self.name)))
        if not isinstance(self.version, Version):
            object.__setattr__(self, "version", Version(str(self.version)))
        if self.marker is not None and not isinstance(self.marker, Marker):
            object.__setattr__(self, "marker", Marker(str(self.marker)))
        if self.requires_python is not None and not isinstance(
            self.requires_python, SpecifierSet
        ):
            object.__setattr__(
                self, "requires_python", SpecifierSet(str(self.requires_python))
            )
        if self.dependencies is not None:
            if not all(isinstance(item, str) for item in self.dependencies):
                raise PylockValidationError("dependencies must contain strings")
        if self.wheels is not None:
            if not all(isinstance(item, PackageWheel) for item in self.wheels):
                raise PylockValidationError("wheels must contain PackageWheel objects")
        sources = sum(
            item is not None
            for item in (self.sdist, self.vcs, self.directory, self.archive)
        )
        if sources > 1:
            raise PylockValidationError(
                "Only one of sdist, vcs, directory, or archive may be provided"
            )

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(
            d,
            {
                "name",
                "version",
                "marker",
                "requires-python",
                "dependencies",
                "wheels",
                "sdist",
                "vcs",
                "directory",
                "archive",
            },
        )
        return cls(
            name=_get_required_as(d, str, _validate_normalized_name, "name"),
            version=_get_required_as(d, str, Version, "version"),
            marker=_get_as(d, str, Marker, "marker"),
            requires_python=_get_as(d, str, SpecifierSet, "requires-python"),
            dependencies=_get_sequence(d, str, "dependencies"),
            wheels=_get_sequence_of_objects(d, PackageWheel, "wheels"),
            sdist=_get_object(d, PackageSdist, "sdist"),
            vcs=_get_object(d, PackageVcs, "vcs"),
            directory=_get_object(d, PackageDirectory, "directory"),
            archive=_get_object(d, PackageArchive, "archive"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)

    def select_wheel(
        self, tags: Collection[Tag] | None = None
    ) -> PackageWheel:
        if not self.wheels:
            raise PylockSelectError(f"No wheels are available for {self.name}")
        compatible_tags = list(sys_tags() if tags is None else tags)
        ranks = {tag: index for index, tag in enumerate(compatible_tags)}
        selected: PackageWheel | None = None
        selected_rank: int | None = None
        for wheel in self.wheels:
            try:
                _, _, _, wheel_tags = parse_wheel_filename(wheel.name)
            except Exception:
                continue
            matching = [ranks[tag] for tag in wheel_tags if tag in ranks]
            if matching:
                rank = min(matching)
                if selected_rank is None or rank < selected_rank:
                    selected = wheel
                    selected_rank = rank
        if selected is None:
            raise PylockSelectError(f"No compatible wheel found for {self.name}")
        return selected


@dataclass(frozen=True)
class Pylock:
    lock_version: Version
    requires_python: SpecifierSet
    packages: Sequence[Package]
    created_by: str | None = None
    environments: Sequence[Marker] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lock_version, Version):
            object.__setattr__(self, "lock_version", Version(str(self.lock_version)))
        if self.lock_version.major != 1:
            raise PylockUnsupportedVersionError(
                f"Unsupported lock version: {self.lock_version}"
            )
        if not isinstance(self.requires_python, SpecifierSet):
            object.__setattr__(
                self, "requires_python", SpecifierSet(str(self.requires_python))
            )
        if self.environments is not None:
            converted: list[Marker] = []
            for environment in self.environments:
                converted.append(
                    environment if isinstance(environment, Marker) else Marker(str(environment))
                )
            object.__setattr__(self, "environments", converted)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        _ensure_only_keys(
            d,
            {
                "lock-version",
                "created-by",
                "requires-python",
                "environments",
                "packages",
            },
        )
        lock_version = _get_required_as(d, str, Version, "lock-version")
        if lock_version.major != 1:
            raise PylockUnsupportedVersionError(
                f"Unsupported lock version: {lock_version}"
            )
        return cls(
            lock_version=lock_version,
            created_by=_get(d, str, "created-by"),
            requires_python=_get_required_as(
                d, str, SpecifierSet, "requires-python"
            ),
            environments=_get_sequence_as(d, str, Marker, "environments"),
            packages=_get_required_sequence_of_objects(d, Package, "packages"),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        try:
            return cls._from_dict(d)
        except PylockValidationError:
            raise
        except Exception as exc:
            raise PylockValidationError(exc) from exc

    @classmethod
    def from_file(cls, path: Path) -> Self:
        try:
            with path.open("rb") as file:
                data = tomllib.load(file)
        except Exception as exc:
            raise PylockValidationError(exc) from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)

    def select_packages(
        self, environment: Environment | None = None
    ) -> Iterator[Package]:
        current_environment = dict(default_environment())
        if environment is not None:
            current_environment.update(environment)
        current_environment["python_full_version"] = _pep440_python_full_version(
            current_environment["python_full_version"]
        )
        if self.environments is not None and not any(
            marker.evaluate(current_environment) for marker in self.environments
        ):
            raise PylockSelectError("The environment is not supported by this lock file")
        python_version = current_environment.get("python_full_version")
        if python_version and not self.requires_python.contains(
            python_version, prereleases=True
        ):
            raise PylockSelectError(
                f"Python {python_version} does not satisfy {self.requires_python}"
            )
        for package in self.packages:
            if package.marker is None or package.marker.evaluate(current_environment):
                yield package