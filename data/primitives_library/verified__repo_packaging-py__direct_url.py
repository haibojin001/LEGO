from __future__ import annotations

import dataclasses
import re
import urllib.parse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    import sys
    from collections.abc import Collection
    from urllib.parse import SplitResult

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = [
    "ArchiveInfo",
    "DirInfo",
    "DirectUrl",
    "DirectUrlValidationError",
    "VcsInfo",
]


def __dir__() -> list[str]:
    return __all__


_T = TypeVar("_T")


class _FromMappingProtocol(Protocol):
    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self: ...


_FromMappingProtocolT = TypeVar("_FromMappingProtocolT", bound=_FromMappingProtocol)


def _json_dict_factory(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {name: value for name, value in items if value is not None}


def _get(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise DirectUrlValidationError(
            f"Unexpected type {type(value).__name__} "
            f"(expected {expected_type.__name__})",
            context=key,
        )
    return value


def _get_required(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T:
    value = _get(d, expected_type, key)
    if value is None:
        raise _DirectUrlRequiredKeyError(key)
    return value


def _get_object(
    d: Mapping[str, Any], target_type: type[_FromMappingProtocolT], key: str
) -> _FromMappingProtocolT | None:
    value = _get(d, Mapping, key)  # type: ignore[type-abstract]
    if value is None:
        return None
    try:
        return target_type._from_dict(value)
    except Exception as exc:
        raise DirectUrlValidationError(exc, context=key) from exc


_PEP610_USER_PASS_ENV_VARS_REGEX = re.compile(
    r"^\$\{[A-Za-z0-9-_]+\}(:\$\{[A-Za-z0-9-_]+\})?$"
)


def _strip_auth_from_netloc(netloc: str, safe_user_passwords: Collection[str]) -> str:
    if "@" not in netloc:
        return netloc

    credentials, host = netloc.rsplit("@", 1)
    if credentials in safe_user_passwords:
        return netloc
    if _PEP610_USER_PASS_ENV_VARS_REGEX.match(credentials):
        return netloc
    return host


def _strip_url(url: str, safe_user_passwords: Collection[str]) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            _strip_auth_from_netloc(parsed.netloc, safe_user_passwords),
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _file_url_has_absolute_path(parsed_url: SplitResult) -> bool:
    return parsed_url.path.startswith("/")


class DirectUrlValidationError(Exception):
    context: str | None = None
    message: str

    def __init__(
        self,
        cause: str | Exception,
        *,
        context: str | None = None,
    ) -> None:
        if isinstance(cause, DirectUrlValidationError):
            if cause.context:
                self.context = (
                    f"{context}.{cause.context}" if context else cause.context
                )
            else:
                self.context = context
            self.message = cause.message
        else:
            self.context = context
            self.message = str(cause)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} in {self.context!r}"
        return self.message


class _DirectUrlRequiredKeyError(DirectUrlValidationError):
    def __init__(self, key: str) -> None:
        super().__init__("Missing required value", context=key)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class VcsInfo:
    vcs: str
    commit_id: str
    requested_revision: str | None = None

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(
            vcs=_get_required(d, str, "vcs"),
            commit_id=_get_required(d, str, "commit_id"),
            requested_revision=_get(d, str, "requested_revision"),
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ArchiveInfo:
    hashes: Mapping[str, str] | None = None

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        hashes = _get(d, Mapping, "hashes")  # type: ignore[type-abstract]
        if hashes is not None and not all(
            isinstance(value, str) for value in hashes.values()
        ):
            raise DirectUrlValidationError(
                "Hash values must be strings",
                context="hashes",
            )

        legacy_hash = _get(d, str, "hash")
        if legacy_hash is not None:
            if "=" not in legacy_hash:
                raise DirectUrlValidationError(
                    "Invalid hash format (expected '<algorithm>=<hash>')",
                    context="hash",
                )

            algorithm, value = legacy_hash.split("=", 1)
            if hashes is None:
                hashes = {algorithm: value}
            else:
                if algorithm not in hashes:
                    raise DirectUrlValidationError(
                        f"Algorithm {algorithm!r} used in hash field "
                        f"is not present in hashes field",
                        context="hashes",
                    )
                if hashes[algorithm] != value:
                    raise DirectUrlValidationError(
                        f"Algorithm {algorithm!r} used in hash field "
                        f"has different value in hashes field",
                        context="hash",
                    )

        return cls(hashes=hashes)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DirInfo:
    editable: bool | None = None

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(editable=_get(d, bool, "editable"))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DirectUrl:
    url: str
    archive_info: ArchiveInfo | None = None
    vcs_info: VcsInfo | None = None
    dir_info: DirInfo | None = None
    subdirectory: str | None = None

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        result = cls(
            url=_get_required(d, str, "url"),
            archive_info=_get_object(d, ArchiveInfo, "archive_info"),
            vcs_info=_get_object(d, VcsInfo, "vcs_info"),
            dir_info=_get_object(d, DirInfo, "dir_info"),
            subdirectory=_get(d, str, "subdirectory"),
        )

        if (
            bool(result.vcs_info)
            + bool(result.archive_info)
            + bool(result.dir_info)
        ) != 1:
            raise DirectUrlValidationError(
                "Exactly one of vcs_info, archive_info, dir_info must be present"
            )

        if result.dir_info is not None:
            parsed = urllib.parse.urlsplit(result.url)
            if parsed.scheme != "file":
                raise DirectUrlValidationError(
                    "URL scheme must be file:// when dir_info is present",
                    context="url",
                )
            if not _file_url_has_absolute_path(parsed):
                raise DirectUrlValidationError(
                    "File URL must be absolute when dir_info is present",
                    context="url",
                )

        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], /) -> Self:
        return cls._from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self, dict_factory=_json_dict_factory)
        result["url"] = self.redacted_url
        return result

    @property
    def redacted_url(self) -> str:
        safe_user_passwords = (
            (self.vcs_info.vcs,) if self.vcs_info is not None else ()
        )
        return _strip_url(self.url, safe_user_passwords)