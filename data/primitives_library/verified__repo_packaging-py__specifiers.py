from __future__ import annotations

import abc
import re
import typing
from typing import Any, Callable, Iterable, Iterator, TypeVar, overload

from .utils import canonicalize_version
from .version import Version

__all__ = ["BaseSpecifier", "InvalidSpecifier", "Specifier", "SpecifierSet"]


def __dir__() -> list[str]:
    return __all__


T = TypeVar("T")
UnparsedVersion = Version | str
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)


class InvalidSpecifier(ValueError):
    """Raised when a version specifier cannot be parsed."""


def _validate_pre(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError("prereleases must be a bool or None")


def _as_version(value: UnparsedVersion) -> Version:
    if isinstance(value, Version):
        return value
    return Version(value)


def _without_local(version: Version) -> Version:
    if version.local is None:
        return version
    return Version(version.public)


class BaseSpecifier(metaclass=abc.ABCMeta):
    __slots__ = ()
    __match_args__ = ("_str",)

    @property
    def _str(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def prereleases(self) -> bool | None:
        raise NotImplementedError

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def contains(self, item: UnparsedVersion, prereleases: bool | None = None) -> bool:
        raise NotImplementedError

    @overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    @abc.abstractmethod
    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        raise NotImplementedError


class Specifier(BaseSpecifier):
    __slots__ = ("_prereleases", "_ranges", "_spec", "_spec_version")

    _operator_re = re.compile(r"(===|~=|==|!=|<=|>=|<|>)")
    _release_re = re.compile(r"^(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*$")

    def __init__(self, spec: str = "", prereleases: bool | None = None) -> None:
        if not isinstance(spec, str):
            raise InvalidSpecifier(f"Invalid specifier: {spec!r}")

        original = spec
        spec = spec.strip()
        match = re.fullmatch(r"\s*(===|~=|==|!=|<=|>=|<|>)\s*([^\s;)]*)\s*", spec)

        if match is None:
            raise InvalidSpecifier(f"Invalid specifier: {original!r}")

        operator, version = match.groups()
        if not version:
            raise InvalidSpecifier(f"Invalid specifier: {original!r}")

        if operator == "===":
            self._spec = (operator, version)
        else:
            wildcard = version.endswith(".*")

            if wildcard:
                if operator not in ("==", "!="):
                    raise InvalidSpecifier(f"Invalid specifier: {original!r}")
                prefix = version[:-2]
                if not self._release_re.fullmatch(prefix):
                    raise InvalidSpecifier(f"Invalid specifier: {original!r}")
                try:
                    Version(prefix + ".0")
                except Exception:
                    raise InvalidSpecifier(f"Invalid specifier: {original!r}") from None
            else:
                try:
                    parsed = Version(version)
                except Exception:
                    raise InvalidSpecifier(f"Invalid specifier: {original!r}") from None

                if operator == "~=":
                    if parsed.local is not None or len(parsed.release) < 2:
                        raise InvalidSpecifier(f"Invalid specifier: {original!r}")
                elif operator not in ("==", "!=") and parsed.local is not None:
                    raise InvalidSpecifier(f"Invalid specifier: {original!r}")

            self._spec = (operator, version)

        self._prereleases = _validate_pre(prereleases)
        self._ranges = None
        self._spec_version = None

    @property
    def operator(self) -> str:
        return self._spec[0]

    @property
    def version(self) -> str:
        return self._spec[1]

    @property
    def _canonical_spec(self) -> tuple[str, str]:
        operator, version = self._spec
        if operator == "===":
            return operator, version

        if version.endswith(".*"):
            prefix = version[:-2]
            try:
                normalized = canonicalize_version(
                    prefix + ".0", strip_trailing_zero=False
                )
                if normalized.endswith(".0"):
                    normalized = normalized[:-2]
                return operator, normalized + ".*"
            except Exception:
                return operator, version.lower()

        try:
            return operator, canonicalize_version(version, strip_trailing_zero=False)
        except Exception:
            return operator, version.lower()

    def __str__(self) -> str:
        return "".join(self._spec)

    def __repr__(self) -> str:
        return f"<Specifier({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = self.__class__(other)
            except InvalidSpecifier:
                return False
        if not isinstance(other, Specifier):
            return NotImplemented
        return self._canonical_spec == other._canonical_spec

    def __contains__(self, item: UnparsedVersion) -> bool:
        return self.contains(item)

    def __copy__(self) -> Specifier:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Specifier:
        return self

    def __reduce__(self) -> tuple[Any, tuple[str, bool | None]]:
        return self.__class__, (str(self), self._prereleases)

    @property
    def prereleases(self) -> bool | None:
        if self._prereleases is not None:
            return self._prereleases

        operator, version = self._spec
        if operator == "===" or version.endswith(".*"):
            return False

        try:
            return Version(version).is_prerelease
        except Exception:
            return False

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        self._prereleases = _validate_pre(value)

    def _require_spec_version(self, version: str | None = None) -> Version:
        target = self._spec[1] if version is None else version
        if self._spec_version is None or version is not None:
            parsed = Version(target)
            if version is None:
                self._spec_version = parsed
            return parsed
        return self._spec_version

    def _match(self, prospective: Version) -> bool:
        operator, spec_string = self._spec

        if operator == "===":
            return str(prospective).lower() == spec_string.lower()

        if spec_string.endswith(".*"):
            prefix = Version(spec_string[:-2] + ".0")
            release_prefix = prefix.release[:-1]
            same_epoch = prospective.epoch == prefix.epoch
            equal_prefix = prospective.release[: len(release_prefix)] == release_prefix
            matched = same_epoch and equal_prefix
            return not matched if operator == "!=" else matched

        spec = self._require_spec_version()

        if operator == "==":
            if spec.local is None:
                return _without_local(prospective) == spec
            return prospective == spec

        if operator == "!=":
            if spec.local is None:
                return _without_local(prospective) != spec
            return prospective != spec

        if operator == ">=":
            return prospective >= spec

        if operator == "<=":
            return _without_local(prospective) <= spec

        if operator == "<":
            candidate = _without_local(prospective)
            if not candidate < spec:
                return False
            return not (
                candidate.is_prerelease and candidate.base_version == spec.base_version
            )

        if operator == ">":
            if not prospective > spec:
                return False
            return not (
                prospective.is_postrelease
                and prospective.base_version == spec.base_version
            )

        if operator == "~=":
            if prospective < spec:
                return False
            prefix = spec.release[:-1]
            return (
                prospective.epoch == spec.epoch
                and prospective.release[: len(prefix)] == prefix
            )

        return False

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
    ) -> bool:
        prereleases = _validate_pre(prereleases)
        prospective = _as_version(item)

        if prereleases is None:
            prereleases = self.prereleases

        if prospective.is_prerelease and not prereleases:
            return False

        return self._match(prospective)

    @overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        prereleases = _validate_pre(prereleases)
        if key is None:
            key = lambda value: value

        allow = self.prereleases if prereleases is None else prereleases
        stored_prereleases: list[Any] = []
        yielded = False

        for item in iterable:
            version = _as_version(key(item))
            if version.is_prerelease and not allow:
                if self._match(version):
                    stored_prereleases.append(item)
                continue

            if self._match(version):
                yielded = True
                yield item

        if not yielded and not allow:
            yield from stored_prereleases


class SpecifierSet(BaseSpecifier):
    __slots__ = ("_prereleases", "_specs", "_ranges")

    def __init__(self, specifiers: str = "", prereleases: bool | None = None) -> None:
        if not isinstance(specifiers, str):
            raise InvalidSpecifier(f"Invalid specifier: {specifiers!r}")

        split = [part.strip() for part in specifiers.split(",")]
        if specifiers.strip():
            if any(not part for part in split):
                raise InvalidSpecifier(f"Invalid specifier: {specifiers!r}")
            self._specs = frozenset(Specifier(part) for part in split)
        else:
            self._specs = frozenset()

        self._prereleases = _validate_pre(prereleases)
        self._ranges = None

    def __str__(self) -> str:
        return ",".join(sorted(str(specifier) for specifier in self._specs))

    def __repr__(self) -> str:
        return f"<SpecifierSet({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(self._specs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = self.__class__(other)
            except InvalidSpecifier:
                return False
        if not isinstance(other, SpecifierSet):
            return NotImplemented
        return self._specs == other._specs

    def __contains__(self, item: UnparsedVersion) -> bool:
        return self.contains(item)

    def __iter__(self) -> Iterator[Specifier]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __copy__(self) -> SpecifierSet:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> SpecifierSet:
        return self

    def __reduce__(self) -> tuple[Any, tuple[str, bool | None]]:
        return self.__class__, (str(self), self._prereleases)

    @property
    def prereleases(self) -> bool | None:
        if self._prereleases is not None:
            return self._prereleases
        return any(specifier.prereleases for specifier in self._specs)

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        self._prereleases = _validate_pre(value)

    def __and__(self, other: object) -> SpecifierSet:
        if not isinstance(other, SpecifierSet):
            return NotImplemented

        if self._prereleases is None:
            prereleases = other._prereleases
        elif other._prereleases is None:
            prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease overrides."
            )

        combined = self._specs | other._specs
        return self.__class__(",".join(str(specifier) for specifier in combined), prereleases)

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
    ) -> bool:
        prereleases = _validate_pre(prereleases)
        prospective = _as_version(item)

        if prereleases is None:
            prereleases = self.prereleases

        if prospective.is_prerelease and not prereleases:
            return False

        return all(specifier._match(prospective) for specifier in self._specs)

    @overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        prereleases = _validate_pre(prereleases)
        if key is None:
            key = lambda value: value

        allow = self.prereleases if prereleases is None else prereleases
        stored_prereleases: list[Any] = []
        yielded = False

        for item in iterable:
            version = _as_version(key(item))
            matches = all(specifier._match(version) for specifier in self._specs)
            if not matches:
                continue

            if version.is_prerelease and not allow:
                stored_prereleases.append(item)
                continue

            yielded = True
            yield item

        if not yielded and not allow:
            yield from stored_prereleases