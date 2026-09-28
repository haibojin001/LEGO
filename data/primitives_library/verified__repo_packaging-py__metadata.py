from __future__ import annotations

import email.header
import email.message
import email.parser
import email.policy
import keyword
import pathlib
import re
import typing
from typing import Any, Generic, Literal, TypedDict, cast

from . import licenses, requirements, specifiers, utils
from . import version as version_module
from .errors import ExceptionGroup, _ErrorCollector

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from .licenses import NormalizedLicenseExpression
    from .version import Version

T = typing.TypeVar("T")

__all__ = [
    "ExceptionGroup",
    "InvalidMetadata",
    "Metadata",
    "RFC822Message",
    "RFC822Policy",
    "RawMetadata",
    "parse_email",
]


def __dir__() -> list[str]:
    return __all__


class InvalidMetadata(ValueError):
    """A metadata field contains invalid data."""

    field: str

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)

    def __reduce__(self) -> tuple[type[InvalidMetadata], tuple[str, str]]:
        return (self.__class__, (self.field, self.args[0]))


class RawMetadata(TypedDict, total=False):
    metadata_version: str
    name: str
    version: str
    platforms: list[str]
    summary: str
    description: str
    keywords: list[str]
    home_page: str
    author: str
    author_email: str
    license: str
    supported_platforms: list[str]
    download_url: str
    classifiers: list[str]
    requires: list[str]
    provides: list[str]
    obsoletes: list[str]
    maintainer: str
    maintainer_email: str
    requires_dist: list[str]
    provides_dist: list[str]
    obsoletes_dist: list[str]
    requires_python: str
    requires_external: list[str]
    project_urls: dict[str, str]
    description_content_type: str
    provides_extra: list[str]
    dynamic: list[str]
    license_expression: str
    license_files: list[str]
    import_names: list[str]
    import_namespaces: list[str]


_STRING_FIELDS = {
    "author",
    "author_email",
    "description",
    "description_content_type",
    "download_url",
    "home_page",
    "license",
    "license_expression",
    "maintainer",
    "maintainer_email",
    "metadata_version",
    "name",
    "requires_python",
    "summary",
    "version",
}

_LIST_FIELDS = {
    "classifiers",
    "dynamic",
    "license_files",
    "obsoletes",
    "obsoletes_dist",
    "platforms",
    "provides",
    "provides_dist",
    "provides_extra",
    "requires",
    "requires_dist",
    "requires_external",
    "supported_platforms",
    "import_names",
    "import_namespaces",
}

_DICT_FIELDS = {"project_urls"}

_EMAIL_TO_RAW_MAPPING = {
    "author": "author",
    "author-email": "author_email",
    "classifier": "classifiers",
    "description": "description",
    "description-content-type": "description_content_type",
    "download-url": "download_url",
    "dynamic": "dynamic",
    "home-page": "home_page",
    "keywords": "keywords",
    "license": "license",
    "license-expression": "license_expression",
    "license-file": "license_files",
    "maintainer": "maintainer",
    "maintainer-email": "maintainer_email",
    "metadata-version": "metadata_version",
    "name": "name",
    "obsoletes": "obsoletes",
    "obsoletes-dist": "obsoletes_dist",
    "platform": "platforms",
    "project-url": "project_urls",
    "provides": "provides",
    "provides-dist": "provides_dist",
    "provides-extra": "provides_extra",
    "requires": "requires",
    "requires-dist": "requires_dist",
    "requires-external": "requires_external",
    "requires-python": "requires_python",
    "summary": "summary",
    "supported-platform": "supported_platforms",
    "version": "version",
    "import-name": "import_names",
    "import-namespace": "import_namespaces",
}

_RAW_TO_EMAIL_MAPPING = {
    value: key.title().replace("-", "-") for key, value in _EMAIL_TO_RAW_MAPPING.items()
}
_RAW_TO_EMAIL_MAPPING.update(
    {
        "author_email": "Author-email",
        "description_content_type": "Description-Content-Type",
        "download_url": "Download-URL",
        "home_page": "Home-page",
        "license_expression": "License-Expression",
        "license_files": "License-File",
        "maintainer_email": "Maintainer-email",
        "metadata_version": "Metadata-Version",
        "obsoletes_dist": "Obsoletes-Dist",
        "project_urls": "Project-URL",
        "provides_dist": "Provides-Dist",
        "provides_extra": "Provides-Extra",
        "requires_dist": "Requires-Dist",
        "requires_external": "Requires-External",
        "requires_python": "Requires-Python",
        "supported_platforms": "Supported-Platform",
        "import_names": "Import-Name",
        "import_namespaces": "Import-Namespace",
    }
)


def _parse_keywords(data: str) -> list[str]:
    return [item.strip() for item in data.split(",")]


def _parse_project_urls(data: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in data:
        label, _, url = (part.strip() for part in item.partition(","))
        if label in result:
            raise KeyError("duplicate labels in project urls")
        result[label] = url
    return result


class RFC822Message(email.message.Message):
    """A Message variant used for RFC 822 style core metadata."""


class _RFC822Policy(email.policy.EmailPolicy):
    def header_source_parse(self, sourcelines: list[str]) -> tuple[str, str]:
        name, value = sourcelines[0].split(":", 1)
        value = value.lstrip(" \t")
        if len(sourcelines) > 1:
            value += "".join(sourcelines[1:])
        return name, value.rstrip("\r\n")

    def header_store_parse(self, name: str, value: Any) -> tuple[str, Any]:
        return name, value


RFC822Policy = _RFC822Policy(utf8=True, message_factory=RFC822Message)


def _get_payload(message: email.message.Message, source: bytes | str) -> str:
    if isinstance(source, str):
        payload = message.get_payload()
        if not isinstance(payload, str):
            raise ValueError("payload is not a string")
        return payload

    payload = message.get_payload(decode=True)
    if not isinstance(payload, bytes):
        raise ValueError("payload in an invalid encoding")
    try:
        return payload.decode("utf8", "strict")
    except UnicodeDecodeError as exc:
        raise ValueError("payload in an invalid encoding") from exc


def parse_email(data: bytes | str) -> tuple[RawMetadata, dict[str, list[str]]]:
    if isinstance(data, bytes):
        message = email.parser.BytesParser(policy=RFC822Policy).parsebytes(data)
    else:
        message = email.parser.Parser(policy=RFC822Policy).parsestr(data)

    raw: RawMetadata = {}
    unparsed: dict[str, list[str]] = {}

    for header in set(message.keys()):
        key = header.lower()
        values = cast(list[str], message.get_all(header, failobj=[]))
        field = _EMAIL_TO_RAW_MAPPING.get(key)

        if field is None:
            unparsed[header] = values
        elif field in _STRING_FIELDS:
            if values:
                raw[field] = values[-1]
            if len(values) > 1:
                unparsed[header] = values[:-1]
        elif field == "keywords":
            raw["keywords"] = _parse_keywords(", ".join(values))
        elif field == "project_urls":
            try:
                raw["project_urls"] = _parse_project_urls(values)
            except KeyError:
                unparsed[header] = values
        else:
            raw[field] = values

    try:
        payload = _get_payload(message, data)
    except ValueError:
        unparsed["payload"] = []
    else:
        if payload:
            if "description" in raw:
                unparsed["description"] = [payload]
            else:
                raw["description"] = payload

    return raw, unparsed


_NOT_FOUND = object()

_VERSION_ORDER = {
    "1.0": 10,
    "1.1": 11,
    "1.2": 12,
    "2.1": 21,
    "2.2": 22,
    "2.3": 23,
    "2.4": 24,
    "2.5": 25,
    "2.6": 26,
}

_FIELD_INTRODUCED = {
    "metadata_version": "1.0",
    "name": "1.0",
    "version": "1.0",
    "platforms": "1.0",
    "summary": "1.0",
    "description": "1.0",
    "keywords": "1.0",
    "home_page": "1.0",
    "author": "1.0",
    "author_email": "1.0",
    "license": "1.0",
    "supported_platforms": "1.1",
    "download_url": "1.1",
    "classifiers": "1.1",
    "requires": "1.1",
    "provides": "1.1",
    "obsoletes": "1.1",
    "maintainer": "1.2",
    "maintainer_email": "1.2",
    "requires_dist": "1.2",
    "provides_dist": "1.2",
    "obsoletes_dist": "1.2",
    "requires_python": "1.2",
    "requires_external": "1.2",
    "project_urls": "1.2",
    "description_content_type": "2.1",
    "provides_extra": "2.1",
    "dynamic": "2.2",
    "license_expression": "2.4",
    "license_files": "2.4",
    "import_names": "2.5",
    "import_namespaces": "2.5",
}


def _invalid(field: str, text: str) -> InvalidMetadata:
    return InvalidMetadata(field, text)


def _require_string(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise _invalid(field, "value must be a string")
    return value


def _validate_name(value: Any) -> str:
    value = _require_string("name", value)
    try:
        utils.canonicalize_name(value, validate=True)
    except Exception:
        raise _invalid("name", f"{value!r} is invalid") from None
    return value


def _validate_version(value: Any) -> Version:
    value = _require_string("version", value)
    try:
        return version_module.Version(value)
    except Exception as exc:
        raise _invalid("version", f"{value!r} is invalid") from exc


def _validate_metadata_version(value: Any) -> str:
    value = _require_string("metadata_version", value)
    if value not in _VERSION_ORDER:
        raise _invalid("metadata_version", f"{value!r} is not a valid metadata version")
    return value


def _validate_requirement(field: str, value: Any) -> requirements.Requirement:
    value = _require_string(field, value)
    try:
        return requirements.Requirement(value)
    except Exception as exc:
        raise _invalid(field, f"{value!r} is invalid") from exc


def _validate_specifier(value: Any) -> specifiers.SpecifierSet:
    value = _require_string("requires_python", value)
    try:
        return specifiers.SpecifierSet(value)
    except Exception as exc:
        raise _invalid("requires_python", f"{value!r} is invalid") from exc


def _validate_license_expression(value: Any) -> NormalizedLicenseExpression:
    value = _require_string("license_expression", value)
    try:
        return licenses.normalize_license_expression(value)
    except Exception as exc:
        raise _invalid("license_expression", f"{value!r} is invalid") from exc


def _validate_extra(value: Any, metadata_version: str) -> str:
    value = _require_string("provides_extra", value)
    try:
        normalized = utils.canonicalize_name(value, validate=True)
    except Exception:
        raise _invalid("provides_extra", f"{value!r} is invalid") from None
    if _VERSION_ORDER[metadata_version] >= 23 and value != normalized:
        raise _invalid(
            "provides_extra",
            f"{value!r} is invalid; it must be normalized to {normalized!r}",
        )
    return normalized


def _validate_license_file(value: Any) -> str:
    value = _require_string("license_files", value)
    path = pathlib.PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise _invalid("license_files", f"{value!r} is invalid")
    return value


def _validate_import_name(field: str, value: Any) -> str:
    value = _require_string(field, value)
    if value == "*" and field == "import_names":
        return value
    if not value or value.startswith(".") or value.endswith("."):
        raise _invalid(field, f"{value!r} is invalid")
    for part in value.split("."):
        if not part.isidentifier() or keyword.iskeyword(part):
            raise _invalid(field, f"{value!r} is invalid")
    return value


class _Validator(Generic[T]):
    def __init__(
        self,
        *,
        required: bool = False,
        validator: Callable[[Any, "Metadata"], T] | None = None,
    ) -> None:
        self.required = required
        self.validator = validator
        self.name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: "Metadata | None", owner: type[Any]) -> T | None | "_Validator[T]":
        if instance is None:
            return self
        value = instance._raw.get(self.name, _NOT_FOUND)
        if value is _NOT_FOUND:
            if self.required:
                raise _invalid(self.name, "field is missing")
            return None

        introduced = _FIELD_INTRODUCED[self.name]
        version = instance._raw.get("metadata_version")
        if isinstance(version, str) and version in _VERSION_ORDER:
            if _VERSION_ORDER[introduced] > _VERSION_ORDER[version]:
                raise _invalid(
                    self.name,
                    f"field is not supported in metadata version {version!r}",
                )

        if self.validator is None:
            return cast(T, value)
        return self.validator(value, instance)


def _string(value: Any, metadata: "Metadata") -> str:
    return _require_string(metadata_field_name(metadata), value)


def metadata_field_name(metadata: "Metadata") -> str:
    return getattr(metadata, "_validation_field", "")


def _list_validator(
    item_validator: Callable[[Any, "Metadata"], Any] | None = None,
) -> Callable[[Any, "Metadata"], list[Any]]:
    def validate(value: Any, metadata: Metadata) -> list[Any]:
        field = metadata._validation_field
        if not isinstance(value, list):
            raise _invalid(field, "value must be a list")
        if item_validator is None:
            for item in value:
                _require_string(field, item)
            return value
        return [item_validator(item, metadata) for item in value]

    return validate


class Metadata:
    metadata_version = _Validator[str](required=True)
    name = _Validator[str](required=True)
    version = _Validator[Version](required=True)
    platforms = _Validator[list[str]]()
    summary = _Validator[str]()
    description = _Validator[str]()
    keywords = _Validator[list[str]]()
    home_page = _Validator[str]()
    author = _Validator[str]()
    author_email = _Validator[str]()
    license = _Validator[str]()
    supported_platforms = _Validator[list[str]]()
    download_url = _Validator[str]()
    classifiers = _Validator[list[str]]()
    requires = _Validator[list[str]]()
    provides = _Validator[list[str]]()
    obsoletes = _Validator[list[str]]()
    maintainer = _Validator[str]()
    maintainer_email = _Validator[str]()
    requires_dist = _Validator[list[requirements.Requirement]]()
    provides_dist = _Validator[list[requirements.Requirement]]()
    obsoletes_dist = _Validator[list[requirements.Requirement]]()
    requires_python = _Validator[specifiers.SpecifierSet]()
    requires_external = _Validator[list[str]]()
    project_urls = _Validator[dict[str, str]]()
    description_content_type = _Validator[str]()
    provides_extra = _Validator[list[str]]()
    dynamic = _Validator[list[str]]()
    license_expression = _Validator[NormalizedLicenseExpression]()
    license_files = _Validator[list[str]]()
    import_names = _Validator[list[str]]()
    import_namespaces = _Validator[list[str]]()

    def __init__(self, **kwargs: Any) -> None:
        self._raw = cast(RawMetadata, kwargs)
        self._validation_field = ""

    @classmethod
    def from_raw(cls, data: RawMetadata, *, validate: bool = True) -> "Metadata":
        instance = cls(**data)
        if validate:
            errors: list[InvalidMetadata] = []
            for field in _FIELD_INTRODUCED:
                instance._validation_field = field
                try:
                    getattr(instance, field)
                except InvalidMetadata as exc:
                    errors.append(exc)
            instance._validation_field = ""
            if errors:
                raise ExceptionGroup("invalid metadata", errors)
        return instance

    @classmethod
    def from_email(cls, data: bytes | str, *, validate: bool = True) -> "Metadata":
        raw, unparsed = parse_email(data)
        if validate and unparsed:
            errors = [
                _invalid(field, f"unrecognized or malformed field {field!r}")
                for field in unparsed
            ]
            raise ExceptionGroup("unparsed metadata", errors)
        return cls.from_raw(raw, validate=validate)

    def _value(self, field: str) -> Any:
        self._validation_field = field
        try:
            if field == "metadata_version":
                return _validate_metadata_version(self._raw[field])
            if field == "name":
                return _validate_name(self._raw[field])
            if field == "version":
                return _validate_version(self._raw[field])
            value = self._raw.get(field, _NOT_FOUND)
            if value is _NOT_FOUND:
                return None
            if field in _STRING_FIELDS:
                return _require_string(field, value)
            if field in _LIST_FIELDS:
                if not isinstance(value, list):
                    raise _invalid(field, "value must be a list")
                if field in {"requires_dist", "provides_dist", "obsoletes_dist"}:
                    return [_validate_requirement(field, item) for item in value]
                if field == "provides_extra":
                    version = self._raw.get("metadata_version", "2.1")
                    return [_validate_extra(item, cast(str, version)) for item in value]
                if field == "license_files":
                    return [_validate_license_file(item) for item in value]
                if field in {"import_names", "import_namespaces"}:
                    result = [_validate_import_name(field, item) for item in value]
                    if field == "import_names" and "*" in result and len(result) != 1:
                        raise _invalid(field, "'*' must be the only import name")
                    return result
                return [_require_string(field, item) for item in value]
            if field == "project_urls":
                if not isinstance(value, dict):
                    raise _invalid(field, "value must be a dictionary")
                for label, url in value.items():
                    if not isinstance(label, str) or not isinstance(url, str):
                        raise _invalid(field, "value must be a dictionary of strings")
                    if not label or not url:
                        raise _invalid(field, "project URL labels and URLs must not be empty")
                return value
            return value
        finally:
            self._validation_field = ""

    def __getattribute__(self, name: str) -> Any:
        fields = _FIELD_INTRODUCED
        if name in fields:
            raw = object.__getattribute__(self, "_raw")
            if name not in raw:
                if name in {"metadata_version", "name", "version"}:
                    raise _invalid(name, "field is missing")
                return None
            return object.__getattribute__(self, "_value")(name)
        return object.__getattribute__(self, name)

    def as_email(self) -> bytes:
        message = RFC822Message()
        message.policy = RFC822Policy
        description: str | None = None

        for field in _FIELD_INTRODUCED:
            if field not in self._raw:
                continue
            value = getattr(self, field)
            if value is None:
                continue
            if field == "description":
                description = cast(str, value)
                continue
            header = _RAW_TO_EMAIL_MAPPING[field]
            if field == "keywords":
                message[header] = ", ".join(cast(list[str], value))
            elif field == "project_urls":
                for label, url in cast(dict[str, str], value).items():
                    message[header] = f"{label}, {url}"
            elif field in _LIST_FIELDS:
                for item in cast(list[Any], value):
                    message[header] = str(item)
            else:
                message[header] = str(value)

        if description is not None:
            message.set_payload(description)

        return message.as_bytes(policy=RFC822Policy)