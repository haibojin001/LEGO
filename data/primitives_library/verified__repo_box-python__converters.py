from __future__ import annotations

import csv
import json
from collections.abc import Callable
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any

from box.exceptions import BoxError

pyyaml_available = True
ruamel_available = True
msgpack_available = True

try:
    from ruamel.yaml import YAML, version_info
except ImportError:
    ruamel_available = False
else:
    if version_info[1] < 17:
        ruamel_available = False

try:
    import yaml
except ImportError:
    pyyaml_available = False

MISSING_PARSER_ERROR = "No YAML Parser available, please install ruamel.yaml>=0.17 or PyYAML"

toml_read_library: Any | None = None
toml_write_library: Any | None = None
toml_decode_error: Callable | None = None

__all__ = [
    "_to_json",
    "_to_yaml",
    "_to_toml",
    "_to_csv",
    "_to_msgpack",
    "_to_toon",
    "_from_json",
    "_from_yaml",
    "_from_toml",
    "_from_csv",
    "_from_msgpack",
    "_from_toon",
]


class BoxTomlDecodeError(BoxError):
    """Toml Decode Error"""


try:
    import toml
except ImportError:
    pass
else:
    toml_read_library = toml
    toml_write_library = toml
    toml_decode_error = toml.TomlDecodeError

    class BoxTomlDecodeError(BoxError, toml.TomlDecodeError):  # type: ignore
        """Toml Decode Error"""


try:
    import tomllib
except ImportError:
    pass
else:
    toml_read_library = tomllib
    toml_decode_error = tomllib.TOMLDecodeError

    class BoxTomlDecodeError(BoxError, tomllib.TOMLDecodeError):  # type: ignore
        """Toml Decode Error"""


try:
    import tomli
except ImportError:
    pass
else:
    toml_read_library = tomli
    toml_decode_error = tomli.TOMLDecodeError

    class BoxTomlDecodeError(BoxError, tomli.TOMLDecodeError):  # type: ignore
        """Toml Decode Error"""


try:
    import tomli_w
except ImportError:
    pass
else:
    toml_write_library = tomli_w


try:
    import msgpack  # type: ignore
except ImportError:
    msgpack = None  # type: ignore
    msgpack_available = False

toon_available = True

try:
    from toon_format import decode as toon_decode
    from toon_format import encode as toon_encode
except ImportError:
    toon_available = False

yaml_available = pyyaml_available or ruamel_available

BOX_PARAMETERS = (
    "default_box",
    "default_box_attr",
    "default_box_none_transform",
    "default_box_create_on_get",
    "frozen_box",
    "camel_killer_box",
    "conversion_box",
    "modify_tuples_box",
    "box_safe_prefix",
    "box_duplicates",
    "box_intact_types",
    "box_dots",
    "box_dots_exclude",
    "box_recast",
    "box_class",
    "box_namespace",
)


def _exists(filename: str | PathLike, create: bool = False) -> Path:
    target = Path(filename)
    if create:
        try:
            target.touch(exist_ok=True)
        except OSError as exc:
            raise BoxError(f"Could not create file {filename} - {exc}")
        return target

    if not target.exists():
        raise BoxError(f'File "{filename}" does not exist')
    if not target.is_file():
        raise BoxError(f"{filename} is not a file")
    return target


def _to_json(
    obj,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    **json_kwargs,
):
    if filename:
        _exists(filename, create=True)
        with open(filename, "w", encoding=encoding, errors=errors) as handle:
            json.dump(obj, handle, ensure_ascii=False, **json_kwargs)
        return None
    return json.dumps(obj, ensure_ascii=False, **json_kwargs)


def _from_json(
    json_string: str | None = None,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    multiline: bool = False,
    **kwargs,
):
    if filename:
        with open(filename, "r", encoding=encoding, errors=errors) as handle:
            if multiline:
                return [
                    json.loads(line.strip(), **kwargs)
                    for line in handle
                    if line.strip() and not line.strip().startswith("#")
                ]
            return json.load(handle, **kwargs)

    if json_string:
        return json.loads(json_string, **kwargs)

    raise BoxError("from_json requires a string or filename")


def _to_yaml(
    obj,
    filename: str | PathLike | None = None,
    default_flow_style: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
    ruamel_typ: str = "rt",
    ruamel_attrs: dict | None = None,
    width: int = 120,
    **yaml_kwargs,
):
    attributes = ruamel_attrs or {}

    if ruamel_available:
        serializer = YAML(typ=ruamel_typ)
        serializer.default_flow_style = default_flow_style
        serializer.width = width
        for name, value in attributes.items():
            setattr(serializer, name, value)

        if filename:
            _exists(filename, create=True)
            with open(filename, "w", encoding=encoding, errors=errors) as handle:
                return serializer.dump(obj, stream=handle, **yaml_kwargs)

        stream = StringIO()
        serializer.dump(obj, stream=stream, **yaml_kwargs)
        return stream.getvalue()

    if pyyaml_available:
        if filename:
            _exists(filename, create=True)
            with open(filename, "w", encoding=encoding, errors=errors) as handle:
                return yaml.dump(
                    obj,
                    stream=handle,
                    default_flow_style=default_flow_style,
                    width=width,
                    **yaml_kwargs,
                )
        return yaml.dump(obj, default_flow_style=default_flow_style, width=width, **yaml_kwargs)

    raise BoxError(MISSING_PARSER_ERROR)


def _from_yaml(
    yaml_string: str | None = None,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    ruamel_typ: str = "rt",
    ruamel_attrs: dict | None = None,
    **kwargs,
):
    attributes = ruamel_attrs or {}

    if filename:
        _exists(filename)
        with open(filename, "r", encoding=encoding, errors=errors) as handle:
            if ruamel_available:
                parser = YAML(typ=ruamel_typ)
                for name, value in attributes.items():
                    setattr(parser, name, value)
                return parser.load(stream=handle)

            if pyyaml_available:
                if "Loader" not in kwargs:
                    kwargs["Loader"] = yaml.SafeLoader
                return yaml.load(handle, **kwargs)

            raise BoxError(MISSING_PARSER_ERROR)

    if yaml_string:
        if ruamel_available:
            parser = YAML(typ=ruamel_typ)
            for name, value in attributes.items():
                setattr(parser, name, value)
            return parser.load(stream=yaml_string)

        if pyyaml_available:
            if "Loader" not in kwargs:
                kwargs["Loader"] = yaml.SafeLoader
            return yaml.load(yaml_string, **kwargs)

        raise BoxError(MISSING_PARSER_ERROR)

    raise BoxError("from_yaml requires a string or filename")


def _to_toml(
    obj,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
):
    if toml_write_library is None:
        raise BoxError("No TOML writer available, please install toml or tomli-w")

    try:
        if filename:
            _exists(filename, create=True)
            if toml_write_library.__name__ == "toml":
                with open(filename, "w", encoding=encoding, errors=errors) as handle:
                    return toml_write_library.dump(obj, handle)
            with open(filename, "wb") as handle:
                return toml_write_library.dump(obj, handle)

        return toml_write_library.dumps(obj)
    except toml_decode_error as exc:  # type: ignore[misc]
        raise BoxTomlDecodeError(exc) from exc


def _from_toml(
    toml_string: str | None = None,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
):
    if toml_read_library is None:
        raise BoxError("No TOML reader available, please install toml, tomli, or use Python 3.11+")

    try:
        if filename:
            _exists(filename)
            if toml_read_library.__name__ == "toml":
                with open(filename, "r", encoding=encoding, errors=errors) as handle:
                    return toml_read_library.load(handle)
            with open(filename, "rb") as handle:
                return toml_read_library.load(handle)

        if toml_string:
            return toml_read_library.loads(toml_string)
    except toml_decode_error as exc:  # type: ignore[misc]
        raise BoxTomlDecodeError(exc) from exc

    raise BoxError("from_toml requires a string or filename")


def _to_csv(
    obj,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    **csv_kwargs,
):
    fieldnames = obj[0].keys()

    if filename:
        _exists(filename, create=True)
        with open(filename, "w", newline="", encoding=encoding, errors=errors) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, **csv_kwargs)
            writer.writeheader()
            writer.writerows(obj)
        return None

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, **csv_kwargs)
    writer.writeheader()
    writer.writerows(obj)
    return output.getvalue()


def _from_csv(
    csv_string: str | None = None,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    **kwargs,
):
    if filename:
        _exists(filename)
        with open(filename, "r", newline="", encoding=encoding, errors=errors) as handle:
            return list(csv.DictReader(handle, **kwargs))

    if csv_string:
        return list(csv.DictReader(StringIO(csv_string), **kwargs))

    raise BoxError("from_csv requires a string or filename")


def _to_msgpack(obj, filename: str | PathLike | None = None, **kwargs):
    if not msgpack_available:
        raise BoxError("msgpack is not installed")

    if filename:
        _exists(filename, create=True)
        with open(filename, "wb") as handle:
            return msgpack.pack(obj, handle, **kwargs)  # type: ignore[union-attr]

    return msgpack.dumps(obj, **kwargs)  # type: ignore[union-attr]


def _from_msgpack(
    msgpack_bytes: bytes | None = None,
    filename: str | PathLike | None = None,
    **kwargs,
):
    if not msgpack_available:
        raise BoxError("msgpack is not installed")

    if filename:
        _exists(filename)
        with open(filename, "rb") as handle:
            return msgpack.unpack(handle, **kwargs)  # type: ignore[union-attr]

    if msgpack_bytes:
        return msgpack.loads(msgpack_bytes, **kwargs)  # type: ignore[union-attr]

    raise BoxError("from_msgpack requires bytes or filename")


def _to_toon(
    obj,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    **kwargs,
):
    if not toon_available:
        raise BoxError("toon_format is not installed")

    encoded = toon_encode(obj, **kwargs)

    if filename:
        _exists(filename, create=True)
        with open(filename, "w", encoding=encoding, errors=errors) as handle:
            handle.write(encoded)
        return None

    return encoded


def _from_toon(
    toon_string: str | None = None,
    filename: str | PathLike | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    **kwargs,
):
    if not toon_available:
        raise BoxError("toon_format is not installed")

    if filename:
        _exists(filename)
        with open(filename, "r", encoding=encoding, errors=errors) as handle:
            return toon_decode(handle.read(), **kwargs)

    if toon_string:
        return toon_decode(toon_string, **kwargs)

    raise BoxError("from_toon requires a string or filename")