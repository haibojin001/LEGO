from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Callable, Generator, Iterable, Mapping
from inspect import signature
from keyword import iskeyword
from os import PathLike
from typing import Any, Literal

import box
from box.converters import (
    BOX_PARAMETERS,
    _from_json,
    _from_msgpack,
    _from_toml,
    _from_toon,
    _from_yaml,
    _to_json,
    _to_msgpack,
    _to_toml,
    _to_toon,
    _to_yaml,
    msgpack_available,
    toon_available,
    toml_read_library,
    toml_write_library,
    yaml_available,
)
from box.exceptions import BoxError, BoxKeyError, BoxTypeError, BoxValueError, BoxWarning

__all__ = ["Box"]

_first_cap_re = re.compile(r"(.)([A-Z][a-z]+)")
_all_cap_re = re.compile(r"([a-z0-9])([A-Z])")
_list_pos_re = re.compile(r"\[(\d+)\]")
NO_DEFAULT = object()
NO_NAMESPACE = object()


def _is_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return bool(get_ipython())


def _exception_cause(e):
    return e.__cause__ if isinstance(e, (BoxKeyError, BoxValueError)) else e


def _camel_killer(attr):
    attr = str(attr)
    return re.sub(r" *_+", "_", _all_cap_re.sub(r"\1_\2", _first_cap_re.sub(r"\1_\2", attr)).lower())


def _recursive_tuples(iterable, box_class, recreate_tuples=False, **kwargs):
    values = []
    for value in iterable:
        if isinstance(value, dict):
            values.append(box_class(value, **kwargs))
        elif isinstance(value, list) or (recreate_tuples and isinstance(value, tuple)):
            values.append(_recursive_tuples(value, box_class, recreate_tuples, **kwargs))
        else:
            values.append(value)
    return tuple(values)


def _parse_box_dots(bx, item, setting=False):
    for index, char in enumerate(item):
        if char == "[":
            return item[:index], item[index:]
        if char == ".":
            return item[:index], item[index + 1:]
    if setting and "." in item:
        return item.split(".", 1)
    raise BoxError("Could not split box dots properly")


def _get_dot_paths(bx, current=""):
    def walk_dict(value, prefix=""):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            if isinstance(child, dict):
                yield from walk_dict(child, path)
            elif isinstance(child, list):
                yield from walk_list(child, path)

    def walk_list(value, prefix=""):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            yield path
            if isinstance(child, dict):
                yield from walk_dict(child, path)
            elif isinstance(child, list):
                yield from walk_list(child, path)

    yield from walk_dict(bx, current)


def _get_box_config():
    return {"__created": False, "__safe_keys": {}}


def _get_property_func(obj, key):
    obj_type = type(obj)
    if not hasattr(obj_type, key):
        return None, None, None
    prop = getattr(obj_type, key)
    if not isinstance(prop, property):
        return None, None, None
    return prop.fget, prop.fset, prop.fdel


def _boxlist_type():
    return getattr(box, "BoxList", list)


class Box(dict):
    _box_config: dict[str, Any]

    _protected_keys = [
        "to_dict",
        "to_json",
        "to_yaml",
        "from_yaml",
        "from_json",
        "from_toml",
        "to_toml",
        "merge_update",
    ] + [name for name in dir({}) if not name.startswith("_")]

    def __new__(
        cls,
        *args: Any,
        default_box: bool = False,
        default_box_attr: Any = NO_DEFAULT,
        default_box_none_transform: bool = True,
        default_box_create_on_get: bool = True,
        frozen_box: bool = False,
        camel_killer_box: bool = False,
        conversion_box: bool = True,
        modify_tuples_box: bool = False,
        box_safe_prefix: str = "x",
        box_duplicates: str = "ignore",
        box_intact_types: tuple | list = (),
        box_recast: dict | None = None,
        box_dots: bool = False,
        box_dots_exclude: str | None = None,
        box_class: dict | type["Box"] | None = None,
        box_namespace: tuple[str, ...] | Literal[False] = (),
        **kwargs: Any,
    ):
        obj = super().__new__(cls)
        object.__setattr__(obj, "_box_config", _get_box_config())
        obj._box_config.update(
            {
                "default_box": default_box,
                "default_box_attr": cls if default_box_attr is NO_DEFAULT else default_box_attr,
                "default_box_none_transform": default_box_none_transform,
                "default_box_create_on_get": default_box_create_on_get,
                "frozen_box": frozen_box,
                "camel_killer_box": camel_killer_box,
                "conversion_box": conversion_box,
                "modify_tuples_box": modify_tuples_box,
                "box_safe_prefix": box_safe_prefix,
                "box_duplicates": box_duplicates,
                "box_intact_types": tuple(box_intact_types),
                "box_recast": box_recast,
                "box_dots": box_dots,
                "box_dots_exclude": re.compile(box_dots_exclude) if box_dots_exclude else None,
                "box_class": box_class or Box,
                "box_namespace": box_namespace,
            }
        )
        return obj

    def __init__(
        self,
        *args: Any,
        default_box: bool = False,
        default_box_attr: Any = NO_DEFAULT,
        default_box_none_transform: bool = True,
        default_box_create_on_get: bool = True,
        frozen_box: bool = False,
        camel_killer_box: bool = False,
        conversion_box: bool = True,
        modify_tuples_box: bool = False,
        box_safe_prefix: str = "x",
        box_duplicates: str = "ignore",
        box_intact_types: tuple | list = (),
        box_recast: dict | None = None,
        box_dots: bool = False,
        box_dots_exclude: str | None = None,
        box_class: dict | type["Box"] | None = None,
        box_namespace: tuple[str, ...] | Literal[False] = (),
        **kwargs: Any,
    ):
        dict.__init__(self)
        config = self._box_config
        config["__created"] = False
        if len(args) > 1:
            raise BoxTypeError(f"Box expected at most 1 argument, got {len(args)}")
        if args:
            source = args[0]
            if isinstance(source, Mapping):
                for key, value in source.items():
                    self[key] = value
            else:
                for key, value in source:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value
        config["__created"] = True

    def _copy_config(self):
        result = {}
        for key, value in self._box_config.items():
            if not key.startswith("__"):
                result[key] = value
        result["box_dots_exclude"] = (
            result["box_dots_exclude"].pattern
            if result.get("box_dots_exclude") is not None
            else None
        )
        return result

    def _conversion_key(self, key):
        if not isinstance(key, str):
            return key
        config = self._box_config
        converted = _camel_killer(key) if config["camel_killer_box"] else key
        if config["conversion_box"]:
            if converted.startswith(config["box_safe_prefix"]) and converted[len(config["box_safe_prefix"]):] in self._protected_keys:
                converted = converted[len(config["box_safe_prefix"]):]
            safe = re.sub(r"\W", "_", converted)
            if safe and safe[0].isdigit():
                safe = "_" + safe
            if iskeyword(safe) or safe in self._protected_keys:
                safe = config["box_safe_prefix"] + safe
            config["__safe_keys"][safe] = converted
        return converted

    def _attribute_key(self, key):
        safe = self._box_config["__safe_keys"].get(key, key)
        if self._box_config["camel_killer_box"]:
            safe = _camel_killer(safe)
        return safe

    def _is_dots_key(self, key):
        if not self._box_config["box_dots"] or not isinstance(key, str):
            return False
        exclude = self._box_config["box_dots_exclude"]
        return (("." in key or "[" in key) and not (exclude and exclude.search(key)))

    def _sub_config(self, key=NO_NAMESPACE):
        config = self._copy_config()
        namespace = config.get("box_namespace", ())
        if namespace is not False and key is not NO_NAMESPACE:
            config["box_namespace"] = tuple(namespace) + (str(key),)
        return config

    def _convert_value(self, value, key=NO_NAMESPACE):
        config = self._box_config
        if config["box_recast"] and key in config["box_recast"]:
            value = config["box_recast"][key](value)
        if isinstance(value, config["box_intact_types"]):
            return value
        if isinstance(value, dict) and not isinstance(value, Box):
            cls = config["box_class"]
            return cls(value, **self._sub_config(key))
        if isinstance(value, list) and not isinstance(value, _boxlist_type()):
            cls = _boxlist_type()
            try:
                return cls(value, box_class=config["box_class"], **self._sub_config(key))
            except TypeError:
                return cls([self._convert_value(v, key) for v in value])
        if isinstance(value, tuple) and config["modify_tuples_box"]:
            return _recursive_tuples(value, config["box_class"], True, **self._sub_config(key))
        return value

    def __setitem__(self, key, value):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        if self._is_dots_key(key):
            self._set_dots(key, value)
            return
        converted = self._conversion_key(key)
        if converted != key and converted in self:
            mode = self._box_config["box_duplicates"]
            message = f"Duplicate conversion attributes exist: {key!r} and {converted!r}"
            if mode == "error":
                raise BoxError(message)
            if mode == "warn":
                warnings.warn(message, BoxWarning)
        dict.__setitem__(self, converted, self._convert_value(value, converted))

    def _set_dots(self, key, value):
        current = self
        parts = re.split(r"(\[\d+\])|\.", key)
        parts = [part for part in parts if part not in (None, "")]
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if part.startswith("["):
                position = int(part[1:-1])
                while len(current) <= position:
                    current.append({} if not next_part.startswith("[") else [])
                current = current[position]
            else:
                if part not in current or current[part] is None:
                    current[part] = [] if next_part.startswith("[") else {}
                current = current[part]
        last = parts[-1]
        if last.startswith("["):
            position = int(last[1:-1])
            while len(current) <= position:
                current.append(None)
            current[position] = self._convert_value(value, last)
        else:
            current[last] = value

    def _get_dots(self, key):
        current = self
        parts = re.split(r"(\[\d+\])|\.", key)
        for part in (p for p in parts if p not in (None, "")):
            if part.startswith("["):
                current = current[int(part[1:-1])]
            else:
                current = dict.__getitem__(current, current._conversion_key(part) if isinstance(current, Box) else part)
        return current

    def __getitem__(self, key):
        if self._is_dots_key(key):
            try:
                return self._get_dots(key)
            except (KeyError, IndexError, TypeError) as exc:
                raise BoxKeyError(str(key)) from exc
        converted = self._conversion_key(key)
        try:
            value = dict.__getitem__(self, converted)
        except KeyError:
            if self._box_config["default_box"]:
                default = self._box_config["default_box_attr"]
                try:
                    value = default() if isinstance(default, type) else copy.deepcopy(default)
                except Exception:
                    value = default
                if self._box_config["default_box_create_on_get"]:
                    self[converted] = value
                    value = dict.__getitem__(self, converted)
                return value
            raise BoxKeyError(str(key)) from None
        if value is None and self._box_config["default_box"] and self._box_config["default_box_none_transform"]:
            default = self._box_config["default_box_attr"]
            value = default() if isinstance(default, type) else copy.deepcopy(default)
            if self._box_config["default_box_create_on_get"]:
                self[converted] = value
                value = dict.__getitem__(self, converted)
        return value

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            return self[self._attribute_key(item)]
        except BoxKeyError as exc:
            raise BoxKeyError(f"'{type(self).__name__}' object has no attribute '{item}'") from exc

    def __setattr__(self, key, value):
        if key == "_box_config" or key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        fget, fset, _ = _get_property_func(self, key)
        if fset is not None:
            fset(self, value)
            return
        self[self._attribute_key(key)] = value

    def __delattr__(self, item):
        if item.startswith("_"):
            object.__delattr__(self, item)
            return
        _, _, fdel = _get_property_func(self, item)
        if fdel is not None:
            fdel(self)
            return
        try:
            del self[self._attribute_key(item)]
        except KeyError as exc:
            raise BoxKeyError(f"'{type(self).__name__}' object has no attribute '{item}'") from exc

    def __delitem__(self, key):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        if self._is_dots_key(key):
            parent, final = key.rsplit(".", 1) if "." in key else ("", key)
            target = self._get_dots(parent) if parent else self
            del target[final]
            return
        dict.__delitem__(self, self._conversion_key(key))

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._box_config.get("__safe_keys", {})))

    def __repr__(self):
        return f"<{type(self).__name__}: {dict.__repr__(self)}>"

    def __str__(self):
        return dict.__repr__(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = type(self)(**self._copy_config())
        memo[id(self)] = result
        for key, value in self.items():
            dict.__setitem__(result, copy.deepcopy(key, memo), copy.deepcopy(value, memo))
        return result

    def __reduce__(self):
        return type(self), (self.to_dict(),), self._copy_config()

    def __setstate__(self, state):
        self._box_config.update(state)

    def copy(self):
        result = type(self)(**self._copy_config())
        for key, value in self.items():
            dict.__setitem__(result, key, value)
        return result

    def clear(self):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        dict.clear(self)

    def pop(self, key, default=NO_DEFAULT):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        converted = self._conversion_key(key)
        if default is NO_DEFAULT:
            return dict.pop(self, converted)
        return dict.pop(self, converted, default)

    def popitem(self):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        return dict.popitem(self)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return self[key]

    def update(self, *args, **kwargs):
        if self._box_config.get("__created") and self._box_config["frozen_box"]:
            raise BoxError("Box is frozen")
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 argument, got {len(args)}")
        if args:
            source = args[0]
            iterable = source.items() if isinstance(source, Mapping) else source
            for key, value in iterable:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def to_dict(self):
        def convert(value):
            if isinstance(value, Box):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                converted = [convert(item) for item in value]
                return tuple(converted) if isinstance(value, tuple) else converted
            return value

        return convert(self)

    def to_list(self):
        return [self.to_dict()]

    def merge_update(self, __m=None, **kwargs):
        if __m is not None:
            if not isinstance(__m, Mapping):
                raise BoxTypeError("merge_update requires a mapping")
            kwargs = dict(__m, **kwargs)
        for key, value in kwargs.items():
            if key in self and isinstance(self[key], Mapping) and isinstance(value, Mapping):
                if not isinstance(self[key], Box):
                    self[key] = Box(self[key], **self._sub_config(key))
                self[key].merge_update(value)
            else:
                self[key] = value
        return self

    def merge(self, __m=None, **kwargs):
        result = copy.deepcopy(self)
        return result.merge_update(__m, **kwargs)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, BoxKeyError):
            return default

    def get_path(self, path, default=NO_DEFAULT):
        try:
            return self._get_dots(path)
        except (KeyError, IndexError, TypeError, BoxKeyError):
            if default is NO_DEFAULT:
                raise BoxKeyError(path) from None
            return default

    def set_path(self, path, value):
        self._set_dots(path, value)
        return self

    def del_path(self, path):
        components = path.rsplit(".", 1)
        if len(components) == 1:
            del self[path]
        else:
            parent = self._get_dots(components[0])
            del parent[components[1]]

    @property
    def keys_dots(self):
        return list(_get_dot_paths(self))

    def to_json(self, filename=None, encoding="utf-8", errors="strict", **kwargs):
        return _to_json(self.to_dict(), filename=filename, encoding=encoding, errors=errors, **kwargs)

    @classmethod
    def from_json(cls, json_string, **kwargs):
        try:
            data = _from_json(json_string, **kwargs)
            return cls(data, **{k: v for k, v in kwargs.items() if k in BOX_PARAMETERS})
        except Exception as exc:
            raise BoxError(str(exc)) from _exception_cause(exc)

    def to_yaml(self, filename=None, encoding="utf-8", errors="strict", **kwargs):
        return _to_yaml(self.to_dict(), filename=filename, encoding=encoding, errors=errors, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_string, **kwargs):
        try:
            data = _from_yaml(yaml_string, **kwargs)
            return cls(data, **{k: v for k, v in kwargs.items() if k in BOX_PARAMETERS})
        except Exception as exc:
            raise BoxError(str(exc)) from _exception_cause(exc)

    def to_toml(self, filename=None, encoding="utf-8", errors="strict", **kwargs):
        return _to_toml(self.to_dict(), filename=filename, encoding=encoding, errors=errors, **kwargs)

    @classmethod
    def from_toml(cls, toml_string, **kwargs):
        try:
            data = _from_toml(toml_string, **kwargs)
            return cls(data, **{k: v for k, v in kwargs.items() if k in BOX_PARAMETERS})
        except Exception as exc:
            raise BoxError(str(exc)) from _exception_cause(exc)

    def to_msgpack(self, filename=None, **kwargs):
        return _to_msgpack(self.to_dict(), filename=filename, **kwargs)

    @classmethod
    def from_msgpack(cls, msgpack_bytes, **kwargs):
        try:
            data = _from_msgpack(msgpack_bytes, **kwargs)
            return cls(data, **{k: v for k, v in kwargs.items() if k in BOX_PARAMETERS})
        except Exception as exc:
            raise BoxError(str(exc)) from _exception_cause(exc)

    def to_toon(self, filename=None, encoding="utf-8", errors="strict", **kwargs):
        return _to_toon(self.to_dict(), filename=filename, encoding=encoding, errors=errors, **kwargs)

    @classmethod
    def from_toon(cls, toon_string, **kwargs):
        try:
            data = _from_toon(toon_string, **kwargs)
            return cls(data, **{k: v for k, v in kwargs.items() if k in BOX_PARAMETERS})
        except Exception as exc:
            raise BoxError(str(exc)) from _exception_cause(exc)

    @classmethod
    def from_file(cls, filename, **kwargs):
        filename = str(filename)
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        methods = {
            "json": cls.from_json,
            "yaml": cls.from_yaml,
            "yml": cls.from_yaml,
            "toml": cls.from_toml,
            "msgpack": cls.from_msgpack,
            "mpk": cls.from_msgpack,
            "toon": cls.from_toon,
        }
        if suffix not in methods:
            raise BoxError(f"Unknown file type: {suffix}")
        return methods[suffix](filename, **kwargs)

    def to_file(self, filename, **kwargs):
        filename = str(filename)
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        methods = {
            "json": self.to_json,
            "yaml": self.to_yaml,
            "yml": self.to_yaml,
            "toml": self.to_toml,
            "msgpack": self.to_msgpack,
            "mpk": self.to_msgpack,
            "toon": self.to_toon,
        }
        if suffix not in methods:
            raise BoxError(f"Unknown file type: {suffix}")
        return methods[suffix](filename=filename, **kwargs)