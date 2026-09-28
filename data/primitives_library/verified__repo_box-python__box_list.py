from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from os import PathLike
from typing import Any

import box
from box.converters import (
    BOX_PARAMETERS,
    _from_csv,
    _from_json,
    _from_msgpack,
    _from_toml,
    _from_toon,
    _from_yaml,
    _to_csv,
    _to_json,
    _to_msgpack,
    _to_toml,
    _to_toon,
    _to_yaml,
    msgpack_available,
    toon_available,
    toml_read_library,
    yaml_available,
)
from box.exceptions import BoxError, BoxTypeError

_list_pos_re = re.compile(r"\[(\d+)\]")


class BoxList(list):
    """
    List subclass which automatically changes dictionaries and lists placed
    inside it into Box and BoxList instances.
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.box_options = {"box_class": box.Box}
        instance.box_options.update(kwargs)
        instance.box_org_ref = None
        return instance

    def __init__(
        self,
        iterable: Iterable | None = None,
        box_class: type[box.Box] = box.Box,
        **box_options,
    ):
        self.box_options = box_options
        self.box_options["box_class"] = box_class
        self.box_org_ref = iterable

        if iterable:
            for value in iterable:
                self.append(value)

        self.box_org_ref = None

        if box_options.get("frozen_box"):

            def frozen(*args, **kwargs):
                raise BoxError("BoxList is frozen")

            for name in (
                "append",
                "extend",
                "insert",
                "pop",
                "remove",
                "reverse",
                "sort",
            ):
                self.__setattr__(name, frozen)

    def __getitem__(self, item):
        if (
            self.box_options.get("box_dots")
            and isinstance(item, str)
            and item.startswith("[")
        ):
            match = _list_pos_re.search(item)
            value = super().__getitem__(int(match.groups()[0]))
            if len(match.group()) == len(item):
                return value
            return value.__getitem__(item[len(match.group()) :].lstrip("."))

        if isinstance(item, tuple):
            value = self
            for index in item:
                if isinstance(value, list):
                    value = value[index]
                else:
                    raise BoxTypeError(
                        f"Cannot numpy-style indexing on {type(value).__name__}."
                    )
            return value

        return super().__getitem__(item)

    def __delitem__(self, key):
        if self.box_options.get("frozen_box"):
            raise BoxError("BoxList is frozen")

        if (
            self.box_options.get("box_dots")
            and isinstance(key, str)
            and key.startswith("[")
        ):
            match = _list_pos_re.search(key)
            position = int(match.groups()[0])

            if len(match.group()) == len(key):
                return super().__delitem__(position)

            child = self[position]
            if hasattr(child, "__delitem__"):
                return child.__delitem__(key[len(match.group()) :].lstrip("."))

        super().__delitem__(key)

    def __setitem__(self, key, value):
        if self.box_options.get("frozen_box"):
            raise BoxError("BoxList is frozen")

        if (
            self.box_options.get("box_dots")
            and isinstance(key, str)
            and key.startswith("[")
        ):
            match = _list_pos_re.search(key)
            position = int(match.groups()[0])

            if position >= len(self) and self.box_options.get("default_box"):
                self.extend([None] * (position - len(self) + 1))

            if len(match.group()) == len(key):
                return super().__setitem__(position, value)

            children = key[len(match.group()) :].lstrip(".")
            if self.box_options.get("default_box"):
                if children[0] == "[":
                    super().__setitem__(position, box.BoxList(**self.box_options))
                else:
                    super().__setitem__(
                        position,
                        self.box_options["box_class"](**self.box_options),
                    )

            return super().__getitem__(position).__setitem__(children, value)

        super().__setitem__(key, value)

    def _is_intact_type(self, obj):
        intact = self.box_options.get("box_intact_types")
        return bool(intact and isinstance(obj, intact))

    def _convert(self, p_object):
        if isinstance(p_object, dict) and not self._is_intact_type(p_object):
            p_object = self.box_options["box_class"](p_object, **self.box_options)
        elif isinstance(p_object, box.Box):
            p_object._box_config.update(self.box_options)

        if isinstance(p_object, list) and not self._is_intact_type(p_object):
            if p_object is self or p_object is self.box_org_ref:
                p_object = self
            else:
                p_object = self.__class__(p_object, **self.box_options)
        elif isinstance(p_object, BoxList):
            p_object.box_options.update(self.box_options)

        return p_object

    def append(self, p_object):
        super().append(self._convert(p_object))

    def extend(self, iterable):
        for value in iterable:
            self.append(value)

    def insert(self, index, p_object):
        super().insert(index, self._convert(p_object))

    def _dotted_helper(self) -> list[str]:
        output = []

        for index, value in enumerate(self):
            emitted = False

            if isinstance(value, box.Box):
                for key in value.keys(dotted=True):
                    output.append(f"[{index}].{key}")
                    emitted = True
            elif isinstance(value, BoxList):
                for key in value._dotted_helper():
                    output.append(f"[{index}]{key}")
                    emitted = True

            if not emitted:
                output.append(f"[{index}]")

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_list()})"

    def __str__(self):
        return str(self.to_list())

    def __copy__(self):
        return self.__class__((value for value in self), **self.box_options)

    def __deepcopy__(self, memo=None):
        copied = self.__class__()
        memo = memo or {}
        memo[id(self)] = copied

        for value in self:
            copied.append(copy.deepcopy(value, memo=memo))

        return copied

    def __hash__(self) -> int:
        if self.box_options.get("frozen_box"):
            value = 98765
            value ^= hash(tuple(self))
            return value
        raise BoxTypeError("unhashable type: 'BoxList'")

    def to_list(self) -> list:
        output: list[Any] = []

        for value in self:
            if value is self:
                output.append(output)
            elif isinstance(value, box.Box):
                output.append(value.to_dict())
            elif isinstance(value, BoxList):
                output.append(value.to_list())
            else:
                output.append(value)

        return output

    @staticmethod
    def _box_arguments(kwargs):
        box_args = {}
        for key in list(kwargs):
            if key in BOX_PARAMETERS:
                box_args[key] = kwargs.pop(key)
        return box_args

    @classmethod
    def _from_data(cls, data, box_args, source):
        if not isinstance(data, list):
            raise BoxError(
                f"BoxList.{source} expected a list, got {type(data).__name__}"
            )
        return cls(data, **box_args)

    def to_json(
        self,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        multiline: bool = False,
        **json_kwargs,
    ):
        if filename and multiline:
            lines = [
                _to_json(
                    value,
                    filename=None,
                    encoding=encoding,
                    errors=errors,
                    **json_kwargs,
                )
                for value in self
            ]
            with open(filename, "w", encoding=encoding, errors=errors) as file:
                file.write("\n".join(lines))
        else:
            return _to_json(
                self.to_list(),
                filename=filename,
                encoding=encoding,
                errors=errors,
                **json_kwargs,
            )

    @classmethod
    def from_json(
        cls,
        json_string: str | None = None,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        multiline: bool = False,
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_json(
            json_string=json_string,
            filename=filename,
            encoding=encoding,
            errors=errors,
            multiline=multiline,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_json")

    def to_yaml(
        self,
        filename: str | PathLike | None = None,
        default_flow_style: bool = False,
        encoding: str = "utf-8",
        errors: str = "strict",
        **yaml_kwargs,
    ):
        return _to_yaml(
            self.to_list(),
            filename=filename,
            default_flow_style=default_flow_style,
            encoding=encoding,
            errors=errors,
            **yaml_kwargs,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str | None = None,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_yaml(
            yaml_string=yaml_string,
            filename=filename,
            encoding=encoding,
            errors=errors,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_yaml")

    def to_toml(
        self,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **toml_kwargs,
    ):
        return _to_toml(
            self.to_list(),
            filename=filename,
            encoding=encoding,
            errors=errors,
            **toml_kwargs,
        )

    @classmethod
    def from_toml(
        cls,
        toml_string: str | None = None,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_toml(
            toml_string=toml_string,
            filename=filename,
            encoding=encoding,
            errors=errors,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_toml")

    def to_msgpack(
        self,
        filename: str | PathLike | None = None,
        **msgpack_kwargs,
    ):
        return _to_msgpack(
            self.to_list(),
            filename=filename,
            **msgpack_kwargs,
        )

    @classmethod
    def from_msgpack(
        cls,
        msgpack_bytes: bytes | None = None,
        filename: str | PathLike | None = None,
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_msgpack(
            msgpack_bytes=msgpack_bytes,
            filename=filename,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_msgpack")

    def to_csv(
        self,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **csv_kwargs,
    ):
        return _to_csv(
            self.to_list(),
            filename=filename,
            encoding=encoding,
            errors=errors,
            **csv_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_csv(
            filename=filename,
            encoding=encoding,
            errors=errors,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_csv")

    def to_toon(
        self,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **toon_kwargs,
    ):
        return _to_toon(
            self.to_list(),
            filename=filename,
            encoding=encoding,
            errors=errors,
            **toon_kwargs,
        )

    @classmethod
    def from_toon(
        cls,
        toon_string: str | None = None,
        filename: str | PathLike | None = None,
        encoding: str = "utf-8",
        errors: str = "strict",
        **kwargs,
    ):
        box_args = cls._box_arguments(kwargs)
        data = _from_toon(
            toon_string=toon_string,
            filename=filename,
            encoding=encoding,
            errors=errors,
            **kwargs,
        )
        return cls._from_data(data, box_args, "from_toon")