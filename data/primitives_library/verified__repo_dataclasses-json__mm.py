import sys
import typing
import warnings
from collections import abc as collections_abc
from copy import deepcopy
from dataclasses import MISSING, fields as dc_fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID

from typing_inspect import is_union_type

from marshmallow import Schema, fields, post_load
from marshmallow.exceptions import ValidationError

from dataclasses_json.core import (
    _ExtendedEncoder,
    _decode_dataclass,
    _is_supported_generic,
    _user_overrides_or_exts,
)
from dataclasses_json.utils import (
    CatchAllVar,
    _get_type_origin,
    _handle_undefined_parameters_safe,
    _is_collection,
    _is_new_type,
    _is_optional,
)

try:
    from dataclasses_json.utils import _issubclass_safe
except ImportError:
    def _issubclass_safe(cls, class_or_tuple):
        try:
            return issubclass(cls, class_or_tuple)
        except TypeError:
            return False


try:
    from dataclasses_json.utils import _timestamp_to_dt_aware
except ImportError:
    def _timestamp_to_dt_aware(timestamp):
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)


class _TimestampField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is not None:
            return value.timestamp()
        if not self.required:
            return None
        raise ValidationError(self.default_error_messages["required"])

    def _deserialize(self, value, attr, data, **kwargs):
        if value is not None:
            return _timestamp_to_dt_aware(value)
        if not self.required:
            return None
        raise ValidationError(self.default_error_messages["required"])


class _IsoField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is not None:
            return value.isoformat()
        if not self.required:
            return None
        raise ValidationError(self.default_error_messages["required"])

    def _deserialize(self, value, attr, data, **kwargs):
        if value is not None:
            return datetime.fromisoformat(value)
        if not self.required:
            return None
        raise ValidationError(self.default_error_messages["required"])


class _UnionField(fields.Field):
    def __init__(self, desc, cls, field, *args, **kwargs):
        self.desc = desc
        self.cls = cls
        self.field = field
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if self.allow_none and value is None:
            return None

        for type_, schema_ in self.desc.items():
            if _issubclass_safe(type(value), type_):
                if is_dataclass(value):
                    result = schema_._serialize(value, attr, obj, **kwargs)
                    result["__type"] = str(type_.__name__)
                    return result
                break
            elif isinstance(value, _get_type_origin(type_)):
                return schema_._serialize(value, attr, obj, **kwargs)
        else:
            warnings.warn(
                f'The type "{type(value).__name__}" (value: "{value}") '
                f"is not in the list of possible types of typing.Union "
                f"(dataclass: {self.cls.__name__}, field: {self.field.name}). "
                f"Value cannot be serialized properly."
            )

        return super()._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        tmp_value = deepcopy(value)

        if isinstance(tmp_value, dict) and "__type" in tmp_value:
            dc_name = tmp_value["__type"]
            for type_, schema_ in self.desc.items():
                if is_dataclass(type_) and type_.__name__ == dc_name:
                    del tmp_value["__type"]
                    return schema_._deserialize(tmp_value, attr, data, **kwargs)
        elif isinstance(tmp_value, dict):
            warnings.warn(
                f'Attempting to deserialize "dict" (value: "{tmp_value}) '
                f'that does not have a "__type" type specifier field into'
                f"(dataclass: {self.cls.__name__}, field: {self.field.name})."
                f"Deserialization may fail, or deserialization to wrong type may occur."
            )
            return super()._deserialize(tmp_value, attr, data, **kwargs)
        else:
            for type_, schema_ in self.desc.items():
                if isinstance(tmp_value, _get_type_origin(type_)):
                    return schema_._deserialize(tmp_value, attr, data, **kwargs)
            else:
                warnings.warn(
                    f'The type "{type(tmp_value).__name__}" (value: "{tmp_value}") '
                    f"is not in the list of possible types of typing.Union "
                    f"(dataclass: {self.cls.__name__}, field: {self.field.name}). "
                    f"Value cannot be deserialized properly."
                )
            return super()._deserialize(tmp_value, attr, data, **kwargs)


class _TupleVarLen(fields.List):
    def _deserialize(self, value, attr, data, **kwargs):
        optional_list = super()._deserialize(value, attr, data, **kwargs)
        return None if optional_list is None else tuple(optional_list)


TYPES = {
    typing.Mapping: fields.Mapping,
    typing.MutableMapping: fields.Mapping,
    typing.List: fields.List,
    typing.Dict: fields.Dict,
    typing.Tuple: fields.Tuple,
    typing.Callable: fields.Function,
    typing.Any: fields.Raw,
    dict: fields.Dict,
    list: fields.List,
    tuple: fields.Tuple,
    str: fields.Str,
    int: fields.Int,
    float: fields.Float,
    bool: fields.Bool,
    datetime: _TimestampField,
    UUID: fields.UUID,
    Decimal: fields.Decimal,
    CatchAllVar: fields.Dict,
}

A = typing.TypeVar("A")
JsonData = typing.Union[str, bytes, bytearray]
TEncoded = typing.Dict[str, typing.Any]
TOneOrMulti = typing.Union[typing.List[A], A]
TOneOrMultiEncoded = typing.Union[typing.List[TEncoded], TEncoded]


if sys.version_info >= (3, 7) or typing.TYPE_CHECKING:
    class SchemaF(Schema, typing.Generic[A]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raise NotImplementedError()

        @typing.overload
        def dump(self, obj: typing.List[A], many: typing.Optional[bool] = None) -> typing.List[TEncoded]:
            pass

        @typing.overload
        def dump(self, obj: A, many: typing.Optional[bool] = None) -> TEncoded:
            pass

        def dump(self, obj: TOneOrMulti, many: typing.Optional[bool] = None) -> TOneOrMultiEncoded:
            pass

        @typing.overload
        def dumps(self, obj: typing.List[A], many: typing.Optional[bool] = None, *args, **kwargs) -> str:
            pass

        @typing.overload
        def dumps(self, obj: A, many: typing.Optional[bool] = None, *args, **kwargs) -> str:
            pass

        def dumps(self, obj: TOneOrMulti, many: typing.Optional[bool] = None, *args, **kwargs) -> str:
            pass

        @typing.overload
        def load(
            self,
            data: typing.List[TEncoded],
            many: bool = True,
            partial: typing.Optional[bool] = None,
            unknown: typing.Optional[str] = None,
        ) -> typing.List[A]:
            pass

        @typing.overload
        def load(
            self,
            data: TEncoded,
            many: None = None,
            partial: typing.Optional[bool] = None,
            unknown: typing.Optional[str] = None,
        ) -> A:
            pass

        def load(
            self,
            data: TOneOrMultiEncoded,
            many: typing.Optional[bool] = None,
            partial: typing.Optional[bool] = None,
            unknown: typing.Optional[str] = None,
        ) -> TOneOrMulti:
            pass

        @typing.overload
        def loads(
            self,
            json_data: JsonData,
            many: typing.Optional[bool] = True,
            partial: typing.Optional[bool] = None,
            unknown: typing.Optional[str] = None,
            **kwargs,
        ) -> typing.List[A]:
            pass

        def loads(
            self,
            json_data: JsonData,
            many: typing.Optional[bool] = None,
            partial: typing.Optional[bool] = None,
            unknown: typing.Optional[str] = None,
            **kwargs,
        ) -> TOneOrMulti:
            pass
else:
    SchemaF = Schema


SchemaType = typing.Type[SchemaF[A]]


def _type_args(type_):
    try:
        return typing.get_args(type_)
    except AttributeError:
        return getattr(type_, "__args__", ())


def _field_options(field, override, infer_missing, partial):
    required = (
        field.default is MISSING
        and field.default_factory is MISSING
        and not _is_optional(field.type)
        and not infer_missing
        and not partial
    )
    options = {"required": required}

    if _is_optional(field.type) or infer_missing:
        options["allow_none"] = True

    letter_case = getattr(override, "letter_case", None)
    if letter_case is not None:
        options["data_key"] = letter_case(field.name)

    return options


def _new_type_supertype(type_):
    return getattr(type_, "__supertype__", type_)


def _is_union(type_):
    try:
        if is_union_type(type_):
            return True
    except TypeError:
        pass
    union_type = getattr(typing, "UnionType", None)
    return union_type is not None and isinstance(type_, union_type)


def _make_type_field(type_, options, mixin, field, cls):
    if type_ is None:
        return fields.Raw(**options)

    if _is_new_type(type_):
        return _make_type_field(_new_type_supertype(type_), options, mixin, field, cls)

    if _is_union(type_):
        union_desc = {
            union_type: _make_type_field(union_type, {}, mixin, field, cls)
            for union_type in _type_args(type_)
        }
        return _UnionField(union_desc, cls, field, **options)

    if type_ in TYPES:
        return TYPES[type_](**options)

    if _issubclass_safe(type_, Enum):
        enum_field = getattr(fields, "Enum", None)
        if enum_field is not None:
            return enum_field(type_, **options)
        return fields.Function(
            serialize=lambda value: value.value if value is not None else None,
            deserialize=lambda value: type_(value) if value is not None else None,
            **options,
        )

    if is_dataclass(type_):
        return fields.Nested(build_schema(type_, mixin), **options)

    if _is_supported_generic(type_):
        generic_args = _type_args(type_)
        generic_type = _get_type_origin(type_)

        if _issubclass_safe(generic_type, tuple):
            if len(generic_args) == 2 and generic_args[1] is Ellipsis:
                return _TupleVarLen(
                    _make_type_field(generic_args[0], {}, mixin, field, cls),
                    **options,
                )
            if generic_args:
                return fields.Tuple(
                    tuple(
                        _make_type_field(arg, {}, mixin, field, cls)
                        for arg in generic_args
                    ),
                    **options,
                )
            return fields.Tuple(**options)

        if _issubclass_safe(
            generic_type,
            (
                dict,
                typing.Mapping,
                typing.MutableMapping,
                collections_abc.Mapping,
                collections_abc.MutableMapping,
            ),
        ):
            key_field = (
                _make_type_field(generic_args[0], {}, mixin, field, cls)
                if len(generic_args) > 0
                else None
            )
            value_field = (
                _make_type_field(generic_args[1], {}, mixin, field, cls)
                if len(generic_args) > 1
                else None
            )
            return fields.Dict(keys=key_field, values=value_field, **options)

        if _issubclass_safe(
            generic_type,
            (
                list,
                set,
                frozenset,
                typing.List,
                typing.Set,
                typing.FrozenSet,
                typing.Sequence,
                typing.MutableSequence,
                typing.MutableSet,
                collections_abc.Sequence,
                collections_abc.MutableSequence,
                collections_abc.Set,
                collections_abc.MutableSet,
            ),
        ):
            inner = (
                _make_type_field(generic_args[0], {}, mixin, field, cls)
                if generic_args
                else fields.Raw()
            )
            return fields.List(inner, **options)

    return fields.Raw(**options)


def build_schema(cls, mixin, infer_missing=False, partial=False):
    schema_fields = {}
    overrides = _user_overrides_or_exts(cls)

    try:
        type_hints = typing.get_type_hints(cls)
    except (NameError, TypeError):
        type_hints = {}

    for field in dc_fields(cls):
        override = overrides.get(field.name)
        if override is None:
            continue

        options = _field_options(field, override, infer_missing, partial)

        encoder = getattr(override, "encoder", None)
        decoder = getattr(override, "decoder", None)
        if encoder is not None:
            options["serialize"] = encoder
        if decoder is not None:
            options["deserialize"] = decoder

        field_type = type_hints.get(field.name, field.type)
        schema_fields[field.name] = _make_type_field(
            field_type,
            options,
            mixin,
            field,
            cls,
        )

    unknown = _handle_undefined_parameters_safe(cls)

    class DataClassSchema(Schema):
        class Meta:
            unknown = unknown

        @post_load
        def make_instance(self, data, **kwargs):
            return _decode_dataclass(cls, data, infer_missing)

    DataClassSchema._declared_fields = schema_fields
    DataClassSchema.__name__ = f"{cls.__name__}Schema"
    DataClassSchema.__qualname__ = DataClassSchema.__name__
    return DataClassSchema