from __future__ import annotations

import abc
import collections
import copy
import datetime as dt
import decimal
import email.utils
import ipaddress
import math
import numbers
import typing
import uuid
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

try:
    from backports.datetime_fromisoformat import MonkeyPatch
except ImportError:
    pass
else:
    MonkeyPatch.patch_fromisoformat()

from marshmallow import class_registry, types, utils, validate
from marshmallow.constants import missing as missing_
from marshmallow.exceptions import (
    StringNotCollectionError,
    ValidationError,
    _FieldInstanceResolutionError,
)
from marshmallow.validate import And, Length

if typing.TYPE_CHECKING:
    from marshmallow.schema import Schema, SchemaMeta

__all__ = [
    "IP",
    "URL",
    "UUID",
    "AwareDateTime",
    "Bool",
    "Boolean",
    "Constant",
    "Date",
    "DateTime",
    "Decimal",
    "Dict",
    "Email",
    "Enum",
    "Field",
    "Float",
    "Function",
    "IPInterface",
    "IPv4",
    "IPv4Interface",
    "IPv6",
    "IPv6Interface",
    "Int",
    "Integer",
    "List",
    "Mapping",
    "Method",
    "NaiveDateTime",
    "Nested",
    "Number",
    "Pluck",
    "Raw",
    "Str",
    "String",
    "Time",
    "TimeDelta",
    "Tuple",
    "Url",
]

_InternalT = typing.TypeVar("_InternalT")
_ProcessorT = typing.TypeVar("_ProcessorT")


class _BaseFieldKwargs(typing.TypedDict, total=False):
    load_default: typing.Any
    dump_default: typing.Any
    data_key: str | None
    attribute: str | None
    validate: typing.Any
    pre_load: typing.Any
    post_load: typing.Any
    required: bool
    allow_none: bool | None
    load_only: bool
    dump_only: bool
    error_messages: typing.Any
    metadata: typing.Mapping[str, typing.Any] | None


def _resolve_field_instance(cls_or_instance):
    if isinstance(cls_or_instance, type):
        if not issubclass(cls_or_instance, Field):
            raise _FieldInstanceResolutionError
        return cls_or_instance()
    if not isinstance(cls_or_instance, Field):
        raise _FieldInstanceResolutionError
    return cls_or_instance


class Field(typing.Generic[_InternalT]):
    _CHECK_ATTRIBUTE = True

    default_error_messages = {
        "required": "Missing data for required field.",
        "null": "Field may not be null.",
        "validator_failed": "Invalid value.",
    }

    def __init__(
        self,
        *,
        load_default=missing_,
        dump_default=missing_,
        data_key=None,
        attribute=None,
        validate=None,
        pre_load=None,
        post_load=None,
        required=False,
        allow_none=None,
        load_only=False,
        dump_only=False,
        error_messages=None,
        metadata=None,
    ):
        self.dump_default = dump_default
        self.load_default = load_default
        self.attribute = attribute
        self.data_key = data_key
        self.validate = validate
        self.validators = self._normalize_processors(validate, param="validate")
        self.pre_load = self._normalize_processors(pre_load, param="pre_load")
        self.post_load = self._normalize_processors(post_load, param="post_load")
        self.allow_none = load_default is None if allow_none is None else allow_none
        self.load_only = load_only
        self.dump_only = dump_only
        if required and load_default is not missing_:
            raise ValueError("'load_default' must not be set for required fields.")
        self.required = required
        self.metadata = metadata or {}
        messages = {}
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, "default_error_messages", {}))
        messages.update(error_messages or {})
        self.error_messages = messages
        self.parent = None
        self.name = None
        self.root = None

    @staticmethod
    def _normalize_processors(processors, *, param):
        if processors is None:
            return []
        if callable(processors):
            return [processors]
        try:
            return list(processors)
        except TypeError as error:
            raise ValueError(
                f'"{param}" must be a callable or a collection of callables.'
            ) from error

    def _bind_to_schema(self, field_name, schema):
        self.parent = self.root = schema
        self.name = field_name
        if self.data_key is None:
            self.data_key = field_name

    @property
    def context(self):
        if self.root is not None:
            return getattr(self.root, "context", {})
        return {}

    def _validate_missing(self, value):
        if value is missing_:
            if self.required:
                raise self.make_error("required")
        elif value is None and not self.allow_none:
            raise self.make_error("null")

    def _validate_all(self, value):
        return And(*self.validators)(value)

    def _validate(self, value):
        if not self.validators:
            return
        try:
            self._validate_all(value)
        except ValidationError as error:
            if error.field_name is None:
                error.field_name = self.name
            raise error

    def _deserialize(self, value, attr, data, **kwargs):
        return value

    def _serialize(self, value, attr, obj, **kwargs):
        return value

    def serialize(self, attr, obj, accessor=None, **kwargs):
        if self._CHECK_ATTRIBUTE:
            accessor = accessor or utils.get_value
            value = accessor(obj, attr, missing_)
        else:
            value = missing_
        if value is missing_:
            default = self.dump_default
            value = default() if callable(default) else default
        if value is missing_:
            return missing_
        return self._serialize(value, attr, obj, **kwargs)

    def deserialize(self, value, attr=None, data=None, **kwargs):
        self._validate_missing(value)
        if value is missing_:
            default = self.load_default
            return default() if callable(default) else default
        if value is None and self.allow_none:
            return None
        for processor in self.pre_load:
            value = processor(value)
        output = self._deserialize(value, attr, data, **kwargs)
        self._validate(output)
        for processor in self.post_load:
            output = processor(output)
        return output

    def _validate_all_processors(self, value):
        return self._validate_all(value)

    def make_error(self, key, **kwargs):
        try:
            message = self.error_messages[key]
        except KeyError as error:
            raise AssertionError(f"Invalid error message key: {key}") from error
        if isinstance(message, (list, tuple)):
            message = [m.format(**kwargs) for m in message]
        else:
            message = message.format(**kwargs)
        return ValidationError(message)

    def __deepcopy__(self, memo):
        return copy.copy(self)


class Raw(Field):
    pass


class String(Field[str]):
    default_error_messages = {
        "invalid": "Not a valid string.",
        "invalid_utf8": "Not a valid utf-8 string.",
    }

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, str):
            return str(value)
        return value

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, (str, bytes)):
            raise self.make_error("invalid")
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise self.make_error("invalid_utf8") from error
        return value


Str = String


class UUID(String):
    default_error_messages = {"invalid_uuid": "Not a valid UUID."}

    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError) as error:
            raise self.make_error("invalid_uuid") from error


class Number(Field):
    num_type = float
    as_string = False
    default_error_messages = {"invalid": "Not a valid number.", "too_large": "Number too large."}

    def __init__(self, *, as_string=False, **kwargs):
        self.as_string = as_string
        super().__init__(**kwargs)

    def _format_num(self, value):
        return self.num_type(value)

    def _validated(self, value):
        if isinstance(value, bool):
            raise self.make_error("invalid")
        try:
            return self._format_num(value)
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid") from error

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        value = self._validated(value)
        if self.as_string:
            return str(value)
        return value

    def _deserialize(self, value, attr, data, **kwargs):
        return self._validated(value)


class Integer(Number):
    num_type = int

    def _format_num(self, value):
        return int(value)


Int = Integer


class Float(Number):
    num_type = float

    def __init__(self, *, allow_nan=False, as_string=False, **kwargs):
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    def _validated(self, value):
        value = super()._validated(value)
        if not self.allow_nan and (math.isnan(value) or math.isinf(value)):
            raise self.make_error("special")
        return value

    default_error_messages = {
        "special": "Special numeric values (nan or infinity) are not permitted."
    }


class Decimal(Number):
    num_type = decimal.Decimal
    default_error_messages = {
        "invalid": "Not a valid number.",
        "too_large": "Number too large.",
        "special": "Special numeric values (nan or infinity) are not permitted.",
    }

    def __init__(
        self,
        places=None,
        rounding=None,
        *,
        allow_nan=False,
        as_string=False,
        **kwargs,
    ):
        self.places = places
        self.rounding = rounding
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    def _format_num(self, value):
        if isinstance(value, float):
            value = str(value)
        value = decimal.Decimal(value)
        if self.places is not None and value.is_finite():
            exp = decimal.Decimal((0, (1,), -self.places))
            value = value.quantize(exp, rounding=self.rounding)
        return value

    def _validated(self, value):
        value = super()._validated(value)
        if not self.allow_nan and not value.is_finite():
            raise self.make_error("special")
        return value

    def _serialize(self, value, attr, obj, **kwargs):
        value = super()._serialize(value, attr, obj, **kwargs)
        if value is None:
            return None
        return str(value) if self.as_string else value


class Boolean(Field[bool]):
    truthy = {"true", "True", "TRUE", "1", 1, True}
    falsy = {"false", "False", "FALSE", "0", 0, False}
    default_error_messages = {"invalid": "Not a valid boolean."}

    def __init__(self, *, truthy=None, falsy=None, **kwargs):
        super().__init__(**kwargs)
        self.truthy = set(truthy) if truthy is not None else self.truthy
        self.falsy = set(falsy) if falsy is not None else self.falsy

    def _serialize(self, value, attr, obj, **kwargs):
        return bool(value)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            if value in self.truthy:
                return True
            if value in self.falsy:
                return False
        except TypeError:
            pass
        raise self.make_error("invalid")


Bool = Boolean


class DateTime(Field[dt.datetime]):
    SERIALIZATION_FUNCS = {
        "iso": lambda value: value.isoformat(),
        "iso8601": lambda value: value.isoformat(),
        "rfc": lambda value: email.utils.format_datetime(value),
        "rfc822": lambda value: email.utils.format_datetime(value),
        "timestamp": lambda value: value.timestamp(),
        "timestamp_ms": lambda value: value.timestamp() * 1000,
    }
    DESERIALIZATION_FUNCS = {
        "iso": lambda value: dt.datetime.fromisoformat(value),
        "iso8601": lambda value: dt.datetime.fromisoformat(value),
        "rfc": lambda value: email.utils.parsedate_to_datetime(value),
        "rfc822": lambda value: email.utils.parsedate_to_datetime(value),
        "timestamp": lambda value: dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc),
        "timestamp_ms": lambda value: dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.timezone.utc),
    }
    DEFAULT_FORMAT = "iso"
    default_error_messages = {
        "invalid": "Not a valid datetime.",
        "format": '"{input}" cannot be formatted as a datetime.',
    }

    def __init__(self, format=None, **kwargs):
        super().__init__(**kwargs)
        self.format = format

    def _get_format(self):
        if self.format is not None:
            return self.format
        opts = getattr(self.root, "opts", None)
        return getattr(opts, "datetimeformat", None) or self.DEFAULT_FORMAT

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, dt.datetime):
            raise self.make_error("invalid", input=value)
        fmt = self._get_format()
        if fmt in self.SERIALIZATION_FUNCS:
            return self.SERIALIZATION_FUNCS[fmt](value)
        return value.strftime(fmt)

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dt.datetime):
            return value
        if not isinstance(value, (str, int, float)):
            raise self.make_error("invalid", input=value)
        fmt = self._get_format()
        try:
            if fmt in self.DESERIALIZATION_FUNCS:
                return self.DESERIALIZATION_FUNCS[fmt](value)
            return dt.datetime.strptime(value, fmt)
        except (TypeError, ValueError, OverflowError) as error:
            raise self.make_error("invalid", input=value) from error


class NaiveDateTime(DateTime):
    def __init__(self, format=None, *, timezone=None, **kwargs):
        self.timezone = timezone
        super().__init__(format=format, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        value = super()._deserialize(value, attr, data, **kwargs)
        if value.tzinfo is not None:
            if self.timezone is not None:
                value = value.astimezone(self.timezone)
            value = value.replace(tzinfo=None)
        return value


class AwareDateTime(DateTime):
    default_error_messages = {
        "invalid": "Not a valid datetime.",
        "format": '"{input}" cannot be formatted as a datetime.',
        "invalid_awareness": "Not a valid aware datetime.",
    }

    def __init__(self, format=None, *, default_timezone=None, **kwargs):
        self.default_timezone = default_timezone
        super().__init__(format=format, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        value = super()._deserialize(value, attr, data, **kwargs)
        if value.tzinfo is None:
            if self.default_timezone is None:
                raise self.make_error("invalid_awareness", input=value)
            value = value.replace(tzinfo=self.default_timezone)
        return value


class Date(Field[dt.date]):
    default_error_messages = {"invalid": "Not a valid date.", "format": '"{input}" cannot be formatted as a date.'}

    def __init__(self, format=None, **kwargs):
        self.format = format
        super().__init__(**kwargs)

    def _get_format(self):
        if self.format is not None:
            return self.format
        return getattr(getattr(self.root, "opts", None), "dateformat", None) or "iso"

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, dt.date):
            raise self.make_error("invalid", input=value)
        fmt = self._get_format()
        return value.isoformat() if fmt in ("iso", "iso8601") else value.strftime(fmt)

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        try:
            fmt = self._get_format()
            return dt.date.fromisoformat(value) if fmt in ("iso", "iso8601") else dt.datetime.strptime(value, fmt).date()
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid", input=value) from error


class Time(Field[dt.time]):
    default_error_messages = {"invalid": "Not a valid time.", "format": '"{input}" cannot be formatted as a time.'}

    def __init__(self, format=None, **kwargs):
        self.format = format
        super().__init__(**kwargs)

    def _get_format(self):
        if self.format is not None:
            return self.format
        return getattr(getattr(self.root, "opts", None), "timeformat", None) or "iso"

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, dt.time):
            raise self.make_error("invalid", input=value)
        fmt = self._get_format()
        return value.isoformat() if fmt in ("iso", "iso8601") else value.strftime(fmt)

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dt.time):
            return value
        try:
            fmt = self._get_format()
            return dt.time.fromisoformat(value) if fmt in ("iso", "iso8601") else dt.datetime.strptime(value, fmt).time()
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid", input=value) from error


class TimeDelta(Field[dt.timedelta]):
    DAYS = "days"
    SECONDS = "seconds"
    MICROSECONDS = "microseconds"
    MILLISECONDS = "milliseconds"
    MINUTES = "minutes"
    HOURS = "hours"
    WEEKS = "weeks"
    default_error_messages = {"invalid": "Not a valid timedelta."}

    def __init__(self, precision="seconds", serialization_type=float, **kwargs):
        self.precision = precision
        self.serialization_type = serialization_type
        super().__init__(**kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return dt.timedelta(**{self.precision: float(value)})
        except (TypeError, ValueError, OverflowError) as error:
            raise self.make_error("invalid") from error

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, dt.timedelta):
            raise self.make_error("invalid")
        factor = {
            self.DAYS: 86400,
            self.SECONDS: 1,
            self.MICROSECONDS: 0.000001,
            self.MILLISECONDS: 0.001,
            self.MINUTES: 60,
            self.HOURS: 3600,
            self.WEEKS: 604800,
        }.get(self.precision)
        if factor is None:
            raise self.make_error("invalid")
        amount = value.total_seconds() / factor
        return self.serialization_type(amount)


class Mapping(Field):
    mapping_type = dict
    default_error_messages = {
        "invalid": "Not a valid mapping type.",
        "invalid_keys": "Not a valid mapping type.",
    }

    def __init__(self, keys=None, values=None, **kwargs):
        self.keys = _resolve_field_instance(keys) if keys is not None else None
        self.values = _resolve_field_instance(values) if values is not None else None
        super().__init__(**kwargs)

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        if self.keys is not None:
            self.keys = copy.deepcopy(self.keys)
            self.keys._bind_to_schema(field_name, schema)
        if self.values is not None:
            self.values = copy.deepcopy(self.values)
            self.values._bind_to_schema(field_name, schema)

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, _Mapping):
            raise self.make_error("invalid")
        result = self.mapping_type()
        for key, val in value.items():
            key = self.keys._serialize(key, None, value, **kwargs) if self.keys else key
            val = self.values._serialize(val, key, value, **kwargs) if self.values else val
            if val is not missing_:
                result[key] = val
        return result

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, _Mapping):
            raise self.make_error("invalid")
        result = self.mapping_type()
        errors = {}
        for key, val in value.items():
            try:
                new_key = self.keys.deserialize(key, attr=None, data=value, **kwargs) if self.keys else key
            except ValidationError as error:
                errors[key] = error.messages
                continue
            try:
                new_value = self.values.deserialize(val, attr=key, data=value, **kwargs) if self.values else val
            except ValidationError as error:
                errors[key] = error.messages
                continue
            result[new_key] = new_value
        if errors:
            raise ValidationError(errors)
        return result


class Dict(Mapping):
    pass


class List(Field[list]):
    default_error_messages = {"invalid": "Not a valid list."}

    def __init__(self, cls_or_instance, **kwargs):
        self.inner = _resolve_field_instance(cls_or_instance)
        super().__init__(**kwargs)

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        self.inner = copy.deepcopy(self.inner)
        self.inner._bind_to_schema(field_name, schema)

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, collections.abc.Collection) or isinstance(value, (str, bytes, _Mapping)):
            raise self.make_error("invalid")
        result = []
        for each in value:
            result.append(self.inner._serialize(each, attr, obj, **kwargs))
        return result

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, (str, bytes)) or not isinstance(value, collections.abc.Collection):
            raise self.make_error("invalid")
        result = []
        errors = {}
        for index, each in enumerate(value):
            try:
                result.append(self.inner.deserialize(each, attr=index, data=value, **kwargs))
            except ValidationError as error:
                errors[index] = error.messages
        if errors:
            raise ValidationError(errors)
        return result


class Tuple(List):
    default_error_messages = {
        "invalid": "Not a valid tuple.",
        "length": "Length must be {expected}.",
    }

    def __init__(self, tuple_fields, **kwargs):
        self.tuple_fields = [_resolve_field_instance(field) for field in tuple_fields]
        Field.__init__(self, **kwargs)

    def _bind_to_schema(self, field_name, schema):
        Field._bind_to_schema(self, field_name, schema)
        self.tuple_fields = [copy.deepcopy(field) for field in self.tuple_fields]
        for field in self.tuple_fields:
            field._bind_to_schema(field_name, schema)

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, (tuple, list)):
            raise self.make_error("invalid")
        if len(value) != len(self.tuple_fields):
            raise self.make_error("length", expected=len(self.tuple_fields))
        return [
            field._serialize(item, attr, obj, **kwargs)
            for field, item in zip(self.tuple_fields, value)
        ]

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, (tuple, list)):
            raise self.make_error("invalid")
        if len(value) != len(self.tuple_fields):
            raise self.make_error("length", expected=len(self.tuple_fields))
        result = []
        errors = {}
        for index, (field, item) in enumerate(zip(self.tuple_fields, value)):
            try:
                result.append(field.deserialize(item, attr=index, data=value, **kwargs))
            except ValidationError as error:
                errors[index] = error.messages
        if errors:
            raise ValidationError(errors)
        return tuple(result)


class Nested(Field):
    default_error_messages = {
        "type": "Invalid type.",
        "invalid": "Invalid value.",
    }

    def __init__(
        self,
        nested,
        *,
        only=None,
        exclude=(),
        many=False,
        unknown=None,
        **kwargs,
    ):
        self.nested = nested
        self.only = only
        self.exclude = exclude
        self.many = many
        self.unknown = unknown
        self._schema = None
        super().__init__(**kwargs)

    @property
    def schema(self):
        if self._schema is not None:
            return self._schema
        nested = self.nested
        if callable(nested) and not isinstance(nested, type):
            nested = nested()
        if isinstance(nested, str):
            if nested == "self":
                nested = self.root.__class__
            else:
                nested = class_registry.get_class(nested)
        if isinstance(nested, _Mapping):
            from marshmallow.schema import Schema
            nested = Schema.from_dict(nested)
        if isinstance(nested, type):
            self._schema = nested(
                many=self.many,
                only=self.only,
                exclude=self.exclude,
                unknown=self.unknown,
            )
        else:
            self._schema = copy.copy(nested)
            self._schema.many = self.many
            if self.only is not None:
                self._schema.only = self.only
            if self.exclude:
                self._schema.exclude = self.exclude
            if self.unknown is not None:
                self._schema.unknown = self.unknown
        return self._schema

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        self._schema = None

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return self.schema.dump(value, many=self.many)

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        if value is None:
            return None
        try:
            return self.schema.load(value, many=self.many, partial=partial, unknown=self.unknown)
        except ValidationError:
            raise


class Pluck(Nested):
    def __init__(self, nested, field, *, many=False, **kwargs):
        self.field = field
        super().__init__(nested, only=(field,), many=many, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        ret = super()._serialize(value, attr, obj, **kwargs)
        if ret is None:
            return None
        if self.many:
            return [item.get(self.field) for item in ret]
        return ret.get(self.field)

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        if value is None:
            return None
        if self.many:
            value = [{self.field: item} for item in value]
        else:
            value = {self.field: value}
        return super()._deserialize(value, attr, data, partial=partial, **kwargs)


class Constant(Field):
    _CHECK_ATTRIBUTE = False

    def __init__(self, constant, *, dump_only=True, **kwargs):
        self.constant = constant
        super().__init__(dump_only=dump_only, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        return self.constant

    def _deserialize(self, value, attr, data, **kwargs):
        return self.constant


class Method(Field):
    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize=None, deserialize=None, **kwargs):
        self.serialize_method_name = serialize
        self.deserialize_method_name = deserialize
        self._serialize_method = None
        self._deserialize_method = None
        super().__init__(**kwargs)

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        if self.serialize_method_name:
            self._serialize_method = getattr(schema, self.serialize_method_name)
        if self.deserialize_method_name:
            self._deserialize_method = getattr(schema, self.deserialize_method_name)

    def _serialize(self, value, attr, obj, **kwargs):
        if self._serialize_method is not None:
            return self._serialize_method(obj)
        return missing_

    def _deserialize(self, value, attr, data, **kwargs):
        if self._deserialize_method is not None:
            return self._deserialize_method(value)
        return value


class Function(Field):
    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize=None, deserialize=None, **kwargs):
        self.serialize_func = serialize
        self.deserialize_func = deserialize
        super().__init__(**kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if self.serialize_func is None:
            return missing_
        try:
            return self.serialize_func(obj, self.context)
        except TypeError:
            return self.serialize_func(obj)

    def _deserialize(self, value, attr, data, **kwargs):
        if self.deserialize_func is None:
            return value
        try:
            return self.deserialize_func(value, self.context)
        except TypeError:
            return self.deserialize_func(value)


class Url(String):
    def __init__(self, *, relative=False, absolute=True, schemes=None, require_tld=True, **kwargs):
        validators = kwargs.pop("validate", None)
        url_validator = validate.URL(
            relative=relative,
            absolute=absolute,
            schemes=schemes,
            require_tld=require_tld,
        )
        if validators is None:
            validators = [url_validator]
        elif callable(validators):
            validators = [validators, url_validator]
        else:
            validators = list(validators) + [url_validator]
        super().__init__(validate=validators, **kwargs)


URL = Url


class Email(String):
    def __init__(self, *args, **kwargs):
        validators = kwargs.pop("validate", None)
        email_validator = validate.Email()
        if validators is None:
            validators = [email_validator]
        elif callable(validators):
            validators = [validators, email_validator]
        else:
            validators = list(validators) + [email_validator]
        super().__init__(*args, validate=validators, **kwargs)


class IP(Field):
    default_error_messages = {"invalid_ip": "Not a valid IP address."}
    ip_class = ipaddress.ip_address

    def __init__(self, *, exploded=False, **kwargs):
        self.exploded = exploded
        super().__init__(**kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return self.ip_class(value)
        except (ValueError, TypeError) as error:
            raise self.make_error("invalid_ip") from error

    def _serialize(self, value, attr, obj, **kwargs):
        try:
            value = self.ip_class(value)
        except (ValueError, TypeError) as error:
            raise self.make_error("invalid_ip") from error
        return value.exploded if self.exploded else str(value)


class IPv4(IP):
    ip_class = ipaddress.IPv4Address


class IPv6(IP):
    ip_class = ipaddress.IPv6Address


class IPInterface(IP):
    ip_class = ipaddress.ip_interface


class IPv4Interface(IPInterface):
    ip_class = ipaddress.IPv4Interface


class IPv6Interface(IPInterface):
    ip_class = ipaddress.IPv6Interface


class Enum(Field):
    default_error_messages = {
        "unknown": "Must be one of: {choices}.",
        "by_value": "Must be one of: {choices}.",
    }

    def __init__(self, enum, *, by_value=False, **kwargs):
        if not isinstance(enum, type) or not issubclass(enum, EnumType):
            raise ValueError('"enum" must be an Enum type.')
        self.enum = enum
        self.by_value = by_value
        if by_value is True:
            self.field = Raw()
        elif isinstance(by_value, Field):
            self.field = by_value
        else:
            self.field = String()
        super().__init__(**kwargs)

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        self.field = copy.deepcopy(self.field)
        self.field._bind_to_schema(field_name, schema)

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, self.enum):
            try:
                value = self.enum(value)
            except (ValueError, TypeError) as error:
                raise self.make_error("unknown", choices=", ".join(e.name for e in self.enum)) from error
        return self.field._serialize(value.value if self.by_value else value.name, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        value = self.field.deserialize(value, attr, data, **kwargs)
        try:
            return self.enum(value) if self.by_value else self.enum[value]
        except (KeyError, ValueError, TypeError) as error:
            choices = ", ".join(
                str(member.value if self.by_value else member.name) for member in self.enum
            )
            raise self.make_error("by_value" if self.by_value else "unknown", choices=choices) from error