import copy
import json
import sys
import warnings
from collections import Counter, defaultdict, namedtuple
from collections.abc import (
    Collection as ABCCollection,
    Mapping as ABCMapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set,
)
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Collection,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)
from uuid import UUID

from typing_inspect import is_union_type

from dataclasses_json import cfg
from dataclasses_json.utils import (
    _NO_ARGS,
    _get_type_arg_param,
    _get_type_args,
    _get_type_cons,
    _get_type_origin,
    _handle_undefined_parameters_safe,
    _is_collection,
    _is_counter,
    _is_generic_dataclass,
    _is_mapping,
    _is_new_type,
    _is_optional,
    _is_tuple,
    _isinstance_safe,
    _issubclass_safe,
)

Json = Union[dict, list, str, int, float, bool, None]

confs = ["encoder", "decoder", "mm_field", "letter_case", "exclude"]
FieldOverride = namedtuple("FieldOverride", confs)

collections_abc_type_to_implementation_type = MappingProxyType(
    {
        ABCCollection: tuple,
        ABCMapping: dict,
        MutableMapping: dict,
        MutableSequence: list,
        MutableSet: set,
        Sequence: tuple,
        Set: frozenset,
    }
)


class _ExtendedEncoder(json.JSONEncoder):
    def default(self, o) -> Json:
        if _isinstance_safe(o, Collection):
            if _isinstance_safe(o, Mapping):
                return dict(o)
            return list(o)
        if _isinstance_safe(o, datetime):
            return o.timestamp()
        if _isinstance_safe(o, UUID):
            return str(o)
        if _isinstance_safe(o, Enum):
            return o.value
        if _isinstance_safe(o, Decimal):
            return str(o)
        return json.JSONEncoder.default(self, o)


def _user_overrides_or_exts(cls):
    global_metadata = defaultdict(dict)

    for field in fields(cls):
        if field.type in cfg.global_config.encoders:
            global_metadata[field.name]["encoder"] = cfg.global_config.encoders[
                field.type
            ]
        if field.type in cfg.global_config.decoders:
            global_metadata[field.name]["decoder"] = cfg.global_config.decoders[
                field.type
            ]
        if field.type in cfg.global_config.mm_fields:
            global_metadata[field.name]["mm_field"] = cfg.global_config.mm_fields[
                field.type
            ]

    try:
        class_config = cls.dataclass_json_config
        if class_config is None:
            class_config = {}
    except AttributeError:
        class_config = {}

    overrides = {}
    for field in fields(cls):
        configuration = {}
        configuration.update(global_metadata[field.name])
        configuration.update(class_config)
        configuration.update(field.metadata.get("dataclasses_json", {}))
        overrides[field.name] = FieldOverride(
            *(configuration.get(name) for name in confs)
        )

    return overrides


def _encode_json_type(value, default=_ExtendedEncoder().default):
    if isinstance(value, Json.__args__):
        if isinstance(value, list):
            return [_encode_json_type(item) for item in value]
        if isinstance(value, dict):
            return {key: _encode_json_type(item) for key, item in value.items()}
        return value
    return default(value)


def _encode_overrides(kvs, overrides, encode_json=False):
    result = {}

    for key, value in kvs.items():
        original_key = key
        if key in overrides:
            override = overrides[key]

            if override.exclude is not None and override.exclude(value):
                continue

            if override.letter_case is not None:
                key = override.letter_case(key)

            if key in result:
                raise ValueError(
                    "Multiple fields map to the same JSON key after letter "
                    f"case encoding: {key}"
                )

            if override.encoder is not None:
                value = override.encoder(value)

        if encode_json:
            value = _encode_json_type(value)

        result[key] = value

    return result


def _decode_letter_case_overrides(field_names, overrides):
    names = {}
    for field_name in field_names:
        override = overrides.get(field_name)
        if override is not None and override.letter_case is not None:
            names[override.letter_case(field_name)] = field_name
    return names


def _is_supported_generic(type_):
    return (
        _is_generic_dataclass(type_)
        or _is_collection(type_)
        or _is_mapping(type_)
        or is_union_type(type_)
    )


def _decode_type(type_, value, infer_missing):
    if type_ is Any:
        return value

    while _is_new_type(type_):
        type_ = type_.__supertype__

    if is_dataclass(type_):
        if is_dataclass(value):
            return value
        return _decode_dataclass(type_, value, infer_missing)

    if _is_supported_generic(type_) and type_ is not str:
        return _decode_generic(type_, value, infer_missing)

    if _issubclass_safe(type_, Enum):
        return type_(value)

    if _issubclass_safe(type_, datetime):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if _issubclass_safe(type_, Decimal):
        return Decimal(str(value))

    if _issubclass_safe(type_, UUID):
        return UUID(value)

    return value


def _collection_implementation(type_):
    implementation = _get_type_cons(type_)
    return collections_abc_type_to_implementation_type.get(
        implementation, implementation
    )


def _decode_generic(type_, value, infer_missing):
    if _is_generic_dataclass(type_):
        origin = _get_type_origin(type_)
        if origin is None:
            origin = _get_type_cons(type_)
        return _decode_dataclass(origin, value, infer_missing)

    if is_union_type(type_):
        arguments = _get_type_args(type_)

        if value is None:
            return None

        for argument in arguments:
            if argument is type(None):
                continue
            try:
                return _decode_type(argument, value, infer_missing)
            except (TypeError, ValueError, AttributeError, KeyError):
                continue

        return value

    if _is_mapping(type_):
        arguments = _get_type_args(type_)
        if arguments is _NO_ARGS:
            return value

        key_type = _get_type_arg_param(type_, 0)
        value_type = _get_type_arg_param(type_, 1)

        decoded = {
            _decode_type(key_type, key, infer_missing): _decode_type(
                value_type, item, infer_missing
            )
            for key, item in value.items()
        }

        if _is_counter(type_):
            return Counter(decoded)

        implementation = _collection_implementation(type_)
        try:
            return implementation(decoded)
        except TypeError:
            return implementation(decoded.items())

    if _is_collection(type_):
        arguments = _get_type_args(type_)

        if _is_tuple(type_):
            if arguments is _NO_ARGS:
                return value

            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return tuple(
                    _decode_type(arguments[0], item, infer_missing)
                    for item in value
                )

            return tuple(
                _decode_type(argument, item, infer_missing)
                for argument, item in zip(arguments, value)
            )

        if arguments is _NO_ARGS:
            return value

        item_type = _get_type_arg_param(type_, 0)
        implementation = _collection_implementation(type_)
        decoded_items = (
            _decode_type(item_type, item, infer_missing) for item in value
        )

        try:
            return implementation(decoded_items)
        except TypeError:
            return list(decoded_items)

    return value


def _decode_dataclass(cls, kvs, infer_missing):
    if _isinstance_safe(kvs, cls):
        return kvs

    overrides = _user_overrides_or_exts(cls)

    if kvs is None and infer_missing:
        kvs = {}

    field_names = [field.name for field in fields(cls)]
    decode_names = _decode_letter_case_overrides(field_names, overrides)
    kvs = {decode_names.get(key, key): value for key, value in kvs.items()}

    missing = {field for field in fields(cls) if field.name not in kvs}
    for field in missing:
        if field.default is not MISSING:
            kvs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kvs[field.name] = field.default_factory()
        elif infer_missing:
            kvs[field.name] = None

    kvs = _handle_undefined_parameters_safe(cls, kvs, usage="from")

    type_hints = get_type_hints(cls)
    init_kwargs = {}

    for field in fields(cls):
        if not field.init:
            continue

        field_value = kvs[field.name]
        field_type = type_hints[field.name]

        if field_value is None:
            if not _is_optional(field_type):
                description = (
                    f"value of non-optional type {field.name} detected when "
                    f"decoding {cls.__name__}"
                )
                if infer_missing:
                    warnings.warn(
                        f"Missing {description} and was defaulted to None by "
                        "infer_missing=True. Set infer_missing=False (the "
                        "default) to prevent this behavior.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        f"'NoneType' object {description}.",
                        RuntimeWarning,
                    )

            init_kwargs[field.name] = field_value
            continue

        normalized_type = field_type
        while _is_new_type(normalized_type):
            normalized_type = normalized_type.__supertype__

        override = overrides.get(field.name)
        if override is not None and override.decoder is not None:
            if normalized_type is type(field_value):
                init_kwargs[field.name] = field_value
            else:
                init_kwargs[field.name] = override.decoder(field_value)
        else:
            init_kwargs[field.name] = _decode_type(
                normalized_type, field_value, infer_missing
            )

    return cls(**init_kwargs)


def _asdict(obj, encode_json=False):
    if is_dataclass(obj):
        overrides = _user_overrides_or_exts(obj.__class__)
        values = {
            field.name: _asdict(getattr(obj, field.name), encode_json=encode_json)
            for field in fields(obj)
        }
        return _encode_overrides(values, overrides, encode_json=encode_json)

    if _isinstance_safe(obj, Mapping):
        return {
            _asdict(key, encode_json=encode_json): _asdict(
                value, encode_json=encode_json
            )
            for key, value in obj.items()
        }

    if (
        _isinstance_safe(obj, Collection)
        and not isinstance(obj, (str, bytes, bytearray))
    ):
        values = (_asdict(value, encode_json=encode_json) for value in obj)

        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*values)

        try:
            return type(obj)(values)
        except TypeError:
            return list(values)

    return copy.deepcopy(obj)


def to_dict(obj, encode_json=False):
    return _asdict(obj, encode_json=encode_json)


def from_dict(cls: Type[TypeVar("T")], kvs, infer_missing=False):
    return _decode_dataclass(cls, kvs, infer_missing)


def to_json(obj, encode_json=False, **kwargs):
    return json.dumps(
        _asdict(obj, encode_json=encode_json),
        cls=_ExtendedEncoder,
        **kwargs,
    )


def from_json(
    cls,
    s,
    *,
    parse_float=None,
    parse_int=None,
    parse_constant=None,
    infer_missing=False,
    **kwargs,
):
    kvs = json.loads(
        s,
        parse_float=parse_float,
        parse_int=parse_int,
        parse_constant=parse_constant,
        **kwargs,
    )
    return _decode_dataclass(cls, kvs, infer_missing)