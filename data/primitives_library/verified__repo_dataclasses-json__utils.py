import inspect
import sys
from collections import Counter
from dataclasses import is_dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Collection,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)


def _get_type_cons(type_):
    if sys.version_info.minor == 6:
        try:
            constructor = type_.__extra__
        except AttributeError:
            try:
                constructor = type_.__origin__
            except AttributeError:
                constructor = type_
            else:
                if constructor is None:
                    constructor = type_
        else:
            if constructor is None:
                try:
                    constructor = type_.__origin__
                except AttributeError:
                    constructor = type_
                else:
                    if constructor is None:
                        constructor = type_
    else:
        constructor = type_.__origin__
    return constructor


_NO_TYPE_ORIGIN = object()


def _get_type_origin(type_):
    try:
        origin = type_.__origin__
    except AttributeError:
        origin = _NO_TYPE_ORIGIN

    if sys.version_info.minor == 6:
        try:
            type_.__extra__
        except AttributeError:
            origin = type_
        else:
            if origin in (None, _NO_TYPE_ORIGIN):
                origin = type_
    elif origin is _NO_TYPE_ORIGIN:
        origin = type_

    return origin


def _hasargs(type_, *args):
    try:
        matched = all(argument in type_.__args__ for argument in args)
    except AttributeError:
        return False
    except TypeError:
        if type_.__args__ is None:
            return False
        raise
    return matched


class _NoArgs(object):
    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


_NO_ARGS = _NoArgs()


def _get_type_args(
    tp: Type, default: Union[Tuple[Type, ...], _NoArgs] = _NO_ARGS
) -> Union[Tuple[Type, ...], _NoArgs]:
    if hasattr(tp, "__args__") and tp.__args__ is not None:
        return tp.__args__
    return default


def _get_type_arg_param(tp: Type, index: int) -> Union[Type, _NoArgs]:
    arguments = _get_type_args(tp)
    if arguments is not _NO_ARGS:
        try:
            return cast(Tuple[Type, ...], arguments)[index]
        except (TypeError, IndexError, NotImplementedError):
            pass
    return _NO_ARGS


def _isinstance_safe(o, t):
    try:
        return isinstance(o, t)
    except Exception:
        return False


def _issubclass_safe(cls, classinfo):
    try:
        return issubclass(cls, classinfo)
    except Exception:
        if _is_new_type(cls):
            return _is_new_type_subclass_safe(cls, classinfo)
        return False


def _is_new_type_subclass_safe(cls, classinfo):
    super_type = getattr(cls, "__supertype__", None)
    if super_type:
        return _is_new_type_subclass_safe(super_type, classinfo)
    try:
        return issubclass(cls, classinfo)
    except Exception:
        return False


def _is_new_type(type_):
    return inspect.isfunction(type_) and hasattr(type_, "__supertype__")


def _is_optional(type_):
    return (
        _issubclass_safe(type_, Optional)
        or _hasargs(type_, type(None))
        or type_ is Any
    )


def _is_counter(type_):
    return _issubclass_safe(_get_type_origin(type_), Counter)


def _is_mapping(type_):
    return _issubclass_safe(_get_type_origin(type_), Mapping)


def _is_collection(type_):
    return _issubclass_safe(_get_type_origin(type_), Collection)


def _is_tuple(type_):
    return _issubclass_safe(_get_type_origin(type_), Tuple)


def _is_nonstr_collection(type_):
    return (
        _issubclass_safe(_get_type_origin(type_), Collection)
        and not _issubclass_safe(type_, str)
    )


def _is_generic_dataclass(type_):
    return is_dataclass(_get_type_origin(type_))


def _timestamp_to_dt_aware(timestamp: float):
    local_timezone = datetime.now(timezone.utc).astimezone().tzinfo
    return datetime.fromtimestamp(timestamp, tz=local_timezone)


def _undefined_parameter_action_safe(cls):
    try:
        configuration = cls.dataclass_json_config
        if configuration is None:
            return
        action = configuration["undefined"]
    except (AttributeError, KeyError):
        return

    if action is None or action.value is None:
        return
    return action


def _handle_undefined_parameters_safe(cls, kvs, usage: str):
    action = _undefined_parameter_action_safe(cls)
    usage = usage.lower()

    if action is None:
        return kvs if usage != "init" else cls.__init__

    if usage == "from":
        return action.value.handle_from_dict(cls=cls, kvs=kvs)
    if usage == "to":
        return action.value.handle_to_dict(obj=cls, kvs=kvs)
    if usage == "dump":
        return action.value.handle_dump(obj=cls)
    if usage == "init":
        return action.value.create_init(obj=cls)

    raise ValueError(
        f"usage must be one of ['to', 'from', 'dump', 'init'], but is '{usage}'"
    )


CatchAllVar = TypeVar("CatchAllVar", bound=Mapping)