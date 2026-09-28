import json
import math


def _reject_constant(value):
    raise ValueError("malformed json")


def _validate(obj, seen=None):
    obj_type = type(obj)

    if obj is None or obj_type is str or obj_type is bool or obj_type is int:
        return

    if obj_type is float:
        if not math.isfinite(obj):
            raise TypeError("non-finite floats are not supported")
        return

    if obj_type is list:
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            raise TypeError("circular references are not supported")
        seen.add(obj_id)
        try:
            for item in obj:
                _validate(item, seen)
        finally:
            seen.remove(obj_id)
        return

    if obj_type is dict:
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            raise TypeError("circular references are not supported")
        seen.add(obj_id)
        try:
            for key, value in obj.items():
                if type(key) is not str:
                    raise TypeError("dict keys must be strings")
                _validate(value, seen)
        finally:
            seen.remove(obj_id)
        return

    raise TypeError(f"unsupported type: {obj_type.__name__}")


def dumps(obj) -> str:
    _validate(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(s: str):
    if type(s) is not str:
        raise TypeError("loads() expects str")

    try:
        obj = json.loads(s, parse_constant=_reject_constant)
    except ValueError:
        raise

    _validate(obj)
    return obj