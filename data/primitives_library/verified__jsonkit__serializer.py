import math


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


def _quote_string(value):
    parts = ['"']

    for ch in value:
        code = ord(ch)

        if ch == '"':
            parts.append('\\"')
        elif ch == "\\":
            parts.append("\\\\")
        elif ch == "\b":
            parts.append("\\b")
        elif ch == "\f":
            parts.append("\\f")
        elif ch == "\n":
            parts.append("\\n")
        elif ch == "\r":
            parts.append("\\r")
        elif ch == "\t":
            parts.append("\\t")
        elif 0x20 <= code <= 0x7E:
            parts.append(ch)
        elif code <= 0xFFFF:
            parts.append("\\u%04x" % code)
        else:
            code -= 0x10000
            high = 0xD800 + (code >> 10)
            low = 0xDC00 + (code & 0x3FF)
            parts.append("\\u%04x\\u%04x" % (high, low))

    parts.append('"')
    return "".join(parts)


def _encode(obj, sort_keys):
    obj_type = type(obj)

    if obj is None:
        return "null"

    if obj_type is str:
        return _quote_string(obj)

    if obj_type is bool:
        return "true" if obj else "false"

    if obj_type is int:
        return str(obj)

    if obj_type is float:
        return repr(obj)

    if obj_type is list:
        return "[" + ",".join(_encode(item, sort_keys) for item in obj) + "]"

    if obj_type is dict:
        keys = sorted(obj) if sort_keys else obj.keys()
        return "{" + ",".join(
            _quote_string(key) + ":" + _encode(obj[key], sort_keys)
            for key in keys
        ) + "}"

    raise TypeError(f"unsupported type: {obj_type.__name__}")


def dumps(obj, sort_keys: bool = False) -> str:
    _validate(obj)
    return _encode(obj, sort_keys)