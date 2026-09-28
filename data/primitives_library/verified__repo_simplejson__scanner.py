"""JSON token scanner."""

import re

from .errors import JSONDecodeError


def _import_c_make_scanner():
    try:
        from ._speedups import make_scanner
    except ImportError:
        return None
    return make_scanner


c_make_scanner = _import_c_make_scanner()

__all__ = ["make_scanner", "JSONDecodeError"]

NUMBER_RE = re.compile(
    r"(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?",
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)


def py_make_scanner(context):
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    number_match = NUMBER_RE.match
    encoding = context.encoding
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    array_hook = context.array_hook
    memo = context.memo

    def scan_token(text, position):
        message = "Expecting value"

        try:
            character = text[position]
        except IndexError:
            raise JSONDecodeError(message, text, position)

        if character == '"':
            return parse_string(text, position + 1, encoding, strict)

        if character == "{":
            return parse_object(
                (text, position + 1),
                encoding,
                strict,
                scan_token,
                object_hook,
                object_pairs_hook,
                memo,
            )

        if character == "[":
            return parse_array((text, position + 1), scan_token, array_hook)

        if character == "n" and text[position:position + 4] == "null":
            return None, position + 4

        if character == "t" and text[position:position + 4] == "true":
            return True, position + 4

        if character == "f" and text[position:position + 5] == "false":
            return False, position + 5

        number = number_match(text, position)
        if number is not None:
            integer, fraction, exponent = number.groups()
            if fraction or exponent:
                value = parse_float(
                    integer + (fraction or "") + (exponent or "")
                )
            else:
                value = parse_int(integer)
            return value, number.end()

        if parse_constant:
            if character == "N" and text[position:position + 3] == "NaN":
                return parse_constant("NaN"), position + 3
            if character == "I" and text[position:position + 8] == "Infinity":
                return parse_constant("Infinity"), position + 8
            if character == "-" and text[position:position + 9] == "-Infinity":
                return parse_constant("-Infinity"), position + 9

        raise JSONDecodeError(message, text, position)

    def scan_once(string, idx):
        if idx < 0:
            raise JSONDecodeError("Expecting value", string, idx)
        try:
            return scan_token(string, idx)
        finally:
            memo.clear()

    return scan_once


make_scanner = c_make_scanner or py_make_scanner