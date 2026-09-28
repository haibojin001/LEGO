from __future__ import absolute_import

import re
import sys

from .compat import PY3, unichr
from .scanner import JSONDecodeError, make_scanner


def _import_c_scanstring():
    try:
        from ._speedups import scanstring
    except ImportError:
        return None
    return scanstring


c_scanstring = _import_c_scanstring()

__all__ = ['JSONDecoder']

FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL


def _floatconstants():
    return float('nan'), float('inf'), float('-inf')


NaN, PosInf, NegInf = _floatconstants()

_CONSTANTS = {
    '-Infinity': NegInf,
    'Infinity': PosInf,
    'NaN': NaN,
}

STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)

BACKSLASH = {
    '"': u'"',
    '\\': u'\\',
    '/': u'/',
    'b': u'\b',
    'f': u'\f',
    'n': u'\n',
    'r': u'\r',
    't': u'\t',
}

DEFAULT_ENCODING = 'utf-8'


if hasattr(sys, 'get_int_max_str_digits'):
    bounded_int = int
else:
    def bounded_int(s, INT_MAX_STR_DIGITS=4300):
        if len(s) > INT_MAX_STR_DIGITS:
            raise ValueError(
                'Exceeds the limit (%s) for integer string conversion: '
                'value has %s digits' % (INT_MAX_STR_DIGITS, len(s))
            )
        return int(s)


def scan_four_digit_hex(s, end, _matcher=re.compile(r'^[0-9a-fA-F]{4}$').match):
    escaped = s[end:end + 4]
    if not _matcher(escaped):
        raise JSONDecodeError('Invalid \\uXXXX escape sequence', s, end - 2)
    try:
        return int(escaped, 16), end + 4
    except ValueError:
        raise JSONDecodeError('Invalid \\uXXXX escape sequence', s, end - 2)


def py_scanstring(s, end, encoding=None, strict=True,
                  _backslash=BACKSLASH, _match=STRINGCHUNK.match,
                  _join=u''.join, _py3=PY3, _maxunicode=sys.maxunicode,
                  _scan_hex=scan_four_digit_hex):
    if encoding is None:
        encoding = DEFAULT_ENCODING

    parts = []
    append = parts.append
    opening_quote = end - 1

    while True:
        match = _match(s, end)
        if match is None:
            raise JSONDecodeError(
                'Unterminated string starting at', s, opening_quote
            )

        end = match.end()
        text, delimiter = match.groups()

        if text:
            if not _py3 and not isinstance(text, unicode):
                text = unicode(text, encoding)
            append(text)

        if delimiter == '"':
            break

        if delimiter != '\\':
            if strict:
                raise JSONDecodeError(
                    'Invalid control character %r at', s, end - 1
                )
            append(delimiter)
            continue

        try:
            escape = s[end]
        except IndexError:
            raise JSONDecodeError(
                'Unterminated string starting at', s, opening_quote
            )

        if escape != 'u':
            try:
                character = _backslash[escape]
            except KeyError:
                raise JSONDecodeError(
                    'Invalid \\X escape sequence %r', s, end
                )
            end += 1
        else:
            codepoint, end = _scan_hex(s, end + 1)

            if (
                _maxunicode > 65535
                and codepoint & 0xfc00 == 0xd800
                and s[end:end + 2] == '\\u'
            ):
                low, low_end = _scan_hex(s, end + 2)
                if low & 0xfc00 == 0xdc00:
                    codepoint = (
                        0x10000
                        + ((codepoint - 0xd800) << 10)
                        + (low - 0xdc00)
                    )
                    end = low_end

            character = unichr(codepoint)

        append(character)

    return _join(parts), end


scanstring = c_scanstring or py_scanstring

WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)
WHITESPACE_STR = ' \t\n\r'


def JSONObject(state, encoding, strict, scan_once, object_hook,
               object_pairs_hook, memo=None,
               _whitespace=WHITESPACE.match, _ws=WHITESPACE_STR):
    s, end = state

    if memo is None:
        memo = {}

    intern_key = memo.setdefault
    pairs = []

    nextchar = s[end:end + 1]

    if nextchar != '"':
        if nextchar in _ws:
            end = _whitespace(s, end).end()
            nextchar = s[end:end + 1]

        if nextchar == '}':
            if object_pairs_hook is not None:
                return object_pairs_hook(pairs), end + 1

            result = {}
            if object_hook is not None:
                result = object_hook(result)
            return result, end + 1

        if nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes or '}'",
                s,
                end,
            )

    end += 1

    while True:
        key, end = scanstring(s, end, encoding, strict)
        key = intern_key(key, key)

        if s[end:end + 1] != ':':
            end = _whitespace(s, end).end()
            if s[end:end + 1] != ':':
                raise JSONDecodeError("Expecting ':' delimiter", s, end)

        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _whitespace(s, end + 1).end()
        except IndexError:
            pass

        value, end = scan_once(s, end)
        pairs.append((key, value))

        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _whitespace(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ''

        end += 1

        if nextchar == '}':
            break

        if nextchar != ',':
            raise JSONDecodeError(
                "Expecting ',' delimiter or '}'",
                s,
                end - 1,
            )

        comma_position = end - 1

        try:
            nextchar = s[end]
            if nextchar in _ws:
                end += 1
                nextchar = s[end]
                if nextchar in _ws:
                    end = _whitespace(s, end + 1).end()
                    nextchar = s[end]
        except IndexError:
            nextchar = ''

        end += 1

        if nextchar != '"':
            if nextchar == '}':
                raise JSONDecodeError(
                    'Illegal trailing comma before end of object',
                    s,
                    comma_position,
                )
            raise JSONDecodeError(
                'Expecting property name enclosed in double quotes',
                s,
                end - 1,
            )

    if object_pairs_hook is not None:
        return object_pairs_hook(pairs), end

    result = dict(pairs)
    if object_hook is not None:
        result = object_hook(result)

    return result, end


def JSONArray(state, scan_once, array_hook=None,
              _whitespace=WHITESPACE.match, _ws=WHITESPACE_STR):
    s, end = state
    values = []

    nextchar = s[end:end + 1]

    if nextchar in _ws:
        end = _whitespace(s, end + 1).end()
        nextchar = s[end:end + 1]

    if nextchar == ']':
        if array_hook is not None:
            values = array_hook(values)
        return values, end + 1

    if nextchar == '':
        raise JSONDecodeError("Expecting value or ']'", s, end)

    append = values.append

    while True:
        value, end = scan_once(s, end)
        append(value)

        nextchar = s[end:end + 1]
        if nextchar in _ws:
            end = _whitespace(s, end + 1).end()
            nextchar = s[end:end + 1]

        end += 1

        if nextchar == ']':
            break

        if nextchar != ',':
            raise JSONDecodeError(
                "Expecting ',' delimiter or ']'", s, end - 1
            )

        comma_position = end - 1

        try:
            nextchar = s[end]
            if nextchar in _ws:
                end += 1
                nextchar = s[end]
                if nextchar in _ws:
                    end = _whitespace(s, end + 1).end()
                    nextchar = s[end]
        except IndexError:
            nextchar = ''

        end += 1

        if nextchar == ']':
            raise JSONDecodeError(
                'Illegal trailing comma before end of array',
                s,
                comma_position,
            )

    if array_hook is not None:
        values = array_hook(values)

    return values, end


class JSONDecoder(object):
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None, allow_nan=False, array_hook=None):
        self.encoding = encoding
        self.object_hook = object_hook
        self.object_pairs_hook = object_pairs_hook
        self.array_hook = array_hook
        self.parse_float = parse_float or float
        self.parse_int = parse_int or bounded_int
        self.parse_constant = (
            parse_constant
            or (_CONSTANTS.__getitem__ if allow_nan else None)
        )
        self.strict = strict
        self.parse_object = JSONObject
        self.parse_array = JSONArray
        self.parse_string = scanstring
        self.memo = {}
        self.scan_once = make_scanner(self)

    def decode(self, s, _whitespace=WHITESPACE.match):
        obj, end = self.raw_decode(s, idx=_whitespace(s, 0).end())
        end = _whitespace(s, end).end()

        if end != len(s):
            raise JSONDecodeError('Extra data', s, end)

        return obj

    def raw_decode(self, s, idx=0, _whitespace=WHITESPACE.match):
        try:
            return self.scan_once(s, idx)
        except StopIteration as err:
            raise JSONDecodeError('Expecting value', s, err.value)