from __future__ import absolute_import

import re
import decimal
import sys
from operator import itemgetter

from .compat import binary_type, text_type, string_types, integer_types, PY3
from .decoder import PosInf
from .raw_json import RawJSON


_HAS_ADD_NOTE = sys.version_info >= (3, 11)


def _import_speedups():
    try:
        from . import _speedups
        return (
            _speedups.encode_basestring_ascii,
            _speedups.encode_basestring,
            _speedups.make_encoder,
        )
    except ImportError:
        return None, None, None


c_encode_basestring_ascii, c_encode_basestring, c_make_encoder = _import_speedups()

ESCAPE = re.compile(r'[\x00-\x1f\\"]')
ESCAPE_ASCII = re.compile(r'([\\"]|[^\ -~])')
HAS_UTF8 = re.compile(r'[\x80-\xff]')

ESCAPE_DCT = {
    '\\': '\\\\',
    '"': '\\"',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
}
for i in range(0x20):
    ESCAPE_DCT.setdefault(chr(i), '\\u%04x' % i)
del i

FLOAT_REPR = repr

_NAN = float('nan')
_INFINITY = float('inf')


def _encode_decimal(value, floatstr):
    value = str(value)
    first = value[:1]
    significant = value[1:2] if first == '-' else first
    if '0' <= significant <= '9':
        return value
    if significant == 'I':
        return floatstr(-_INFINITY if first == '-' else _INFINITY)
    return floatstr(_NAN)


if sys.version_info >= (3, 15):
    _dict_types = (dict, frozendict)
else:
    _dict_types = dict


def py_encode_basestring(s, _PY3=PY3, _q=u'"'):
    if _PY3:
        if isinstance(s, bytes):
            s = str(s, 'utf-8')
        elif type(s) is not str:
            s = str.__str__(s)
    else:
        if isinstance(s, str) and HAS_UTF8.search(s) is not None:
            s = unicode(s, 'utf-8')
        elif type(s) not in (str, unicode):
            if isinstance(s, str):
                s = str.__str__(s)
            else:
                s = unicode.__getnewargs__(s)[0]

    def replace(match):
        return ESCAPE_DCT[match.group(0)]

    return _q + ESCAPE.sub(replace, s) + _q


def py_encode_basestring_ascii(s, _PY3=PY3):
    if _PY3:
        if isinstance(s, bytes):
            s = str(s, 'utf-8')
        elif type(s) is not str:
            s = str.__str__(s)
    else:
        if isinstance(s, str) and HAS_UTF8.search(s) is not None:
            s = unicode(s, 'utf-8')
        elif type(s) not in (str, unicode):
            if isinstance(s, str):
                s = str.__str__(s)
            else:
                s = unicode.__getnewargs__(s)[0]

    def replace(match):
        char = match.group(0)
        try:
            return ESCAPE_DCT[char]
        except KeyError:
            number = ord(char)
            if number < 0x10000:
                return '\\u%04x' % number
            number -= 0x10000
            high = 0xd800 | ((number >> 10) & 0x3ff)
            low = 0xdc00 | (number & 0x3ff)
            return '\\u%04x\\u%04x' % (high, low)

    return '"' + str(ESCAPE_ASCII.sub(replace, s)) + '"'


encode_basestring_ascii = c_encode_basestring_ascii or py_encode_basestring_ascii
encode_basestring = c_encode_basestring or py_encode_basestring


class JSONEncoder(object):
    item_separator = ', '
    key_separator = ': '

    def __init__(
        self,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=False,
        sort_keys=False,
        indent=None,
        separators=None,
        encoding='utf-8',
        default=None,
        use_decimal=True,
        namedtuple_as_object=True,
        tuple_as_array=True,
        bigint_as_string=False,
        item_sort_key=None,
        for_json=False,
        ignore_nan=False,
        int_as_string_bitcount=None,
        iterable_as_array=False,
    ):
        self.skipkeys = skipkeys
        self.ensure_ascii = ensure_ascii
        self.check_circular = check_circular
        self.allow_nan = allow_nan
        self.sort_keys = sort_keys

        if indent is not None and not isinstance(indent, string_types):
            indent = ' ' * indent
        self.indent = indent

        if separators is not None:
            self.item_separator, self.key_separator = separators
        elif indent is not None:
            self.item_separator = ','

        self.encoding = encoding
        if default is not None:
            self.default = default
        self.use_decimal = use_decimal
        self.namedtuple_as_object = namedtuple_as_object
        self.tuple_as_array = tuple_as_array
        self.bigint_as_string = bigint_as_string
        self.item_sort_key = item_sort_key
        self.for_json = for_json
        self.ignore_nan = ignore_nan
        self.int_as_string_bitcount = int_as_string_bitcount
        self.iterable_as_array = iterable_as_array
        self.encode_basestring = (
            encode_basestring_ascii if ensure_ascii else encode_basestring
        )

    def default(self, o):
        raise TypeError(
            'Object of type %s is not JSON serializable' %
            o.__class__.__name__
        )

    def encode(self, o):
        if isinstance(o, string_types) or isinstance(o, binary_type):
            return self.encode_basestring(o)
        return ''.join(self.iterencode(o, _one_shot=True))

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None

        encoder = self.encode_basestring
        indent = self.indent
        item_separator = self.item_separator
        key_separator = self.key_separator
        sort_keys = self.sort_keys
        skipkeys = self.skipkeys
        use_decimal = self.use_decimal
        namedtuple_as_object = self.namedtuple_as_object
        tuple_as_array = self.tuple_as_array
        bigint_as_string = self.bigint_as_string
        item_sort_key = self.item_sort_key
        for_json = self.for_json
        ignore_nan = self.ignore_nan
        int_as_string_bitcount = self.int_as_string_bitcount
        iterable_as_array = self.iterable_as_array
        allow_nan = self.allow_nan
        default = self.default

        def floatstr(value):
            if value != value:
                text = 'NaN'
            elif value == _INFINITY:
                text = 'Infinity'
            elif value == -_INFINITY:
                text = '-Infinity'
            else:
                return FLOAT_REPR(value)

            if ignore_nan:
                return 'null'
            if not allow_nan:
                raise ValueError(
                    'Out of range float values are not JSON compliant'
                )
            return text

        def intstr(value):
            if int_as_string_bitcount is not None:
                limit = 1 << int_as_string_bitcount
                if value >= limit or value <= -limit:
                    return encoder(str(value))
            elif bigint_as_string:
                limit = 1 << 53
                if value >= limit or value <= -limit:
                    return encoder(str(value))
            return str(value)

        def mark(value):
            if markers is None:
                return
            markerid = id(value)
            if markerid in markers:
                raise ValueError('Circular reference detected')
            markers[markerid] = value

        def unmark(value):
            if markers is not None:
                del markers[id(value)]

        def encode_key(key):
            if isinstance(key, string_types) or isinstance(key, binary_type):
                return encoder(key)
            if key is True:
                return encoder('true')
            if key is False:
                return encoder('false')
            if key is None:
                return encoder('null')
            if isinstance(key, integer_types):
                return encoder(intstr(key))
            if isinstance(key, float):
                return encoder(floatstr(key))
            if use_decimal and isinstance(key, decimal.Decimal):
                return encoder(_encode_decimal(key, floatstr))
            if skipkeys:
                return None
            raise TypeError(
                'keys must be str, int, float, bool or None, not %s' %
                key.__class__.__name__
            )

        def iterencode_list(values, current_indent_level):
            if not values:
                yield '[]'
                return

            mark(values)
            current_indent_level += 1
            if indent is not None:
                newline_indent = '\n' + (indent * current_indent_level)
                separator = item_separator + newline_indent
                yield '[' + newline_indent
            else:
                newline_indent = None
                separator = item_separator
                yield '['

            first = True
            for value in values:
                if first:
                    first = False
                else:
                    yield separator
                for chunk in iterencode_value(value, current_indent_level):
                    yield chunk

            if newline_indent is not None:
                yield '\n' + (indent * (current_indent_level - 1))
            yield ']'
            unmark(values)

        def iterencode_dict(mapping, current_indent_level):
            if not mapping:
                yield '{}'
                return

            mark(mapping)
            current_indent_level += 1

            if item_sort_key is not None:
                items = sorted(mapping.items(), key=item_sort_key)
            elif sort_keys:
                items = sorted(mapping.items(), key=itemgetter(0))
            else:
                items = mapping.items()

            if indent is not None:
                newline_indent = '\n' + (indent * current_indent_level)
                separator = item_separator + newline_indent
                yield '{' + newline_indent
            else:
                newline_indent = None
                separator = item_separator
                yield '{'

            first = True
            for key, value in items:
                encoded_key = encode_key(key)
                if encoded_key is None:
                    continue
                if first:
                    first = False
                else:
                    yield separator
                yield encoded_key
                yield key_separator
                for chunk in iterencode_value(value, current_indent_level):
                    yield chunk

            if newline_indent is not None:
                yield '\n' + (indent * (current_indent_level - 1))
            yield '}'
            unmark(mapping)

        def iterencode_default(value, current_indent_level):
            mark(value)
            try:
                replacement = default(value)
                for chunk in iterencode_value(replacement, current_indent_level):
                    yield chunk
            except Exception as exc:
                if _HAS_ADD_NOTE and isinstance(exc, TypeError):
                    try:
                        exc.add_note(
                            'when serializing %s object' %
                            value.__class__.__name__
                        )
                    except Exception:
                        pass
                raise
            finally:
                unmark(value)

        def iterencode_value(value, current_indent_level):
            if isinstance(value, string_types) or isinstance(value, binary_type):
                yield encoder(value)
            elif value is None:
                yield 'null'
            elif value is True:
                yield 'true'
            elif value is False:
                yield 'false'
            elif isinstance(value, integer_types):
                yield intstr(value)
            elif isinstance(value, float):
                yield floatstr(value)
            elif use_decimal and isinstance(value, decimal.Decimal):
                yield _encode_decimal(value, floatstr)
            elif isinstance(value, RawJSON):
                yield value.encoded_json
            elif isinstance(value, _dict_types):
                for chunk in iterencode_dict(value, current_indent_level):
                    yield chunk
            elif isinstance(value, (list, tuple)):
                if (
                    namedtuple_as_object and
                    isinstance(value, tuple) and
                    hasattr(value, '_asdict') and
                    callable(value._asdict)
                ):
                    mark(value)
                    try:
                        for chunk in iterencode_dict(
                            value._asdict(), current_indent_level
                        ):
                            yield chunk
                    finally:
                        unmark(value)
                elif isinstance(value, list) or tuple_as_array:
                    for chunk in iterencode_list(value, current_indent_level):
                        yield chunk
                else:
                    for chunk in iterencode_default(value, current_indent_level):
                        yield chunk
            elif for_json and hasattr(value, 'for_json') and callable(value.for_json):
                mark(value)
                try:
                    for chunk in iterencode_value(
                        value.for_json(), current_indent_level
                    ):
                        yield chunk
                finally:
                    unmark(value)
            elif iterable_as_array:
                try:
                    iterator = iter(value)
                except TypeError:
                    for chunk in iterencode_default(value, current_indent_level):
                        yield chunk
                else:
                    mark(value)
                    try:
                        for chunk in iterencode_list(
                            list(iterator), current_indent_level
                        ):
                            yield chunk
                    finally:
                        unmark(value)
            else:
                for chunk in iterencode_default(value, current_indent_level):
                    yield chunk

        return iterencode_value(o, 0)


class JSONEncoderForHTML(JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(JSONEncoderForHTML, self).__init__(*args, **kwargs)
        base_encoder = self.encode_basestring

        def encode_for_html(value):
            return (
                base_encoder(value)
                .replace('&', '\\u0026')
                .replace('<', '\\u003c')
                .replace('>', '\\u003e')
            )

        self.encode_basestring = encode_for_html