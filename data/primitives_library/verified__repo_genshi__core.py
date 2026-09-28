import operator
import sys
import types
from functools import reduce

try:
    from genshi.compat import stringrepr, string_types, text_type
except (ImportError, TypeError):
    compat = types.ModuleType('genshi.compat')
    compat.IS_PYTHON2 = False
    compat.text_type = str
    compat.string_types = (str,)
    compat.integer_types = (int,)
    compat.numeric_types = (float, int)
    compat.stringrepr = repr
    compat.isstring = lambda value: isinstance(value, compat.string_types)

    def add_metaclass(metaclass):
        def wrapper(cls):
            attributes = dict(cls.__dict__)
            attributes.pop('__dict__', None)
            attributes.pop('__weakref__', None)
            return metaclass(cls.__name__, cls.__bases__, attributes)
        return wrapper

    compat.add_metaclass = add_metaclass
    compat.bytes_type = bytes
    compat.basestring = (str, bytes)
    compat.unicode = str
    sys.modules['genshi.compat'] = compat

    stringrepr = compat.stringrepr
    string_types = compat.string_types
    text_type = compat.text_type

from genshi.util import stripentities, striptags

__all__ = [
    'Stream', 'Markup', 'escape', 'unescape', 'Attrs', 'Namespace', 'QName'
]
__docformat__ = 'restructuredtext en'


class StreamEventKind(str):
    """Identifier used for the different kinds of markup stream events."""

    __slots__ = ()
    _instances = {}

    def __new__(cls, value):
        try:
            return cls._instances[value]
        except KeyError:
            instance = str.__new__(cls, value)
            cls._instances[value] = instance
            return instance


class Stream(object):
    """An iterable sequence of markup events."""

    __slots__ = ('events', 'serializer')

    START = StreamEventKind('START')
    END = StreamEventKind('END')
    TEXT = StreamEventKind('TEXT')
    XML_DECL = StreamEventKind('XML_DECL')
    DOCTYPE = StreamEventKind('DOCTYPE')
    START_NS = StreamEventKind('START_NS')
    END_NS = StreamEventKind('END_NS')
    START_CDATA = StreamEventKind('START_CDATA')
    END_CDATA = StreamEventKind('END_CDATA')
    PI = StreamEventKind('PI')
    COMMENT = StreamEventKind('COMMENT')

    def __init__(self, events, serializer=None):
        self.events = events
        self.serializer = serializer

    def __iter__(self):
        return iter(self.events)

    def __str__(self):
        return self.render(encoding=None)

    def __or__(self, function):
        return Stream(_ensure(function(self)), serializer=self.serializer)

    def filter(self, *filters):
        return reduce(operator.or_, (self,) + filters)

    def render(self, method=None, encoding=None, out=None, **kwargs):
        from genshi.output import encode

        if method is None:
            method = self.serializer or 'xml'
        return encode(
            self.serialize(method=method, **kwargs),
            method=method,
            encoding=encoding,
            out=out
        )

    def select(self, path, namespaces=None, variables=None):
        from genshi.path import Path

        return Path(path).select(self, namespaces, variables)

    def serialize(self, method='xml', **kwargs):
        from genshi.output import get_serializer

        serializer = get_serializer(method, **kwargs)
        return serializer(self)


def _ensure(stream):
    """Turn serialized text yielded by pipeline stages back into text events."""
    for event in stream:
        if isinstance(event, string_types):
            event = (Stream.TEXT, event, (None, -1, -1))
        yield event


class Markup(text_type):
    """Unicode text known to contain safe markup."""

    def __new__(cls, text='', *args, **kwargs):
        if isinstance(text, Markup):
            return text
        return text_type.__new__(cls, text, *args, **kwargs)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, stringrepr(self))

    def __html__(self):
        return self

    def __add__(self, other):
        if isinstance(other, string_types):
            return Markup(text_type.__add__(self, escape(other)))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, string_types):
            return Markup(text_type.__add__(escape(other), self))
        return NotImplemented

    def __mod__(self, args):
        if isinstance(args, dict):
            args = dict((key, escape(value)) for key, value in args.items())
        elif isinstance(args, tuple):
            args = tuple(escape(value) for value in args)
        else:
            args = escape(args)
        return Markup(text_type.__mod__(self, args))

    def join(self, sequence, escape_quotes=True):
        return Markup(text_type.join(
            self, (escape(item, quotes=escape_quotes) for item in sequence)
        ))

    def stripentities(self, keepxmlentities=False):
        return Markup(stripentities(self, keepxmlentities=keepxmlentities))

    def striptags(self):
        return Markup(striptags(self))


def escape(text, quotes=True):
    """Convert text into markup-safe text by escaping XML special characters."""
    if not text:
        return Markup()
    if isinstance(text, Markup):
        return text
    text = text_type(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    if quotes:
        text = text.replace('"', '&#34;')
        text = text.replace("'", '&#39;')
    return Markup(text)


def unescape(text):
    """Replace character and entity references in text with their characters."""
    return text_type(stripentities(text))


class Attrs(tuple):
    """An immutable collection of markup attributes."""

    __slots__ = ()

    def __contains__(self, name):
        for attr_name, value in self:
            if attr_name == name:
                return True
        return False

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, list.__repr__(list(self)))

    def get(self, name, default=None):
        for attr_name, value in self:
            if attr_name == name:
                return value
        return default

    def __getitem__(self, index):
        if isinstance(index, string_types):
            return self.get(index)
        return tuple.__getitem__(self, index)

    def __or__(self, attrs):
        if not isinstance(attrs, Attrs):
            attrs = Attrs(attrs)
        return Attrs(
            [attribute for attribute in self if attribute[0] not in attrs] +
            list(attrs)
        )

    def __add__(self, attrs):
        return Attrs(tuple.__add__(self, tuple(attrs)))

    def __radd__(self, attrs):
        return Attrs(tuple(attrs) + tuple(self))


class Namespace(str):
    """A namespace URI that can conveniently create qualified names."""

    def __new__(cls, uri):
        return str.__new__(cls, uri)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, stringrepr(self))

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return QName('{%s}%s' % (self, name))

    def __getitem__(self, name):
        return QName('{%s}%s' % (self, name))

    def __contains__(self, qname):
        try:
            return qname.startswith('{%s}' % self)
        except AttributeError:
            return False


class QName(str):
    """A qualified XML name in Clark notation."""

    def __new__(cls, qname):
        return str.__new__(cls, qname)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, stringrepr(self))

    @property
    def namespace(self):
        if self.startswith('{'):
            end = self.find('}')
            if end >= 0:
                return Namespace(self[1:end])
        return None

    @property
    def localname(self):
        if self.startswith('{'):
            end = self.find('}')
            if end >= 0:
                return self[end + 1:]
        return self


START = Stream.START
END = Stream.END
TEXT = Stream.TEXT
XML_DECL = Stream.XML_DECL
DOCTYPE = Stream.DOCTYPE
START_NS = Stream.START_NS
END_NS = Stream.END_NS
START_CDATA = Stream.START_CDATA
END_CDATA = Stream.END_CDATA
PI = Stream.PI
COMMENT = Stream.COMMENT

XML_NAMESPACE = Namespace('http://www.w3.org/XML/1998/namespace')
XML_LANG = XML_NAMESPACE['lang']
XML_SPACE = XML_NAMESPACE['space']
XML_BASE = XML_NAMESPACE['base']