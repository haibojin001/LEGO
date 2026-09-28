from collections import deque
import os

from genshi.compat import add_metaclass, numeric_types, string_types, text_type, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError

__all__ = [
    'Context', 'DirectiveFactory', 'Template', 'TemplateError',
    'TemplateRuntimeError', 'TemplateSyntaxError', 'BadDirectiveError'
]

__docformat__ = 'restructuredtext en'


class TemplateError(Exception):
    def __init__(self, message, filename=None, lineno=-1, offset=-1):
        if filename is None:
            filename = '<string>'
        self.msg = message
        if filename != '<string>' or lineno >= 0:
            message = '%s (%s, line %d)' % (message, filename, lineno)
        Exception.__init__(self, message)
        self.filename = filename
        self.lineno = lineno
        self.offset = offset


class TemplateSyntaxError(TemplateError):
    def __init__(self, message, filename=None, lineno=-1, offset=-1):
        if isinstance(message, SyntaxError) and message.lineno is not None:
            message = str(message).replace(
                ' (line %d)' % message.lineno, ''
            )
        TemplateError.__init__(self, message, filename, lineno, offset)


class BadDirectiveError(TemplateSyntaxError):
    def __init__(self, name, filename=None, lineno=-1):
        TemplateSyntaxError.__init__(
            self, 'bad directive "%s"' % name, filename, lineno
        )


class TemplateRuntimeError(TemplateError):
    pass


class Context(object):
    def __init__(self, **data):
        self.frames = deque([data])
        self.pop = self.frames.popleft
        self.push = self.frames.appendleft
        self._match_templates = []
        self._choice_stack = []

        def defined(name):
            return name in self

        def value_of(name, default=None):
            return self.get(name, default)

        data.setdefault('defined', defined)
        data.setdefault('value_of', value_of)

    def __repr__(self):
        return repr(list(self.frames))

    def __contains__(self, key):
        return self._find(key)[1] is not None

    has_key = __contains__

    def __delitem__(self, key):
        for frame in self.frames:
            if key in frame:
                del frame[key]

    def __getitem__(self, key):
        value, frame = self._find(key)
        if frame is None:
            raise KeyError(key)
        return value

    def __len__(self):
        return len(self.items())

    def __setitem__(self, key, value):
        self.frames[0][key] = value

    def _find(self, key, default=None):
        for frame in self.frames:
            if key in frame:
                return frame[key], frame
        return default, None

    def get(self, key, default=None):
        for frame in self.frames:
            if key in frame:
                return frame[key]
        return default

    def keys(self):
        keys = []
        for frame in self.frames:
            for key in frame:
                if key not in keys:
                    keys.append(key)
        return keys

    def items(self):
        return [(key, self.get(key)) for key in self.keys()]

    def update(self, mapping):
        self.frames[0].update(mapping)

    def copy(self):
        ctxt = Context()
        ctxt.frames.pop()
        ctxt.frames.extend(self.frames)
        ctxt._match_templates.extend(self._match_templates)
        ctxt._choice_stack.extend(self._choice_stack)
        return ctxt


def _apply_directives(stream, directives, ctxt, vars):
    if directives:
        return directives[0](
            iter(stream), ctxt, directives=directives[1:], **vars
        )
    return stream


try:
    _SUB = StreamEventKind('SUB')
except TypeError:
    _SUB = 'SUB'


class DirectiveFactory(type):
    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)

        directives = getattr(cls, 'directives', ())
        by_name = {}

        for directive in directives:
            names = []

            if isinstance(directive, (tuple, list)) and directive:
                if len(directive) > 1 and isinstance(directive[0], string_types):
                    names.append(directive[0])
                    candidate = directive[-1]
                else:
                    candidate = directive[-1]
            else:
                candidate = directive

            for attr in ('name', 'tagname', 'directive_name'):
                value = getattr(candidate, attr, None)
                if value is not None:
                    names.append(value)

            if not names and isinstance(candidate, string_types):
                names.append(candidate)

            for directive_name in names:
                by_name[directive_name] = candidate

        cls._dir_by_name = by_name


@add_metaclass(DirectiveFactory)
class Template(object):
    directives = []
    filters = []

    def __init__(self, source, basedir=None, filename=None, loader=None,
                 encoding=None, lookup='strict', allow_exec=True):
        self.loader = loader
        self.lookup = lookup
        self.allow_exec = allow_exec

        source_filename = getattr(source, 'name', None)
        if filename is None:
            filename = source_filename or '<string>'
        self.filename = filename

        if basedir is None and filename and filename != '<string>':
            basedir = os.path.dirname(os.path.abspath(filename))
        self.basedir = basedir

        try:
            self.stream = self._prepare(self._parse(source, encoding))
        except TemplateError:
            raise
        except ParseError as exc:
            error_filename = getattr(exc, 'filename', None) or self.filename
            error_lineno = getattr(exc, 'lineno', -1)
            error_offset = getattr(exc, 'offset', -1)
            raise TemplateSyntaxError(
                exc, error_filename, error_lineno, error_offset
            )

    def _parse(self, source, encoding=None):
        raise NotImplementedError

    def _prepare(self, stream):
        return stream

    def _flatten(self, stream, ctxt, **vars):
        for event in stream:
            kind, data, pos = event
            if kind is _SUB or kind == _SUB:
                directives, substream = data
                substream = _apply_directives(
                    substream, directives, ctxt, vars
                )
                for subevent in self._flatten(substream, ctxt, **vars):
                    yield subevent
            else:
                yield event

    def _match(self, stream, ctxt, **vars):
        return stream

    def generate(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(
                'generate() takes at most 1 positional argument (%d given)'
                % len(args)
            )

        if args:
            context = args[0]
            if isinstance(context, Context):
                ctxt = context
            else:
                ctxt = Context(**context)
            if kwargs:
                ctxt.update(kwargs)
        else:
            ctxt = Context(**kwargs)

        stream = self._flatten(self.stream, ctxt, **kwargs)
        stream = self._match(stream, ctxt, **kwargs)

        for filter_ in self.filters:
            stream = filter_(stream, ctxt)

        return Stream(stream)