import re
import sys

from genshi.builder import Element
from genshi.compat import string_types, text_type
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path

__all__ = ['Transformer', 'StreamBuffer', 'InjectorTransformation', 'ENTER',
           'EXIT', 'INSIDE', 'OUTSIDE', 'BREAK']


class TransformMark(str):
    __slots__ = ()
    _instances = {}

    def __new__(cls, value):
        if value in cls._instances:
            return cls._instances[value]
        instance = str.__new__(cls, value)
        cls._instances[value] = instance
        return instance


ENTER = TransformMark('ENTER')
INSIDE = TransformMark('INSIDE')
OUTSIDE = TransformMark('OUTSIDE')
ATTR = TransformMark('ATTR')
EXIT = TransformMark('EXIT')
BREAK = TransformMark('BREAK')


class PushBackStream(object):
    def __init__(self, stream):
        self.stream = iter(stream)
        self.peek = None

    def push(self, event):
        assert self.peek is None
        self.peek = event

    def __iter__(self):
        while True:
            if self.peek is not None:
                event = self.peek
                self.peek = None
                yield event
                continue
            try:
                yield next(self.stream)
            except StopIteration:
                return


def _path_test(path):
    test = path.test()

    def matches(event):
        try:
            return test(event, {}, {})
        except TypeError:
            try:
                return test(event)
            except TypeError:
                return test(event, None, None)
    return matches


def _marked(mark):
    return mark in (ENTER, INSIDE, OUTSIDE, ATTR, EXIT)


def _attrs(value):
    if isinstance(value, Attrs):
        return value
    return Attrs(value)


def _as_stream(content):
    if content is None:
        return ()
    if isinstance(content, Stream):
        return content
    if isinstance(content, Element):
        return content.generate()
    if isinstance(content, string_types):
        return ((TEXT, _ensure(content), (None, -1, -1)),)
    if isinstance(content, Markup):
        return ((TEXT, content, (None, -1, -1)),)
    if hasattr(content, 'generate'):
        return content.generate()
    try:
        return iter(content)
    except TypeError:
        return ((TEXT, _ensure(content), (None, -1, -1)),)


def _inject(content, pos=None):
    for event in _as_stream(content):
        if len(event) == 3:
            yield event
        else:
            yield event


class StreamBuffer(object):
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(event)

    def extend(self, events):
        self.events.extend(events)

    def reset(self):
        del self.events[:]

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __bool__(self):
        return bool(self.events)

    __nonzero__ = __bool__

    def __call__(self):
        return Stream(iter(self.events))


class Transformer(object):
    __slots__ = ['transforms']

    def __init__(self, path='.'):
        self.transforms = [SelectTransformation(path)]

    def _mark(self, stream):
        for event in stream:
            if (isinstance(event, tuple) and len(event) == 2 and
                    event[0] in (ENTER, INSIDE, OUTSIDE, ATTR, EXIT, BREAK,
                                 None)):
                yield event
            else:
                yield None, event

    def _unmark(self, stream):
        for mark, event in stream:
            if mark is not BREAK:
                yield event

    def __call__(self, stream, keep_marks=False):
        events = self._mark(stream)
        for transform in self.transforms:
            events = transform(events)
        if not keep_marks:
            events = self._unmark(events)
        return Stream(events, serializer=getattr(stream, 'serializer', None))

    def apply(self, function):
        transformer = Transformer()
        transformer.transforms = self.transforms[:]
        if isinstance(function, Transformer):
            transformer.transforms.extend(function.transforms)
        else:
            transformer.transforms.append(function)
        return transformer

    def select(self, path):
        return self.apply(SelectTransformation(path))

    def invert(self):
        return self.apply(InvertTransformation())

    def end(self):
        return self.apply(EndTransformation())

    def empty(self):
        return self.apply(EmptyTransformation())

    def remove(self):
        return self.apply(RemoveTransformation())

    def unwrap(self):
        return self.apply(UnwrapTransformation())

    def wrap(self, element):
        return self.apply(WrapTransformation(element))

    def replace(self, content):
        return self.apply(ReplaceTransformation(content))

    def before(self, content):
        return self.apply(BeforeTransformation(content))

    def after(self, content):
        return self.apply(AfterTransformation(content))

    def prepend(self, content):
        return self.apply(PrependTransformation(content))

    def append(self, content):
        return self.apply(AppendTransformation(content))

    def attr(self, name, value=None):
        return self.apply(AttrTransformation(name, value))

    def map(self, function, kind=None):
        return self.apply(MapTransformation(function, kind))

    def substitute(self, pattern, replace, count=0):
        return self.apply(SubstituteTransformation(pattern, replace, count))

    def rename(self, name):
        return self.apply(RenameTransformation(name))

    def copy(self, buffer):
        return self.apply(CopyTransformation(buffer))

    def cut(self, buffer, accumulate=False):
        return self.apply(CutTransformation(buffer, accumulate))

    def filter(self, filter, **kwargs):
        return self.apply(FilterTransformation(filter, **kwargs))

    def trace(self, prefix=None, file=None):
        return self.apply(TraceTransformation(prefix, file))


class SelectTransformation(object):
    def __init__(self, path):
        self.path = path if isinstance(path, Path) else Path(path)

    def __call__(self, stream):
        test = _path_test(self.path)
        depths = []

        for oldmark, event in stream:
            if oldmark is BREAK:
                yield oldmark, event
                continue

            kind, data, pos = event
            result = test(event)

            eligible = oldmark is None or _marked(oldmark)
            selected = bool(result) and eligible

            if kind is START:
                for index in range(len(depths)):
                    depths[index] += 1

                if selected:
                    depths.append(1)
                    if isinstance(result, Attrs):
                        yield ATTR, (kind, (data[0], result), pos)
                    else:
                        yield ENTER, event
                elif depths:
                    yield INSIDE, event
                else:
                    yield None, event
                continue

            if kind is END:
                closing = bool(depths and depths[-1] == 1)
                if depths:
                    for index in range(len(depths)):
                        depths[index] -= 1
                    while depths and depths[-1] == 0:
                        depths.pop()

                if closing:
                    yield EXIT, event
                elif depths:
                    yield INSIDE, event
                elif selected:
                    yield OUTSIDE, event
                else:
                    yield None, event
                continue

            if selected:
                if isinstance(result, Attrs) and kind is START:
                    yield ATTR, (kind, (data[0], result), pos)
                else:
                    yield OUTSIDE, event
            elif depths:
                yield INSIDE, event
            else:
                yield None, event


class InvertTransformation(object):
    def __call__(self, stream):
        for mark, event in stream:
            if mark is BREAK:
                yield mark, event
            elif mark is None:
                yield OUTSIDE, event
            else:
                yield None, event


class EndTransformation(object):
    def __call__(self, stream):
        for mark, event in stream:
            if mark is BREAK:
                yield mark, event
            else:
                yield None, event


class EmptyTransformation(object):
    def __call__(self, stream):
        for mark, event in stream:
            if mark is not INSIDE:
                yield mark, event


class RemoveTransformation(object):
    def __call__(self, stream):
        for mark, event in stream:
            if mark is None or mark is BREAK:
                yield mark, event


class UnwrapTransformation(object):
    def __call__(self, stream):
        for mark, event in stream:
            if mark not in (ENTER, EXIT):
                yield mark, event


class InjectorTransformation(object):
    def __init__(self, content):
        self.content = content

    def _content(self, pos=None):
        return _inject(self.content, pos)


class WrapTransformation(InjectorTransformation):
    def __init__(self, element):
        self.element = element

    def _wrapper(self):
        element = self.element
        if not isinstance(element, Element):
            try:
                element = element()
            except TypeError:
                element = Element(element)
        return list(_as_stream(element))

    def __call__(self, stream):
        wrapper = self._wrapper()
        starts = [event for event in wrapper if event[0] is START]
        ends = [event for event in wrapper if event[0] is END]
        start = starts[0] if starts else None
        end = ends[-1] if ends else None

        pending = False
        for mark, event in stream:
            if mark is ENTER:
                if start is not None:
                    yield None, start
                yield mark, event
                pending = True
            elif mark is EXIT:
                yield mark, event
                if end is not None:
                    yield None, end
                pending = False
            elif mark is OUTSIDE:
                if start is not None:
                    yield None, start
                yield mark, event
                if end is not None:
                    yield None, end
            else:
                yield mark, event


class ReplaceTransformation(InjectorTransformation):
    def __call__(self, stream):
        skipping = 0
        for mark, event in stream:
            if mark is ENTER:
                if not skipping:
                    for injected in self._content(event[2]):
                        yield None, injected
                skipping += 1
            elif mark is EXIT:
                if skipping:
                    skipping -= 1
            elif mark is OUTSIDE:
                for injected in self._content(event[2]):
                    yield None, injected
            elif not skipping:
                yield mark, event


class BeforeTransformation(InjectorTransformation):
    def __call__(self, stream):
        for mark, event in stream:
            if mark in (ENTER, OUTSIDE):
                for injected in self._content(event[2]):
                    yield None, injected
            yield mark, event


class AfterTransformation(InjectorTransformation):
    def __call__(self, stream):
        for mark, event in stream:
            yield mark, event
            if mark in (EXIT, OUTSIDE):
                for injected in self._content(event[2]):
                    yield None, injected


class PrependTransformation(InjectorTransformation):
    def __call__(self, stream):
        for mark, event in stream:
            yield mark, event
            if mark is ENTER:
                for injected in self._content(event[2]):
                    yield None, injected


class AppendTransformation(InjectorTransformation):
    def __call__(self, stream):
        for mark, event in stream:
            if mark is EXIT:
                for injected in self._content(event[2]):
                    yield None, injected
            yield mark, event


class AttrTransformation(object):
    def __init__(self, name, value=None):
        self.name = QName(name)
        self.value = value

    def __call__(self, stream):
        for mark, event in stream:
            kind, data, pos = event
            if mark in (ENTER, ATTR) and kind is START:
                tag, attrs = data
                attrs = list(attrs)
                oldvalue = None
                found = False
                for index, (name, value) in enumerate(attrs):
                    if name == self.name:
                        oldvalue = value
                        found = True
                        break

                value = self.value
                if callable(value):
                    try:
                        value = value(name=self.name, value=oldvalue)
                    except TypeError:
                        try:
                            value = value(oldvalue)
                        except TypeError:
                            value = value()

                if value is None:
                    attrs = [(name, val) for name, val in attrs
                             if name != self.name]
                elif found:
                    attrs[index] = (self.name, value)
                else:
                    attrs.append((self.name, value))

                yield mark, (kind, (tag, Attrs(attrs)), pos)
            else:
                yield mark, event


class MapTransformation(object):
    def __init__(self, function, kind=None):
        self.function = function
        self.kind = kind

    def __call__(self, stream):
        for mark, event in stream:
            kind, data, pos = event
            if mark is not None and mark is not BREAK and (
                    self.kind is None or kind is self.kind or kind == self.kind):
                data = self.function(data)
                event = (kind, data, pos)
            yield mark, event


class SubstituteTransformation(object):
    def __init__(self, pattern, replace, count=0):
        self.pattern = re.compile(pattern) if isinstance(pattern, string_types) else pattern
        self.replace = replace
        self.count = count

    def __call__(self, stream):
        for mark, event in stream:
            kind, data, pos = event
            if mark is not None and mark is not BREAK and kind is TEXT:
                data = self.pattern.sub(self.replace, data, self.count)
                event = (kind, data, pos)
            yield mark, event


class RenameTransformation(object):
    def __init__(self, name):
        self.name = QName(name)

    def __call__(self, stream):
        for mark, event in stream:
            kind, data, pos = event
            if mark is not None and mark is not BREAK:
                if kind is START:
                    event = (kind, (self.name, data[1]), pos)
                elif kind is END:
                    event = (kind, self.name, pos)
            yield mark, event


class CopyTransformation(object):
    def __init__(self, buffer):
        self.buffer = buffer

    def __call__(self, stream):
        for mark, event in stream:
            if mark is not None and mark is not BREAK:
                self.buffer.append(event)
            yield mark, event


class CutTransformation(object):
    def __init__(self, buffer, accumulate=False):
        self.buffer = buffer
        self.accumulate = accumulate

    def __call__(self, stream):
        if not self.accumulate:
            self.buffer.reset()

        cutting = False
        emitted_break = False

        for mark, event in stream:
            if mark is BREAK:
                yield mark, event
                continue

            if mark is ENTER:
                if not cutting:
                    if emitted_break:
                        yield BREAK, event
                    emitted_break = True
                cutting = True
                self.buffer.append(event)
            elif mark is INSIDE:
                self.buffer.append(event)
            elif mark is EXIT:
                self.buffer.append(event)
                cutting = False
            elif mark is OUTSIDE:
                if emitted_break:
                    yield BREAK, event
                emitted_break = True
                self.buffer.append(event)
            else:
                yield mark, event


class FilterTransformation(object):
    def __init__(self, filter, **kwargs):
        self.filter = filter
        self.kwargs = kwargs

    def __call__(self, stream):
        selected = []
        depth = 0

        def flush():
            if not selected:
                return ()
            source = Stream((event for mark, event in selected))
            try:
                result = self.filter(source, **self.kwargs)
            except TypeError:
                result = self.filter(source)
            return ((INSIDE, event) for event in result)

        for mark, event in stream:
            if mark is ENTER:
                if depth == 0:
                    selected.append((mark, event))
                else:
                    selected.append((mark, event))
                depth += 1
            elif depth:
                selected.append((mark, event))
                if mark is EXIT:
                    depth -= 1
                    if depth == 0:
                        for output in flush():
                            yield output
                        selected[:] = []
            elif mark is OUTSIDE:
                source = Stream((event,))
                try:
                    result = self.filter(source, **self.kwargs)
                except TypeError:
                    result = self.filter(source)
                for output in result:
                    yield OUTSIDE, output
            else:
                yield mark, event

        if selected:
            for output in flush():
                yield output


class TraceTransformation(object):
    def __init__(self, prefix=None, file=None):
        self.prefix = prefix
        self.file = file

    def __call__(self, stream):
        output = self.file if self.file is not None else sys.stdout
        for event in stream:
            if self.prefix is not None:
                print('%s%s' % (self.prefix, event), file=output)
            else:
                print(event, file=output)
            yield event