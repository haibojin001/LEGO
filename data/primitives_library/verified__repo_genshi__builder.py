# -*- coding: utf-8 -*-
"""Convenience objects for constructing Genshi markup streams."""

from genshi.compat import numeric_types, string_types, text_type
from genshi.core import Attrs, Markup, Namespace, QName, Stream, START, END, TEXT

__all__ = ['Fragment', 'Element', 'ElementFactory', 'tag']
__docformat__ = 'restructuredtext en'


class Fragment(object):
    """A collection of markup nodes without an enclosing element."""

    __slots__ = ['children']

    def __init__(self):
        self.children = []

    def __add__(self, other):
        result = Fragment()
        result(self, other)
        return result

    def __call__(self, *args):
        for item in args:
            self.append(item)
        return self

    def __iter__(self):
        return self._generate()

    def __repr__(self):
        return '<%s>' % type(self).__name__

    def __str__(self):
        return str(self.generate())

    def __unicode__(self):
        return text_type(self.generate())

    def __html__(self):
        return Markup(self.generate())

    def append(self, node):
        atomic = (Stream, Element) + string_types + numeric_types

        if isinstance(node, atomic):
            self.children.append(node)
            return

        if isinstance(node, Fragment):
            self.children.extend(node.children)
            return

        if node is None:
            return

        try:
            for item in node:
                self.append(item)
        except TypeError:
            self.children.append(node)

    def _generate(self):
        for node in self.children:
            if isinstance(node, Fragment):
                for event in node._generate():
                    yield event
            elif isinstance(node, Stream):
                for event in node:
                    yield event
            else:
                if not isinstance(node, string_types):
                    node = text_type(node)
                yield TEXT, node, (None, -1, -1)

    def generate(self):
        return Stream(self._generate())


def _kwargs_to_attrs(kwargs):
    result = []
    seen = set()

    for key, value in kwargs.items():
        key = key.rstrip('_').replace('_', '-')
        if value is not None and key not in seen:
            result.append((QName(key), text_type(value)))
            seen.add(key)

    return Attrs(result)


class Element(Fragment):
    """A markup element assembled using the builder pattern."""

    __slots__ = ['tag', 'attrib']

    def __init__(self, tag_, **attrib):
        Fragment.__init__(self)
        self.tag = QName(tag_)
        self.attrib = _kwargs_to_attrs(attrib)

    def __call__(self, *args, **kwargs):
        self.attrib |= _kwargs_to_attrs(kwargs)
        Fragment.__call__(self, *args)
        return self

    def __repr__(self):
        return '<%s "%s">' % (type(self).__name__, self.tag)

    def _generate(self):
        yield START, (self.tag, self.attrib), (None, -1, -1)
        for event in Fragment._generate(self):
            yield event
        yield END, self.tag, (None, -1, -1)


class ElementFactory(object):
    """Factory that creates elements through attribute access."""

    def __init__(self, namespace=None):
        self.namespace = namespace

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        if self.namespace:
            return Element(self.namespace[name])
        return Element(name)

    def __call__(self, *args):
        return Fragment()(*args)


tag = ElementFactory()