"""Lightweight sequence subclasses with named field access."""

import sys as _sys
from collections import OrderedDict
from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter


__all__ = ['namedlist', 'namedtuple']


_repr_tmpl = '{name}=%r'

_imm_field_tmpl = """\
    {name} = _property(_itemgetter({index:d}), doc='Alias for field {index:d}')
"""

_m_field_tmpl = """\
    {name} = _property(_itemgetter({index:d}), _itemsetter({index:d}), doc='Alias for field {index:d}')
"""


def _itemsetter(index):
    def set_item(obj, value):
        obj[index] = value
    return set_item


def _validate_names(typename, field_names, rename):
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = [str(name) for name in field_names]

    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            invalid = (
                not all(char.isalnum() or char == '_' for char in name)
                or _iskeyword(name)
                or not name
                or name[0].isdigit()
                or name.startswith('_')
                or name in seen
            )
            if invalid:
                field_names[index] = '_%d' % index
            seen.add(name)

    for name in [typename] + field_names:
        if not name or not all(char.isalnum() or char == '_' for char in name):
            raise ValueError(
                'Type names and field names must be non-empty and can only '
                'contain alphanumeric characters and underscores: %r' % name
            )
        if _iskeyword(name):
            raise ValueError(
                'Type names and field names cannot be a keyword: %r' % name
            )
        if name[0].isdigit():
            raise ValueError(
                'Type names and field names cannot start with a number: %r'
                % name
            )

    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError(
                'Field names cannot start with an underscore: %r' % name
            )
        if name in seen:
            raise ValueError(
                'Encountered duplicate field name: %r' % name
            )
        seen.add(name)

    return field_names


def _create_source(typename, field_names, mutable):
    arg_list = repr(tuple(field_names)).replace("'", "")[1:-1]
    repr_fmt = ', '.join(
        _repr_tmpl.format(name=name) for name in field_names
    )
    field_tmpl = _m_field_tmpl if mutable else _imm_field_tmpl
    field_defs = '\n'.join(
        field_tmpl.format(index=index, name=name)
        for index, name in enumerate(field_names)
    )

    if mutable:
        return """\
class {typename}(list):
    '{typename}({arg_list})'

    __slots__ = ()

    _fields = {field_names!r}

    def __new__(_cls, *args, **kwargs):
        'Create new instance of {typename}({arg_list})'
        return _list.__new__(_cls)

    def __init__(self, {arg_list}):
        self[:] = ({arg_list})

    @classmethod
    def _make(cls, iterable, new=_list, len=len):
        'Make a new {typename} object from a sequence or iterable'
        result = new(iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d}'
                            ' arguments, got %d' % len(result))
        return cls(*result)

    def __repr__(self):
        'Return a nicely formatted representation string'
        tmpl = self.__class__.__name__ + '({repr_fmt})'
        return tmpl % tuple(self)

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values'
        return OrderedDict(zip(self._fields, self))

    def _replace(_self, **kwds):
        'Return a new {typename} object replacing field(s) with new values'
        result = _self._make(map(kwds.pop, {field_names!r}, _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % kwds.keys())
        return result

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    __dict__ = _property(_asdict)

    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        pass

{field_defs}
""".format(
            typename=typename,
            arg_list=arg_list,
            field_names=tuple(field_names),
            num_fields=len(field_names),
            repr_fmt=repr_fmt,
            field_defs=field_defs,
        )

    return """\
class {typename}(tuple):
    '{typename}({arg_list})'

    __slots__ = ()

    _fields = {field_names!r}

    def __new__(_cls, {arg_list}):
        'Create new instance of {typename}({arg_list})'
        return _tuple.__new__(_cls, ({arg_list}))

    @classmethod
    def _make(cls, iterable, new=_tuple.__new__, len=len):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d}'
                            ' arguments, got %d' % len(result))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        tmpl = self.__class__.__name__ + '({repr_fmt})'
        return tmpl % self

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values'
        return OrderedDict(zip(self._fields, self))

    def _replace(_self, **kwds):
        'Return a new {typename} object replacing field(s) with new values'
        result = _self._make(map(kwds.pop, {field_names!r}, _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % kwds.keys())
        return result

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    __dict__ = _property(_asdict)

    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        pass

{field_defs}
""".format(
        typename=typename,
        arg_list=arg_list,
        field_names=tuple(field_names),
        num_fields=len(field_names),
        repr_fmt=repr_fmt,
        field_defs=field_defs,
    )


def _make_type(typename, field_names, verbose, rename, mutable):
    field_names = _validate_names(typename, field_names, rename)
    source = _create_source(typename, field_names, mutable)

    if verbose:
        print(source)

    namespace = {
        '_itemgetter': _itemgetter,
        '_itemsetter': _itemsetter,
        '_property': property,
        '_tuple': tuple,
        '_list': list,
        'OrderedDict': OrderedDict,
        '__name__': ('namedlist_' if mutable else 'namedtuple_') + typename,
    }

    try:
        exec(source, namespace)
    except SyntaxError as exc:
        raise SyntaxError(exc.msg + ':\n' + source)

    result = namespace[typename]

    try:
        frame = _sys._getframe(2)
        result.__module__ = frame.f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result


def namedtuple(typename, field_names, verbose=False, rename=False):
    """Return a new tuple subclass with named fields."""
    return _make_type(typename, field_names, verbose, rename, False)


def namedlist(typename, field_names, verbose=False, rename=False):
    """Return a new mutable list subclass with named fields."""
    return _make_type(typename, field_names, verbose, rename, True)