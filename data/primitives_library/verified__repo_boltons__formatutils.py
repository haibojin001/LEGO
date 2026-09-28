import re
from string import Formatter


__all__ = [
    'DeferredValue',
    'get_format_args',
    'tokenize_format_str',
    'construct_format_field_str',
    'infer_positional_format_args',
    'BaseFormatField',
]


_UNSET = object()

_pos_farg_re = re.compile(r'({{)|(}})|({[:!.\[}])')

_INTCHARS = 'bcdoxXn'
_FLOATCHARS = 'eEfFgGn%'
_TYPE_MAP = {char: int for char in _INTCHARS}
_TYPE_MAP.update((char, float) for char in _FLOATCHARS)
_TYPE_MAP['s'] = str


class DeferredValue:
    """A wrapper which defers invocation of a zero-argument callable."""

    def __init__(self, func, cache_value=True):
        self.func = func
        self.cache_value = cache_value
        self._value = _UNSET

    def get_value(self):
        """Get the wrapped callable's value, evaluating it as needed."""
        if self.cache_value and self._value is not _UNSET:
            return self._value

        value = self.func()
        if self.cache_value:
            self._value = value
        return value

    def __str__(self):
        return str(self.get_value())

    def __repr__(self):
        return repr(self.get_value())

    def __format__(self, format_spec):
        return format(self.get_value(), format_spec)


def construct_format_field_str(fname, fspec, conv):
    """
    Constructs a format field string from the field name, spec, and
    conversion character.
    """
    if fname is None:
        return ''

    ret = '{' + fname
    if conv:
        ret += '!' + conv
    if fspec:
        ret += ':' + fspec
    ret += '}'
    return ret


def split_format_str(fstr):
    """Split a format string into literal and field-string pairs."""
    ret = []
    formatter = Formatter()

    for lit, fname, fspec, conv in formatter.parse(fstr):
        if fname is None:
            ret.append((lit, None))
        else:
            ret.append((lit, construct_format_field_str(fname, fspec, conv)))
    return ret


def infer_positional_format_args(fstr):
    """
    Convert automatic positional fields into explicitly numbered fields.
    """
    ret = []
    max_anon = 0
    prev_end = 0

    for match in _pos_farg_re.finditer(fstr):
        start, end = match.span()
        group = match.group()

        if prev_end < start:
            ret.append(fstr[prev_end:start])
        prev_end = end

        if group == '{{' or group == '}}':
            ret.append(group)
            continue

        ret.append('{' + str(max_anon) + group[1:])
        max_anon += 1

    ret.append(fstr[prev_end:])
    return ''.join(ret)


def get_format_args(fstr):
    """
    Turn a format string into positional and named argument requirements.

    The returned value is a two-tuple: a list of positional argument
    requirements followed by a list of named argument requirements. Every
    requirement is a ``(name, type)`` tuple.
    """
    formatter = Formatter()
    fargs = []
    fkwargs = []
    dedup = set()

    def add_arg(argname, type_char='s'):
        if argname in dedup:
            return
        dedup.add(argname)

        argtype = _TYPE_MAP.get(type_char, str)
        try:
            fargs.append((int(argname), argtype))
        except ValueError:
            fkwargs.append((argname, argtype))

    for lit, fname, fspec, conv in formatter.parse(fstr):
        if fname is None:
            continue

        type_char = fspec[-1:]
        fname_list = re.split('[.[]', fname)
        if len(fname_list) > 1:
            raise ValueError('encountered compound format arg: %r' % fname)

        try:
            base_fname = fname_list[0]
            assert base_fname
        except (IndexError, AssertionError):
            raise ValueError('encountered anonymous positional argument')

        add_arg(fname, type_char)

        for sublit, subfname, subspec, subconversion in formatter.parse(fspec):
            if subfname is not None:
                add_arg(subfname)

    return fargs, fkwargs


def tokenize_format_str(fstr, resolve_pos=True):
    """
    Tokenize a format string into literal strings and BaseFormatField objects.

    If *resolve_pos* is true, automatic positional fields are converted to
    explicitly numbered fields first.
    """
    ret = []

    if resolve_pos:
        fstr = infer_positional_format_args(fstr)

    formatter = Formatter()
    for lit, fname, fspec, conv in formatter.parse(fstr):
        if lit:
            ret.append(lit)
        if fname is not None:
            ret.append(BaseFormatField(fname, fspec, conv))

    return ret


class BaseFormatField:
    """
    A representation of a replacement field in a bracket-style format string.
    """

    def __init__(self, fname, fspec='', conv=None):
        self.set_fname(fname)
        self.set_fspec(fspec)
        self.set_conv(conv)

    def set_fname(self, fname):
        """Set the field name."""
        path_list = re.split('[.[]', fname)
        self.base_name = path_list[0]
        self.fname = fname
        self.subpath = path_list[1:]
        self.is_positional = not self.base_name or self.base_name.isdigit()

    def set_fspec(self, fspec):
        """Set the field format specification."""
        self.fspec = fspec
        self.type_char = fspec[-1:] or None

    def set_conv(self, conv):
        """Set the field conversion."""
        self.conv = conv

    def __str__(self):
        return construct_format_field_str(self.fname, self.fspec, self.conv)

    def __repr__(self):
        return '%s(%r, %r, %r)' % (
            self.__class__.__name__,
            self.fname,
            self.fspec,
            self.conv,
        )