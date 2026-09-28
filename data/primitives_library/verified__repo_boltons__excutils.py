import sys
import traceback
import linecache
from collections import namedtuple


__all__ = ['ExceptionCauseMixin']


class ExceptionCauseMixin(Exception):
    """
    Mixin for exceptions that retain information about an exception which
    caused them.

    It should generally precede builtin exception classes in an inheritance
    declaration, since builtin exception implementations do not consistently
    delegate through ``super()``.
    """

    cause = None

    def __new__(cls, *args, **kwargs):
        cause = None
        if args and isinstance(args[0], Exception):
            cause = args[0]
            args = args[1:]

        instance = super().__new__(cls, *args, **kwargs)
        instance.cause = cause

        if cause is None:
            return instance

        inherited_root = getattr(cause, 'root_cause', None)
        instance.root_cause = cause if inherited_root is None else inherited_root

        inherited_trace = getattr(cause, 'full_trace', None)
        if inherited_trace is not None:
            instance.full_trace = list(inherited_trace)
            instance._tb = list(cause._tb)
            instance._stack = list(cause._stack)
            return instance

        try:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_type is None and exc_value is None:
                return instance

            if cause is exc_value or inherited_root is exc_value:
                instance._tb = _extract_from_tb(exc_tb)
                instance._stack = _extract_from_frame(exc_tb.tb_frame)
                instance.full_trace = instance._stack[:-1] + instance._tb
        finally:
            del exc_tb

        return instance

    def get_str(self):
        parts = []
        trace_text = self._get_trace_str()
        if trace_text:
            parts.extend(('Traceback (most recent call last):\n', trace_text))
        parts.append(self._get_exc_str())
        return ''.join(parts)

    def _get_message(self):
        args = getattr(self, 'args', [])
        if self.cause:
            args = args[1:]
        if args and args[0]:
            return args[0]
        return ''

    def _get_trace_str(self):
        if not self.cause:
            return super().__repr__()
        if self.full_trace:
            return ''.join(traceback.format_list(self.full_trace))
        return ''

    def _get_exc_str(self, incl_name=True):
        cause_text = _format_exc(self.root_cause)
        message = self._get_message()

        parts = []
        if incl_name:
            parts.extend((self.__class__.__name__, ': '))

        if message:
            parts.extend((message, ' (caused by ', cause_text, ')'))
        else:
            parts.extend((' caused by ', cause_text))

        return ''.join(parts)

    def __str__(self):
        if not self.cause:
            return super().__str__()

        trace_text = self._get_trace_str()
        if trace_text:
            parts = []
            message = self._get_message()
            if message:
                parts.extend((message, ' --- '))
            parts.extend((
                'Wrapped traceback (most recent call last):\n',
                trace_text,
                self._get_exc_str(incl_name=True),
            ))
            return ''.join(parts)

        return self._get_exc_str(incl_name=False)


def _format_exc(exc, message=None):
    if message is None:
        message = exc
    return traceback._format_final_exc_line(
        exc.__class__.__name__, message
    ).rstrip()


_BaseTBItem = namedtuple('_BaseTBItem', 'filename lineno name line')


class _TBItem(_BaseTBItem):
    def __repr__(self):
        return super().__repr__() + ' <%r>' % self.frame_id


class _DeferredLine:
    def __init__(self, filename, lineno, module_globals=None):
        self.filename = filename
        self.lineno = lineno
        globals_map = module_globals or {}
        self.module_globals = {
            key: value
            for key, value in globals_map.items()
            if key in ('__name__', '__loader__')
        }

    def __eq__(self, other):
        return (self.lineno, self.filename) == (other.lineno, other.filename)

    def __ne__(self, other):
        return (self.lineno, self.filename) != (other.lineno, other.filename)

    def __str__(self):
        if hasattr(self, '_line'):
            return self._line

        linecache.checkcache(self.filename)
        line = linecache.getline(
            self.filename,
            self.lineno,
            self.module_globals,
        )
        if line:
            line = line.strip()
        else:
            line = None
        self._line = line
        return line

    def __repr__(self):
        return repr(str(self))

    def __len__(self):
        return len(str(self))

    def strip(self):
        return str(self).strip()


def _extract_from_frame(f=None, limit=None):
    entries = []

    if f is None:
        f = sys._getframe(1)

    if limit is None:
        limit = getattr(sys, 'tracebacklimit', 1000)

    count = 0
    while f is not None and count < limit:
        code = f.f_code
        filename = code.co_filename
        lineno = f.f_lineno
        name = code.co_name
        line = _DeferredLine(filename, lineno, f.f_globals)
        item = _TBItem(filename, lineno, name, line)
        item.frame_id = id(f)
        entries.append(item)
        f = f.f_back
        count += 1

    entries.reverse()
    return entries


def _extract_from_tb(tb, limit=None):
    entries = []

    if limit is None:
        limit = getattr(sys, 'tracebacklimit', 1000)

    count = 0
    while tb is not None and count < limit:
        frame = tb.tb_frame
        code = frame.f_code
        filename = code.co_filename
        lineno = tb.tb_lineno
        name = code.co_name
        line = _DeferredLine(filename, lineno, frame.f_globals)
        item = _TBItem(filename, lineno, name, line)
        item.frame_id = id(frame)
        entries.append(item)
        tb = tb.tb_next
        count += 1

    return entries


class MathError(ExceptionCauseMixin, ValueError):
    pass


def whoops_math():
    return 1 / 0


def math_lol(n=0):
    if n < 3:
        return math_lol(n=n + 1)
    try:
        return whoops_math()
    except ZeroDivisionError as error:
        exc = MathError(error, 'ya done messed up')
        raise exc


def main():
    try:
        math_lol()
    except ValueError as error:
        exc = MathError(error, 'hi')
        raise exc


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import pdb
        pdb.post_mortem()
        raise