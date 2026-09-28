import re
import sys
import linecache


__all__ = [
    'ExceptionInfo', 'TracebackInfo', 'Callpoint',
    'ContextualExceptionInfo', 'ContextualTracebackInfo',
    'ContextualCallpoint', 'print_exception', 'ParsedException',
]


class _DeferredLine:
    __slots__ = ('filename', 'lineno', '_line', '_mod_name', '_mod_loader')

    def __init__(self, filename, lineno, module_globals=None):
        self.filename = filename
        self.lineno = lineno
        if module_globals is None:
            self._mod_name = None
            self._mod_loader = None
        else:
            self._mod_name = module_globals.get('__name__')
            self._mod_loader = module_globals.get('__loader__')

    def __str__(self):
        try:
            return self._line
        except AttributeError:
            pass
        try:
            linecache.checkcache(self.filename)
            globals_dict = {
                '__name__': self._mod_name,
                '__loader__': self._mod_loader,
            }
            line = linecache.getline(self.filename, self.lineno, globals_dict)
            line = line.rstrip()
        except (KeyError, OSError):
            line = ''
        self._line = line
        return line

    def __repr__(self):
        return repr(str(self))

    def __len__(self):
        return len(str(self))

    def __eq__(self, other):
        try:
            return (self.filename, self.lineno) == (other.filename, other.lineno)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other


class Callpoint:
    __slots__ = (
        'func_name', 'lineno', 'module_name', 'module_path', 'lasti', 'line',
    )

    def __init__(self, module_name, module_path, func_name,
                 lineno, lasti, line=None):
        self.func_name = func_name
        self.lineno = lineno
        self.module_name = module_name
        self.module_path = module_path
        self.lasti = lasti
        self.line = line

    @classmethod
    def from_current(cls, level=1):
        return cls.from_frame(sys._getframe(level))

    @classmethod
    def from_frame(cls, frame):
        code = frame.f_code
        return cls(
            frame.f_globals.get('__name__', ''),
            code.co_filename,
            code.co_name,
            frame.f_lineno,
            frame.f_lasti,
            _DeferredLine(code.co_filename, frame.f_lineno, frame.f_globals),
        )

    @classmethod
    def from_tb(cls, tb):
        frame = tb.tb_frame
        code = frame.f_code
        return cls(
            frame.f_globals.get('__name__', ''),
            code.co_filename,
            code.co_name,
            tb.tb_lineno,
            tb.tb_lasti,
            _DeferredLine(code.co_filename, tb.tb_lineno, frame.f_globals),
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get('module_name'),
            data.get('module_path'),
            data.get('func_name'),
            data.get('lineno'),
            data.get('lasti'),
            data.get('line'),
        )

    def to_dict(self):
        ret = {}
        for slot in self.__slots__:
            try:
                value = getattr(self, slot)
            except AttributeError:
                continue
            if isinstance(value, _DeferredLine):
                value = str(value)
            ret[slot] = value
        return ret

    def tb_frame_str(self):
        ret = '  File "{}", line {}, in {}\n'.format(
            self.module_path, self.lineno, self.func_name)
        if self.line:
            ret += '    {}\n'.format(str(self.line).strip())
        return ret

    def __repr__(self):
        cls_name = self.__class__.__name__
        values = [getattr(self, attr, None) for attr in self.__slots__]
        if not any(values):
            return object.__repr__(self)
        return '{}({})'.format(cls_name, ', '.join(repr(v) for v in values))


class TracebackInfo:
    __slots__ = ('frames',)

    _callpoint_type = Callpoint

    def __init__(self, frames):
        self.frames = list(frames)

    @classmethod
    def from_frame(cls, frame=None, limit=None):
        if frame is None:
            frame = sys._getframe().f_back

        frames = []
        while frame is not None:
            frames.append(cls._callpoint_type.from_frame(frame))
            if limit is not None and len(frames) >= limit:
                break
            frame = frame.f_back
        frames.reverse()
        return cls(frames)

    @classmethod
    def from_current(cls, limit=None):
        return cls.from_frame(sys._getframe().f_back, limit=limit)

    @classmethod
    def from_traceback(cls, tb=None, limit=None):
        if tb is None:
            tb = sys.exc_info()[2]

        frames = []
        while tb is not None:
            frames.append(cls._callpoint_type.from_tb(tb))
            if limit is not None and len(frames) >= limit:
                break
            tb = tb.tb_next
        return cls(frames)

    @classmethod
    def from_dict(cls, data):
        frames = [
            cls._callpoint_type.from_dict(frame)
            for frame in data.get('frames', ())
        ]
        return cls(frames)

    def to_dict(self):
        return {'frames': [frame.to_dict() for frame in self.frames]}

    def get_formatted(self):
        return ''.join(frame.tb_frame_str() for frame in self.frames)

    def __str__(self):
        return self.get_formatted()

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.frames)


class ExceptionInfo:
    __slots__ = ('exc_type', 'exc_msg', 'tb_info')

    _tb_info_type = TracebackInfo

    def __init__(self, exc_type, exc_msg, tb_info):
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.tb_info = tb_info

    @classmethod
    def from_exc_info(cls, exc_type, exc, tb):
        if isinstance(exc_type, type):
            exc_type = exc_type.__name__
        else:
            exc_type = str(exc_type)
        return cls(exc_type, str(exc), cls._tb_info_type.from_traceback(tb))

    @classmethod
    def from_current(cls):
        return cls.from_exc_info(*sys.exc_info())

    @classmethod
    def from_exception(cls, exc):
        return cls.from_exc_info(type(exc), exc, exc.__traceback__)

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get('exc_type'),
            data.get('exc_msg'),
            cls._tb_info_type.from_dict(data.get('tb_info', {})),
        )

    def to_dict(self):
        return {
            'exc_type': self.exc_type,
            'exc_msg': self.exc_msg,
            'tb_info': self.tb_info.to_dict(),
        }

    def get_formatted(self):
        ret = 'Traceback (most recent call last):\n'
        ret += self.tb_info.get_formatted()
        if self.exc_msg:
            ret += '{}: {}\n'.format(self.exc_type, self.exc_msg)
        else:
            ret += '{}\n'.format(self.exc_type)
        return ret

    def __str__(self):
        return self.get_formatted()

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.exc_type, self.exc_msg, self.tb_info)


class ContextualCallpoint(Callpoint):
    __slots__ = ('pre_lines', 'post_lines', 'locals')

    def __init__(self, module_name, module_path, func_name, lineno, lasti,
                 line=None, pre_lines=None, post_lines=None, locals=None):
        super().__init__(module_name, module_path, func_name, lineno, lasti, line)
        self.pre_lines = pre_lines if pre_lines is not None else []
        self.post_lines = post_lines if post_lines is not None else []
        self.locals = locals if locals is not None else {}

    @staticmethod
    def _context(filename, lineno, module_globals, context_lines):
        try:
            linecache.checkcache(filename)
            globals_dict = {
                '__name__': module_globals.get('__name__'),
                '__loader__': module_globals.get('__loader__'),
            }
            lines = linecache.getlines(filename, globals_dict)
        except (KeyError, OSError):
            lines = []

        before_start = max(0, lineno - context_lines - 1)
        before_end = max(0, lineno - 1)
        after_start = lineno
        after_end = lineno + context_lines
        pre_lines = [line.rstrip() for line in lines[before_start:before_end]]
        post_lines = [line.rstrip() for line in lines[after_start:after_end]]
        return pre_lines, post_lines

    @staticmethod
    def _locals(frame):
        ret = {}
        for key, value in frame.f_locals.items():
            try:
                ret[key] = repr(value)
            except Exception:
                ret[key] = '<unrepresentable {}>'.format(type(value).__name__)
        return ret

    @classmethod
    def from_frame(cls, frame, context_lines=5):
        code = frame.f_code
        pre_lines, post_lines = cls._context(
            code.co_filename, frame.f_lineno, frame.f_globals, context_lines)
        return cls(
            frame.f_globals.get('__name__', ''),
            code.co_filename,
            code.co_name,
            frame.f_lineno,
            frame.f_lasti,
            _DeferredLine(code.co_filename, frame.f_lineno, frame.f_globals),
            pre_lines,
            post_lines,
            cls._locals(frame),
        )

    @classmethod
    def from_tb(cls, tb, context_lines=5):
        frame = tb.tb_frame
        code = frame.f_code
        pre_lines, post_lines = cls._context(
            code.co_filename, tb.tb_lineno, frame.f_globals, context_lines)
        return cls(
            frame.f_globals.get('__name__', ''),
            code.co_filename,
            code.co_name,
            tb.tb_lineno,
            tb.tb_lasti,
            _DeferredLine(code.co_filename, tb.tb_lineno, frame.f_globals),
            pre_lines,
            post_lines,
            cls._locals(frame),
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get('module_name'),
            data.get('module_path'),
            data.get('func_name'),
            data.get('lineno'),
            data.get('lasti'),
            data.get('line'),
            data.get('pre_lines'),
            data.get('post_lines'),
            data.get('locals'),
        )

    def to_dict(self):
        ret = {
            'func_name': self.func_name,
            'lineno': self.lineno,
            'module_name': self.module_name,
            'module_path': self.module_path,
            'lasti': self.lasti,
            'line': str(self.line) if isinstance(self.line, _DeferredLine) else self.line,
            'pre_lines': self.pre_lines,
            'post_lines': self.post_lines,
            'locals': self.locals,
        }
        return ret


class ContextualTracebackInfo(TracebackInfo):
    _callpoint_type = ContextualCallpoint

    @classmethod
    def from_frame(cls, frame=None, limit=None, context_lines=5):
        if frame is None:
            frame = sys._getframe().f_back
        frames = []
        while frame is not None:
            frames.append(cls._callpoint_type.from_frame(frame, context_lines))
            if limit is not None and len(frames) >= limit:
                break
            frame = frame.f_back
        frames.reverse()
        return cls(frames)

    @classmethod
    def from_current(cls, limit=None, context_lines=5):
        return cls.from_frame(
            sys._getframe().f_back, limit=limit, context_lines=context_lines)

    @classmethod
    def from_traceback(cls, tb=None, limit=None, context_lines=5):
        if tb is None:
            tb = sys.exc_info()[2]
        frames = []
        while tb is not None:
            frames.append(cls._callpoint_type.from_tb(tb, context_lines))
            if limit is not None and len(frames) >= limit:
                break
            tb = tb.tb_next
        return cls(frames)


class ContextualExceptionInfo(ExceptionInfo):
    _tb_info_type = ContextualTracebackInfo

    @classmethod
    def from_exc_info(cls, exc_type, exc, tb, limit=None, context_lines=5):
        if isinstance(exc_type, type):
            exc_type = exc_type.__name__
        else:
            exc_type = str(exc_type)
        tb_info = cls._tb_info_type.from_traceback(
            tb, limit=limit, context_lines=context_lines)
        return cls(exc_type, str(exc), tb_info)

    @classmethod
    def from_current(cls, limit=None, context_lines=5):
        return cls.from_exc_info(*sys.exc_info(), limit=limit,
                                 context_lines=context_lines)

    @classmethod
    def from_exception(cls, exc, limit=None, context_lines=5):
        return cls.from_exc_info(type(exc), exc, exc.__traceback__,
                                 limit=limit, context_lines=context_lines)


def print_exception(exc_type, exc, tb, limit=None, file=None):
    if file is None:
        file = sys.stderr
    info = ExceptionInfo.from_exc_info(exc_type, exc, tb)
    if limit is not None:
        info.tb_info.frames = info.tb_info.frames[:limit]
    file.write(info.get_formatted())


class ParsedException:
    __slots__ = ('exc_type', 'exc_msg', 'frames')

    _frame_re = re.compile(
        r'^\s*File "(?P<module_path>.*?)", line (?P<lineno>\d+),'
        r'(?: in (?P<func_name>.*))?$')
    _exc_re = re.compile(
        r'^(?P<exc_type>(?:[\w.]+(?:Error|Exception|Warning)?|'
        r'(?:\w+\.)*\w+))(?::\s?(?P<exc_msg>.*))?$')

    def __init__(self, exc_type, exc_msg, frames):
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.frames = list(frames)

    @classmethod
    def from_string(cls, text):
        lines = text.splitlines()
        traceback_indexes = [
            i for i, line in enumerate(lines)
            if line.strip() == 'Traceback (most recent call last):'
        ]
        start = traceback_indexes[-1] + 1 if traceback_indexes else 0

        frames = []
        exception_line = None
        index = start
        while index < len(lines):
            match = cls._frame_re.match(lines[index])
            if match:
                groups = match.groupdict()
                line = None
                if index + 1 < len(lines) and not cls._frame_re.match(lines[index + 1]):
                    candidate = lines[index + 1]
                    if candidate.startswith((' ', '\t')):
                        line = candidate.strip()
                        index += 1
                frames.append(Callpoint(
                    None,
                    groups['module_path'],
                    groups.get('func_name') or '<module>',
                    int(groups['lineno']),
                    None,
                    line,
                ))
            elif lines[index].strip() and not lines[index].startswith((' ', '\t')):
                exception_line = lines[index].strip()
            index += 1

        if exception_line is None:
            nonempty = [line.strip() for line in lines if line.strip()]
            exception_line = nonempty[-1] if nonempty else ''

        match = cls._exc_re.match(exception_line)
        if match:
            exc_type = match.group('exc_type')
            exc_msg = match.group('exc_msg') or ''
        else:
            exc_type = exception_line
            exc_msg = ''
        return cls(exc_type, exc_msg, frames)

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get('exc_type'),
            data.get('exc_msg'),
            [Callpoint.from_dict(frame) for frame in data.get('frames', ())],
        )

    def to_dict(self):
        return {
            'exc_type': self.exc_type,
            'exc_msg': self.exc_msg,
            'frames': [frame.to_dict() if hasattr(frame, 'to_dict') else frame
                       for frame in self.frames],
        }

    def get_formatted(self):
        ret = 'Traceback (most recent call last):\n'
        for frame in self.frames:
            if hasattr(frame, 'tb_frame_str'):
                ret += frame.tb_frame_str()
            else:
                ret += str(frame)
        if self.exc_msg:
            ret += '{}: {}\n'.format(self.exc_type, self.exc_msg)
        else:
            ret += '{}\n'.format(self.exc_type)
        return ret

    def __str__(self):
        return self.get_formatted()

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.exc_type, self.exc_msg, self.frames)