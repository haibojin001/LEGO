import re

from genshi.compat import text_type
from genshi.core import TEXT
from genshi.template.base import BadDirectiveError, Template, \
                                 TemplateSyntaxError, EXEC, INCLUDE, SUB
from genshi.template.eval import Suite
from genshi.template.directives import *
from genshi.template.interpolation import interpolate

__all__ = ['NewTextTemplate', 'OldTextTemplate', 'TextTemplate']
__docformat__ = 'restructuredtext en'


class NewTextTemplate(Template):
    directives = [
        ('def', DefDirective),
        ('when', WhenDirective),
        ('otherwise', OtherwiseDirective),
        ('for', ForDirective),
        ('if', IfDirective),
        ('choose', ChooseDirective),
        ('with', WithDirective)
    ]
    serializer = 'text'

    _DIRECTIVE_RE = r'((?<!\\)%s\s*(\w+)\s*(.*?)\s*%s|(?<!\\)%s.*?%s)'
    _ESCAPE_RE = r'\\\n|\\\r\n|\\(\\)|\\(%s)|\\(%s)'

    def __init__(self, source, filepath=None, filename=None, loader=None,
                 encoding=None, lookup='strict', allow_exec=False,
                 delims=('{%', '%}', '{#', '#}')):
        self.delimiters = delims
        Template.__init__(self, source, filepath=filepath, filename=filename,
                          loader=loader, encoding=encoding, lookup=lookup)

    def _get_delims(self):
        return self._delims

    def _set_delims(self, delims):
        if len(delims) != 4:
            raise ValueError('delimiers tuple must have exactly four elements')
        self._delims = delims
        escaped = [re.escape(item) for item in delims]
        self._directive_re = re.compile(
            self._DIRECTIVE_RE % tuple(escaped), re.DOTALL
        )
        self._escape_re = re.compile(
            self._ESCAPE_RE % tuple(re.escape(item) for item in delims[::2])
        )

    delimiters = property(
        _get_delims, _set_delims,
        """The delimiters used for directive and comment constructs."""
    )

    def _parse(self, source, encoding):
        stream = []
        openings = {}
        depth = 0

        source = source.read()
        if not isinstance(source, text_type):
            source = source.decode(encoding or 'utf-8', 'replace')

        cursor = 0
        lineno = 1
        unescape = self._escape_re.sub

        def replace_escape(match):
            values = [value for value in match.groups() if value]
            return values[0] if values else ''

        for match in self._directive_re.finditer(source):
            start, end = match.span(1)

            if start > cursor:
                text = unescape(replace_escape, source[cursor:start])
                for event in interpolate(text, self.filepath, lineno,
                                         lookup=self.lookup):
                    stream.append(event)
                lineno += len(text.splitlines())

            lineno += len(source[start:end].splitlines())
            command, value = match.group(2, 3)

            if command == 'include':
                position = (self.filename, lineno, 0)
                included = list(interpolate(value, self.filepath, lineno, 0,
                                           lookup=self.lookup))
                if len(included) == 1 and included[0][0] is TEXT:
                    included = included[0][1]
                stream.append((INCLUDE, (included, None, []), position))

            elif command == 'python':
                if not self.allow_exec:
                    raise TemplateSyntaxError('Python code blocks not allowed',
                                              self.filepath, lineno)
                try:
                    suite = Suite(value, self.filepath, lineno,
                                  lookup=self.lookup)
                except SyntaxError as error:
                    raise TemplateSyntaxError(
                        error, self.filepath, lineno + (error.lineno or 1) - 1
                    )
                stream.append((EXEC, suite, (self.filename, lineno, 0)))

            elif command == 'end':
                depth -= 1
                if depth in openings:
                    directive, index = openings.pop(depth)
                    contents = stream[index:]
                    stream[index:] = [
                        (SUB, ([directive], contents),
                         (self.filepath, lineno, 0))
                    ]

            elif command:
                directive_class = self.get_directive(command)
                if directive_class is None:
                    raise BadDirectiveError(command)
                directive = (
                    0, directive_class, value, None,
                    (self.filepath, lineno, 0)
                )
                openings[depth] = (directive, len(stream))
                depth += 1

            cursor = end

        if cursor < len(source):
            text = unescape(replace_escape, source[cursor:])
            for event in interpolate(text, self.filepath, lineno,
                                     lookup=self.lookup):
                stream.append(event)

        return stream


class OldTextTemplate(Template):
    directives = [
        ('def', DefDirective),
        ('when', WhenDirective),
        ('otherwise', OtherwiseDirective),
        ('for', ForDirective),
        ('if', IfDirective),
        ('choose', ChooseDirective),
        ('with', WithDirective)
    ]
    serializer = 'text'

    _directive_re = re.compile(
        r'^[ \t]*#([A-Za-z_]\w*)[ \t]*(.*?)[ \t]*(?:\r?\n)?$'
    )
    _escaped_directive_re = re.compile(r'^([ \t]*)\\#')

    def _parse(self, source, encoding):
        source = source.read()
        if not isinstance(source, text_type):
            source = source.decode(encoding or 'utf-8', 'replace')

        stream = []
        openings = {}
        depth = 0
        python_block = None

        def append_text(value, line):
            value = self._escaped_directive_re.sub(r'\1#', value)
            for event in interpolate(value, self.filepath, line,
                                     lookup=self.lookup):
                stream.append(event)

        lines = source.splitlines(True)

        for lineno, line in enumerate(lines, 1):
            match = self._directive_re.match(line)

            if python_block is not None:
                if match and match.group(1) == 'end':
                    start_line, code = python_block
                    python_block = None
                    try:
                        suite = Suite(''.join(code), self.filepath, start_line,
                                      lookup=self.lookup)
                    except SyntaxError as error:
                        raise TemplateSyntaxError(
                            error, self.filepath,
                            start_line + (error.lineno or 1) - 1
                        )
                    stream.append((EXEC, suite, (self.filename, lineno, 0)))
                    continue
                python_block[1].append(line)
                continue

            if not match:
                append_text(line, lineno)
                continue

            command, value = match.group(1), match.group(2)

            if command == 'include':
                position = (self.filename, lineno, 0)
                included = list(interpolate(value, self.filepath, lineno, 0,
                                           lookup=self.lookup))
                if len(included) == 1 and included[0][0] is TEXT:
                    included = included[0][1]
                stream.append((INCLUDE, (included, None, []), position))
                continue

            if command == 'python':
                if not self.allow_exec:
                    raise TemplateSyntaxError('Python code blocks not allowed',
                                              self.filepath, lineno)
                if value:
                    try:
                        suite = Suite(value, self.filepath, lineno,
                                      lookup=self.lookup)
                    except SyntaxError as error:
                        raise TemplateSyntaxError(
                            error, self.filepath,
                            lineno + (error.lineno or 1) - 1
                        )
                    stream.append((EXEC, suite, (self.filename, lineno, 0)))
                else:
                    python_block = [lineno + 1, []]
                continue

            if command == 'end':
                depth -= 1
                if depth in openings:
                    directive, index = openings.pop(depth)
                    contents = stream[index:]
                    stream[index:] = [
                        (SUB, ([directive], contents),
                         (self.filepath, lineno, 0))
                    ]
                continue

            directive_class = self.get_directive(command)
            if directive_class is None:
                raise BadDirectiveError(command)

            directive = (
                0, directive_class, value, None,
                (self.filepath, lineno, 0)
            )
            openings[depth] = (directive, len(stream))
            depth += 1

        if python_block is not None:
            start_line, code = python_block
            try:
                suite = Suite(''.join(code), self.filepath, start_line,
                              lookup=self.lookup)
            except SyntaxError as error:
                raise TemplateSyntaxError(
                    error, self.filepath, start_line + (error.lineno or 1) - 1
                )
            stream.append((EXEC, suite, (self.filename, start_line, 0)))

        return stream


TextTemplate = OldTextTemplate