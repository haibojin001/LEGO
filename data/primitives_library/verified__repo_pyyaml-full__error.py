__all__ = ['Mark', 'YAMLError', 'MarkedYAMLError']


class Mark:
    def __init__(self, name, index, line, column, buffer, pointer):
        self.name = name
        self.index = index
        self.line = line
        self.column = column
        self.buffer = buffer
        self.pointer = pointer

    def get_snippet(self, indent=4, max_length=75):
        if self.buffer is None:
            return None

        beginning = self.pointer
        prefix = ''
        while beginning > 0 and self.buffer[beginning - 1] not in '\0\r\n\x85\u2028\u2029':
            beginning -= 1
            if self.pointer - beginning > max_length / 2 - 1:
                prefix = ' ... '
                beginning += 5
                break

        ending = self.pointer
        suffix = ''
        while ending < len(self.buffer) and self.buffer[ending] not in '\0\r\n\x85\u2028\u2029':
            ending += 1
            if ending - self.pointer > max_length / 2 - 1:
                suffix = ' ... '
                ending -= 5
                break

        text = self.buffer[beginning:ending]
        return (
            ' ' * indent + prefix + text + suffix + '\n'
            + ' ' * (indent + self.pointer - beginning + len(prefix)) + '^'
        )

    def __str__(self):
        location = '  in "%s", line %d, column %d' % (
            self.name,
            self.line + 1,
            self.column + 1,
        )
        snippet = self.get_snippet()
        if snippet is not None:
            location += ':\n' + snippet
        return location


class YAMLError(Exception):
    pass


class MarkedYAMLError(YAMLError):
    def __init__(self, context=None, context_mark=None,
                 problem=None, problem_mark=None, note=None):
        self.context = context
        self.context_mark = context_mark
        self.problem = problem
        self.problem_mark = problem_mark
        self.note = note

    def __str__(self):
        parts = []

        if self.context is not None:
            parts.append(self.context)

        show_context_mark = (
            self.context_mark is not None
            and (
                self.problem is None
                or self.problem_mark is None
                or self.context_mark.name != self.problem_mark.name
                or self.context_mark.line != self.problem_mark.line
                or self.context_mark.column != self.problem_mark.column
            )
        )
        if show_context_mark:
            parts.append(str(self.context_mark))

        if self.problem is not None:
            parts.append(self.problem)

        if self.problem_mark is not None:
            parts.append(str(self.problem_mark))

        if self.note is not None:
            parts.append(self.note)

        return '\n'.join(parts)