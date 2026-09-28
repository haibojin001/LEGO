__all__ = ['JSONDecodeError']


def linecol(doc, pos):
    line = doc.count('\n', 0, pos) + 1
    if line == 1:
        column = pos + 1
    else:
        column = pos - doc.rfind('\n', 0, pos)
    return line, column


def errmsg(msg, doc, pos, end=None):
    line, column = linecol(doc, pos)
    detail = msg.replace('%r', repr(doc[pos:pos + 1]))
    if end is None:
        return '%s: line %d column %d (char %d)' % (
            detail, line, column, pos
        )
    endline, endcolumn = linecol(doc, end)
    return '%s: line %d column %d - line %d column %d (char %d - %d)' % (
        detail, line, column, endline, endcolumn, pos, end
    )


class JSONDecodeError(ValueError):
    def __init__(self, msg, doc, pos, end=None):
        ValueError.__init__(self, errmsg(msg, doc, pos, end))
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.end = end
        self.lineno, self.colno = linecol(doc, pos)
        if end is None:
            self.endlineno = None
            self.endcolno = None
        else:
            self.endlineno, self.endcolno = linecol(doc, end)

    def __reduce__(self):
        return self.__class__, (self.msg, self.doc, self.pos, self.end)