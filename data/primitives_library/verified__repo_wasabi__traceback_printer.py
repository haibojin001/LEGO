import os
from typing import Optional, Tuple, Union

from .util import NO_UTF8, color, supports_ansi


LINE_EDGE = "|_" if NO_UTF8 else "└─"
LINE_FORK = "|__" if NO_UTF8 else "├─"
LINE_PATH = "__" if NO_UTF8 else "──"


class TracebackPrinter(object):
    def __init__(
        self,
        color_error: Union[str, int] = "red",
        color_tb: Union[str, int] = "blue",
        color_highlight: Union[str, int] = "yellow",
        indent: int = 2,
        tb_base: Optional[str] = None,
        tb_exclude: Tuple = tuple(),
        tb_range_start: int = -5,
        tb_range_end: int = -2,
    ):
        self.color_error = color_error
        self.color_tb = color_tb
        self.color_highlight = color_highlight
        self.indent = " " * indent

        if tb_base == ".":
            tb_base = os.getcwd() + os.path.sep
        elif tb_base is not None:
            tb_base = "/{}/".format(tb_base)

        self.tb_base = tb_base
        self.tb_exclude = tuple(tb_exclude)
        self.tb_range_start = tb_range_start
        self.tb_range_end = tb_range_end
        self.supports_ansi = supports_ansi()

    def __call__(self, title: str, *texts, **settings) -> str:
        highlight = settings.get("highlight", False)
        traceback_data = settings.get("tb", None)

        if self.supports_ansi:
            title = color(title, fg=self.color_error, bold=True)

        if texts:
            detail = "\n" + "\n".join(self.indent + text for text in texts)
        else:
            detail = ""

        if traceback_data:
            traceback_text = self._get_traceback(traceback_data, highlight)
        else:
            traceback_text = ""

        return "\n\n{}{}{}{}\n".format(
            self.indent, title, detail, traceback_text
        )

    def _get_traceback(self, tb, highlight):
        records = [
            record for record in tb
            if not record[0].endswith(self.tb_exclude)
        ]

        if self.tb_range_end is None:
            records = records[self.tb_range_start:]
        else:
            records = records[self.tb_range_start:self.tb_range_end]

        formatted = []
        total = len(records)
        for index, (path, line, function, source) in enumerate(records):
            formatted.append(
                self._format_traceback(
                    path,
                    line,
                    function,
                    source,
                    index,
                    total,
                    highlight,
                )
            )

        body = "\n".join(formatted).strip()
        heading = "Traceback:"
        if self.supports_ansi:
            heading = color(heading, fg=self.color_tb, bold=True)

        return "\n\n{0}{1}\n{0}{2}".format(self.indent, heading, body)

    def _format_traceback(self, path, line, fn, text, i, count, highlight):
        marker = LINE_EDGE if i == count - 1 else LINE_FORK
        marker += LINE_PATH * i

        if self.tb_base and self.tb_base in path:
            path = path.rsplit(self.tb_base, 1)[1]

        if i == count - 1:
            source = self._format_user_error(text, i, highlight)
        else:
            source = ""

        if self.supports_ansi:
            fn = color(fn, bold=True)
            path = color(path, underline=True)

        return "{}{} {} in {}:{}{}".format(
            self.indent, marker, fn, path, line, source
        )

    def _format_user_error(self, text, i, highlight):
        pointer = ("  " * i) + " >>>"

        if self.supports_ansi:
            pointer = color(pointer, fg=self.color_error)

        if highlight and self.supports_ansi:
            replacement = color(highlight, fg=self.color_highlight)
            text = text.replace(highlight, replacement)

        return "\n{}  {} {}".format(self.indent, pointer, text)