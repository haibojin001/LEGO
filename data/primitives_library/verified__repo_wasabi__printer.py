import datetime
import itertools
import os
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from multiprocessing import Process
from typing import Any, Collection, Dict, NoReturn, Optional, Union, cast, overload

from .compat import Literal
from .tables import row, table
from .util import COLORS, ICONS, MESSAGES, can_render
from .util import color as _color
from .util import locale_escape, supports_ansi, wrap


class Printer(object):
    def __init__(
        self,
        pretty: bool = True,
        no_print: bool = False,
        colors: Optional[Dict] = None,
        icons: Optional[Dict] = None,
        line_max: int = 80,
        animation: str = "⠙⠹⠸⠼⠴⠦⠧⠇⠏",
        animation_ascii: str = "|/-\\",
        hide_animation: bool = False,
        ignore_warnings: bool = False,
        env_prefix: str = "WASABI",
        timestamp: bool = False,
    ):
        env_log_friendly = os.getenv("{}_LOG_FRIENDLY".format(env_prefix), False)
        env_no_pretty = os.getenv("{}_NO_PRETTY".format(env_prefix), False)
        self._counts: Counter = Counter()
        self.pretty = pretty and not env_no_pretty
        self.no_print = no_print
        self.show_color = supports_ansi() and not env_log_friendly
        self.hide_animation = hide_animation or env_log_friendly
        self.ignore_warnings = ignore_warnings
        self.line_max = line_max
        self.colors = dict(COLORS)
        self.icons = dict(ICONS)
        self.timestamp = timestamp
        if colors:
            self.colors.update(colors)
        if icons:
            self.icons.update(icons)
        self.anim = animation if can_render(animation) else animation_ascii

    @property
    def counts(self) -> Counter:
        return self._counts

    def good(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Optional[int] = None,
    ):
        return self._get_msg(
            title, text, style=MESSAGES.GOOD, show=show, spaced=spaced, exits=exits
        )

    @overload
    def fail(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Optional[Literal[0, False]] = None,
    ):
        ...

    @overload
    def fail(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Literal[1, True] = True,
    ) -> NoReturn:
        ...

    def fail(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Optional[Union[int, bool]] = None,
    ) -> Union[str, None, NoReturn]:
        return self._get_msg(
            title, text, style=MESSAGES.FAIL, show=show, spaced=spaced, exits=exits
        )

    def warn(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Optional[int] = None,
    ):
        return self._get_msg(
            title, text, style=MESSAGES.WARN, show=show, spaced=spaced, exits=exits
        )

    def info(
        self,
        title: Any = "",
        text: Any = "",
        show: bool = True,
        spaced: bool = False,
        exits: Optional[int] = None,
    ):
        return self._get_msg(
            title, text, style=MESSAGES.INFO, show=show, spaced=spaced, exits=exits
        )

    def text(
        self,
        title: Any = "",
        text: Any = "",
        color: Optional[Union[str, int]] = None,
        bg_color: Optional[Union[str, int]] = None,
        icon: Optional[str] = None,
        spaced: bool = False,
        show: bool = True,
        no_print: bool = False,
        exits: Optional[int] = None,
    ):
        if not show:
            return
        if self.pretty:
            color = self.colors.get(cast(str, color), color)
            bg_color = self.colors.get(cast(str, bg_color), bg_color)
            icon = self.icons.get(cast(str, icon))
            if icon:
                title = locale_escape("{} {}".format(icon, title)).strip()
            if self.show_color:
                title = _color(title, fg=color, bg=bg_color)
            title = wrap(title, indent=0)
        if text:
            title = "{}\n{}".format(title, wrap(text, indent=0))
        if self.timestamp:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            title = "{}\t{}".format(now, title)
        if exits is not None or spaced:
            title = "\n{}\n".format(title)
        if not self.no_print and not no_print:
            print(title)
        if exits is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            if self.no_print or no_print and exits != 0:
                try:
                    raise RuntimeError(title.strip())
                except Exception as e:
                    tb = "\n".join(traceback.format_stack()[:-3])
                    raise SystemExit("{}\n{}".format(tb, e))
            sys.exit(exits)
        if self.no_print or no_print:
            return title

    def divider(
        self,
        text: str = "",
        char: str = "=",
        show: bool = True,
        icon: Optional[str] = None,
    ):
        if icon:
            icon = self.icons.get(icon, icon)
            text = "{} {}".format(icon, text)
        if text:
            text = " {} ".format(text)
        return self.text(text.center(self.line_max, char), show=show)

    def table(
        self,
        data: Collection[Collection[Any]],
        header: Optional[Collection[Any]] = None,
        divider: bool = False,
        widths: Union[str, Collection[Union[int, str]]] = "auto",
        spacing: int = 3,
        aligns: Optional[Collection[str]] = None,
        multiline: bool = False,
        show: bool = True,
    ):
        return self.text(
            table(
                data,
                header=header,
                divider=divider,
                widths=widths,
                spacing=spacing,
                aligns=aligns,
                multiline=multiline,
            ),
            show=show,
        )

    def row(
        self,
        data: Collection[Any],
        widths: Union[str, Collection[Union[int, str]]] = "auto",
        spacing: int = 3,
        aligns: Optional[Collection[str]] = None,
        multiline: bool = False,
        show: bool = True,
    ):
        return self.text(
            row(
                data,
                widths=widths,
                spacing=spacing,
                aligns=aligns,
                multiline=multiline,
            ),
            show=show,
        )

    @contextmanager
    def loading(self, text: Any = "", spinner: bool = True):
        if self.hide_animation or not spinner or self.no_print:
            if not self.no_print:
                self.text(text)
            yield
            return

        process = Process(target=self._spinner, args=(text,))
        process.start()
        try:
            yield
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            print("\r", end="")

    def _get_msg(
        self,
        title: Any,
        text: Any,
        style: str,
        show: bool,
        spaced: bool,
        exits: Optional[Union[int, bool]],
    ):
        self._counts[style] += 1
        if style == MESSAGES.WARN and self.ignore_warnings:
            show = False
        return self.text(
            title,
            text,
            color=style,
            icon=style,
            show=show,
            spaced=spaced,
            exits=exits,
        )

    def _spinner(self, text: Any) -> None:
        for char in itertools.cycle(self.anim):
            print("\r{} {}".format(char, text), end="")
            sys.stdout.flush()
            time.sleep(0.1)