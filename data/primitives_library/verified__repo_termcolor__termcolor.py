from __future__ import annotations

import os
import sys
from functools import cache

TYPE_CHECKING = False

ATTRIBUTES: dict[str, int] = {
    "bold": 1,
    "dark": 2,
    "italic": 3,
    "underline": 4,
    "blink": 5,
    "reverse": 7,
    "concealed": 8,
    "strike": 9,
}

HIGHLIGHTS: dict[str, int] = {
    "on_black": 40,
    "on_grey": 40,
    "on_red": 41,
    "on_green": 42,
    "on_yellow": 43,
    "on_blue": 44,
    "on_magenta": 45,
    "on_cyan": 46,
    "on_light_grey": 47,
    "on_dark_grey": 100,
    "on_light_red": 101,
    "on_light_green": 102,
    "on_light_yellow": 103,
    "on_light_blue": 104,
    "on_light_magenta": 105,
    "on_light_cyan": 106,
    "on_white": 107,
}

COLORS: dict[str, int] = {
    "black": 30,
    "grey": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "light_grey": 37,
    "dark_grey": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_magenta": 95,
    "light_cyan": 96,
    "white": 97,
}

RESET = "\033[0m"


@cache
def can_colorize(
    *, no_color: bool | None = None, force_color: bool | None = None
) -> bool:
    if no_color is not None and no_color:
        return False
    if force_color is not None and force_color:
        return True

    if os.environ.get("ANSI_COLORS_DISABLED"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True

    if os.environ.get("TERM") == "dumb":
        return False
    if not hasattr(sys.stdout, "fileno"):
        return False

    try:
        return os.isatty(sys.stdout.fileno())
    except OSError:
        return sys.stdout.isatty()


def _check_rgb(rgb: tuple[int, int, int]) -> None:
    valid = len(rgb) == 3 and all(0 <= component <= 255 for component in rgb)
    if not valid:
        raise ValueError(f"Expected a tuple of 3 ints in range 0-255, got {rgb!r}")


def colored(
    text: object,
    color: str | tuple[int, int, int] | None = None,
    on_color: str | tuple[int, int, int] | None = None,
    attrs=None,
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
) -> str:
    result = str(text)

    if not can_colorize(no_color=no_color, force_color=force_color):
        return result

    standard = "\033[%dm%s"
    foreground_rgb = "\033[38;2;%d;%d;%dm%s"
    background_rgb = "\033[48;2;%d;%d;%dm%s"

    if color is not None:
        if isinstance(color, str):
            result = standard % (COLORS[color], result)
        elif isinstance(color, tuple):
            _check_rgb(color)
            result = foreground_rgb % (*color, result)

    if on_color is not None:
        if isinstance(on_color, str):
            result = standard % (HIGHLIGHTS[on_color], result)
        elif isinstance(on_color, tuple):
            _check_rgb(on_color)
            result = background_rgb % (*on_color, result)

    if attrs is not None:
        for attribute in attrs:
            result = standard % (ATTRIBUTES[attribute], result)

    return result + RESET


def cprint(
    text: object,
    color: str | tuple[int, int, int] | None = None,
    on_color: str | tuple[int, int, int] | None = None,
    attrs=None,
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
    **kwargs,
) -> None:
    print(
        colored(
            text,
            color,
            on_color,
            attrs,
            no_color=no_color,
            force_color=force_color,
        ),
        **kwargs,
    )