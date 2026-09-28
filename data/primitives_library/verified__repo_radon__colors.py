"""Terminal color constants and color-selection helpers."""

import os
import sys


def color_enabled():
    mode = os.environ.get("COLOR", "auto")
    return mode == "yes" or (mode == "auto" and sys.stdout.isatty())


try:
    import colorama as _colorama

    _colorama.init(strip=not color_enabled())

    GREEN = _colorama.Fore.GREEN
    YELLOW = _colorama.Fore.YELLOW
    RED = _colorama.Fore.RED
    MAGENTA = _colorama.Fore.MAGENTA
    CYAN = _colorama.Fore.CYAN
    WHITE = _colorama.Fore.WHITE
    BRIGHT = _colorama.Style.BRIGHT
    RESET = _colorama.Style.RESET_ALL
except ImportError:
    GREEN = ""
    YELLOW = ""
    RED = ""
    MAGENTA = ""
    CYAN = ""
    WHITE = ""
    BRIGHT = ""
    RESET = ""


RANKS_COLORS = {
    "A": GREEN,
    "B": GREEN,
    "C": YELLOW,
    "D": YELLOW,
    "E": RED,
    "F": RED,
}

LETTERS_COLORS = {
    "F": MAGENTA,
    "C": CYAN,
    "M": WHITE,
}

MI_RANKS = {
    "A": GREEN,
    "B": YELLOW,
    "C": RED,
}

TEMPLATE = "{0}{1} {reset}{2}:{3} {4} - {5}{6}{reset}"