import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union

STDOUT_ENCODING = sys.stdout.encoding if hasattr(sys.stdout, "encoding") else None
ENCODING = STDOUT_ENCODING or "ascii"
NO_UTF8 = ENCODING.lower() not in ("utf8", "utf-8")

ENV_ANSI_DISABLED = "ANSI_COLORS_DISABLED"


class MESSAGES(object):
    GOOD = "good"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


COLORS = {
    MESSAGES.GOOD: 2,
    MESSAGES.FAIL: 1,
    MESSAGES.WARN: 3,
    MESSAGES.INFO: 4,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "pink": 5,
    "cyan": 6,
    "white": 7,
    "grey": 8,
    "black": 16,
}

ICONS = {
    MESSAGES.GOOD: "✔" if not NO_UTF8 else "[+]",
    MESSAGES.FAIL: "✘" if not NO_UTF8 else "[x]",
    MESSAGES.WARN: "⚠" if not NO_UTF8 else "[!]",
    MESSAGES.INFO: "ℹ" if not NO_UTF8 else "[i]",
}

INSERT_SYMBOL = "+"
DELETE_SYMBOL = "-"


def color(
    text: str,
    fg: Optional[Union[str, int]] = None,
    bg: Optional[Union[str, int]] = None,
    bold: bool = False,
    underline: bool = False,
) -> str:
    fg = COLORS.get(fg, fg)
    bg = COLORS.get(bg, bg)
    if not any([fg, bg, bold]):
        return text
    styles = []
    if bold:
        styles.append("1")
    if underline:
        styles.append("4")
    if fg:
        styles.append("38;5;{}".format(fg))
    if bg:
        styles.append("48;5;{}".format(bg))
    return "\x1b[{}m{}\x1b[0m".format(";".join(styles), text)


def wrap(text: Any, wrap_max: int = 80, indent: int = 4) -> str:
    padding = " " * indent
    width = wrap_max - len(padding)
    return textwrap.fill(
        str(text),
        width=width,
        initial_indent=padding,
        subsequent_indent=padding,
        break_long_words=False,
        break_on_hyphens=False,
    )


def format_repr(obj: Any, max_len: int = 50, ellipsis: str = "...") -> str:
    value = repr(obj)
    if len(value) >= max_len:
        half = int(max_len / 2)
        return "{} {} {}".format(value[:half], ellipsis, value[-half:])
    return value


def diff_strings(
    a: str,
    b: str,
    fg: Union[str, int] = "black",
    bg: Union[Tuple[str, str], Tuple[int, int]] = ("green", "red"),
    add_symbols: bool = False,
) -> str:
    before = a.split("\n")
    after = b.split("\n")
    result = []
    matcher = difflib.SequenceMatcher(None, before, after)

    for operation, start_a, end_a, start_b, end_b in matcher.get_opcodes():
        if operation == "equal":
            result.extend(before[start_a:end_a])

        if operation == "insert" or operation == "replace":
            for line in after[start_b:end_b]:
                if add_symbols:
                    line = "{} {}".format(INSERT_SYMBOL, line)
                result.append(color(line, fg=fg, bg=bg[0]))

        if operation == "delete" or operation == "replace":
            for line in before[start_a:end_a]:
                if add_symbols:
                    line = "{} {}".format(DELETE_SYMBOL, line)
                result.append(color(line, fg=fg, bg=bg[1]))

    return "\n".join(result)


def get_raw_input(
    description: str, default: Optional[Union[str, bool]] = False, indent: int = 4
) -> str:
    suffix = " (default: {})".format(default) if default else ""
    prompt = wrap("{}{}: ".format(description, suffix), indent=indent)
    return input(prompt)


def locale_escape(string: Any, errors: str = "replace") -> str:
    return str(string).encode(ENCODING, errors).decode("utf8")


def can_render(string: str) -> bool:
    try:
        string.encode(ENCODING)
        return True
    except UnicodeEncodeError:
        return False


def supports_ansi() -> bool:
    if os.getenv(ENV_ANSI_DISABLED):
        return False
    try:
        from colorama import just_fix_windows_console
    except ImportError:
        if sys.platform == "win32" and "ANSICON" not in os.environ:
            return False
    else:
        just_fix_windows_console()
    return True