import sys

NO_COLORS = 0
ANSI_8_COLORS = 8
ANSI_16_COLORS = 16
ANSI_256_COLORS = 256
TRUE_COLORS = 0xFFFFFF


def detect_color_support(env):
    forced_modes = (
        ("COLORFUL_DISABLE", NO_COLORS),
        ("COLORFUL_FORCE_8_COLORS", ANSI_8_COLORS),
        ("COLORFUL_FORCE_16_COLORS", ANSI_16_COLORS),
        ("COLORFUL_FORCE_256_COLORS", ANSI_256_COLORS),
        ("COLORFUL_FORCE_TRUE_COLORS", TRUE_COLORS),
    )

    for variable, mode in forced_modes:
        if env.get(variable, "0") == "1":
            return mode

    output = sys.stdout
    if hasattr(output, "isatty") and not output.isatty():
        return NO_COLORS

    colorterm = env.get("COLORTERM")
    if colorterm == "truecolor" or colorterm == "24bit":
        return TRUE_COLORS
    if colorterm == "8bit":
        return ANSI_256_COLORS

    terminal_program = env.get("TERM_PROGRAM")
    if terminal_program == "iTerm.app" or terminal_program == "Hyper":
        return TRUE_COLORS
    if terminal_program == "Apple_Terminal":
        return ANSI_256_COLORS

    terminal = env.get("TERM")
    if terminal in (
        "screen-256",
        "screen-256color",
        "xterm-256",
        "xterm-256color",
    ):
        return ANSI_256_COLORS

    if terminal in (
        "screen",
        "xterm",
        "vt100",
        "color",
        "ansi",
        "cygwin",
        "linux",
    ):
        return ANSI_16_COLORS

    if colorterm:
        return ANSI_16_COLORS

    return ANSI_8_COLORS