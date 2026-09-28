import math

MODIFIERS = {
    'reset': (0, 0),
    'bold': (1, 22),
    'dimmed': (2, 22),
    'italic': (3, 23),
    'underlined': (4, 24),
    'blinkslow': (5, 25),
    'blinkrapid': (6, 25),
    'inversed': (7, 27),
    'concealed': (8, 28),
    'struckthrough': (9, 29)
}

MODIFIER_RESET_OFFSET = 21

FOREGROUND_COLOR_OFFSET = 30
BACKGROUND_COLOR_OFFSET = 40

COLOR_CLOSE_OFFSET = 9

CSI = '\033['

ANSI_ESCAPE_CODE = '{csi}{{code}}m'.format(csi=CSI)

NEST_PLACEHOLDER = ANSI_ESCAPE_CODE.format(code=26)


def round(value):
    lower = math.floor(value)
    if value - lower < 0.5:
        return int(lower)
    return int(math.ceil(value))


def rgb_to_ansi256(r, g, b):
    if r == g and g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round(((r - 8) / 247.0) * 24) + 232

    red = 36 * round(r / 255.0 * 5.0)
    green = 6 * round(g / 255.0 * 5.0)
    blue = round(b / 255.0 * 5.0)
    return 16 + red + green + blue


def rgb_to_ansi16(r, g, b, use_bright=False):
    blue = round(b / 255.0) << 2
    green = round(g / 255.0) << 1
    red = round(r / 255.0)
    return (90 if use_bright else 30) + (blue | green | red)