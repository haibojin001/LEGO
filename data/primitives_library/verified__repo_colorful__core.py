import os

from . import ansi
from . import colors
from . import styles
from . import terminal


DEFAULT_RGB_TXT_PATH = os.environ.get(
    'COLORFUL_DEFAULT_COLOR_PALETTE',
    os.path.join(os.path.dirname(__file__), 'data', 'rgb.txt')
)

COLOR_PALETTE = colors.parse_colors(path=DEFAULT_RGB_TXT_PATH)

COLORNAMES_COLORS_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'colornames.json'
)


class ColorfulError(Exception):
    pass


class ColorfulAttributeError(AttributeError, ColorfulError):
    pass


def translate_rgb_to_ansi_code(red, green, blue, offset, colormode):
    if colormode == terminal.NO_COLORS:
        return '', ''

    close_code = ansi.ANSI_ESCAPE_CODE.format(
        code=offset + ansi.COLOR_CLOSE_OFFSET
    )

    if colormode in (terminal.ANSI_8_COLORS, terminal.ANSI_16_COLORS):
        converted = ansi.rgb_to_ansi16(red, green, blue)
        open_code = ansi.ANSI_ESCAPE_CODE.format(
            code=converted + offset - ansi.FOREGROUND_COLOR_OFFSET
        )
        return open_code, close_code

    if colormode == terminal.ANSI_256_COLORS:
        converted = ansi.rgb_to_ansi256(red, green, blue)
        open_code = ansi.ANSI_ESCAPE_CODE.format(
            code='{};5;{}'.format(8 + offset, converted)
        )
        return open_code, close_code

    if colormode == terminal.TRUE_COLORS:
        open_code = ansi.ANSI_ESCAPE_CODE.format(
            code='{};2;{};{};{}'.format(8 + offset, red, green, blue)
        )
        return open_code, close_code

    raise ColorfulAttributeError(
        'invalid color mode "{}"'.format(colormode)
    )


def translate_colorname_to_ansi_code(colorname, offset, colormode, colorpalette):
    try:
        red, green, blue = colorpalette[colorname]
    except KeyError:
        raise ColorfulAttributeError(
            'the color "{}" is unknown. Use a color in your color palette '
            '(by default: X11 rgb.txt)'.format(colorname)
        )

    return translate_rgb_to_ansi_code(
        red, green, blue, offset, colormode
    )


def resolve_modifier_to_ansi_code(modifiername, colormode):
    if colormode == terminal.NO_COLORS:
        return '', ''

    try:
        opening, closing = ansi.MODIFIERS[modifiername]
    except KeyError:
        raise ColorfulAttributeError(
            'the modifier "{}" is unknown. Use one of: {}'.format(
                modifiername, ansi.MODIFIERS.keys()
            )
        )

    return (
        ansi.ANSI_ESCAPE_CODE.format(code=opening),
        ansi.ANSI_ESCAPE_CODE.format(code=closing)
    )


def translate_style(style, colormode, colorpalette):
    parts = iter(style.split('_'))
    starts = []
    ends = []

    try:
        current = None

        for candidate in parts:
            current = candidate
            if current not in ansi.MODIFIERS:
                break

            opening, closing = resolve_modifier_to_ansi_code(
                current, colormode
            )
            starts.append(opening)
            ends.append(closing)
        else:
            raise StopIteration()

        if current != 'on':
            opening, closing = translate_colorname_to_ansi_code(
                current,
                ansi.FOREGROUND_COLOR_OFFSET,
                colormode,
                colorpalette
            )
            starts.append(opening)
            ends.append(closing)
            next(parts)

        current = next(parts)
        opening, closing = translate_colorname_to_ansi_code(
            current,
            ansi.BACKGROUND_COLOR_OFFSET,
            colormode,
            colorpalette
        )
        starts.append(opening)
        ends.append(closing)
    except StopIteration:
        pass

    return ''.join(starts), ''.join(ends)


def style_string(string, ansi_style, colormode, nested=False):
    opening, closing = ansi_style
    content = str(string).replace(ansi.NEST_PLACEHOLDER, opening)

    return '{}{}{}{}'.format(
        opening,
        content,
        closing,
        ansi.NEST_PLACEHOLDER if nested else ''
    )


class ColorfulString:
    def __init__(self, orig_string, styled_string, colorful_ctx):
        self.orig_string = str(orig_string)
        self.styled_string = str(styled_string)
        self.colorful_ctx = colorful_ctx

    def __str__(self):
        if self.colorful_ctx.colormode == terminal.NO_COLORS:
            return self.orig_string
        return self.styled_string

    def __repr__(self):
        return repr(str(self))

    def __len__(self):
        return len(self.orig_string)

    def __iter__(self):
        return iter(self.styled_string)

    def __add__(self, other):
        if isinstance(other, ColorfulString):
            return ColorfulString(
                self.orig_string + other.orig_string,
                self.styled_string + other.styled_string,
                self.colorful_ctx
            )

        return ColorfulString(
            self.orig_string + other,
            self.styled_string + other,
            self.colorful_ctx
        )

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        if isinstance(other, ColorfulString):
            return ColorfulString(
                other.orig_string + self.orig_string,
                other.styled_string + self.styled_string,
                self.colorful_ctx
            )

        return ColorfulString(
            other + self.orig_string,
            other + self.styled_string,
            self.colorful_ctx
        )

    def __eq__(self, other):
        if isinstance(other, ColorfulString):
            return self.orig_string == other.orig_string
        return self.orig_string == other

    def __ne__(self, other):
        return not self == other

    def __contains__(self, item):
        return item in self.orig_string

    def __getitem__(self, item):
        return self.orig_string[item]

    def __format__(self, format_spec):
        return format(str(self), format_spec)


class Colorful:
    def __init__(self, colorpalette=None, colormode=None):
        if colorpalette is None:
            colorpalette = COLOR_PALETTE

        if colormode is None:
            colormode = terminal.detect_color_support(os.environ)

        self.colorpalette = colorpalette
        self.colormode = colormode

    def __getattr__(self, style):
        ansi_style = translate_style(
            style,
            self.colormode,
            self.colorpalette
        )

        def apply_style(string):
            return ColorfulString(
                string,
                style_string(
                    string,
                    ansi_style,
                    self.colormode,
                    nested=True
                ),
                self
            )

        return apply_style