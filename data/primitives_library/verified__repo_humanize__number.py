from __future__ import annotations

import bisect

from .i18n import _gettext as _
from .i18n import _ngettext, decimal_separator, thousands_separator
from .i18n import _ngettext_noop as NS_
from .i18n import _pgettext as P_

__lazy_modules__ = {"bisect"}

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import TypeAlias

    NumberOrString: TypeAlias = float | str


_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
}
_SUPERSCRIPT_TRANS = str.maketrans(_SUPERSCRIPT_MAP)

_ORDINAL_SUFFIXES = ("th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th")
_APNUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

_METRIC_PREFIXES = {
    -8: "y",
    -7: "z",
    -6: "a",
    -5: "f",
    -4: "p",
    -3: "n",
    -2: "μ",
    -1: "m",
    0: "",
    1: "k",
    2: "M",
    3: "G",
    4: "T",
    5: "P",
    6: "E",
    7: "Z",
    8: "Y",
}


def _format_not_finite(value: float) -> str:
    import math

    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "-Inf" if value < 0 else "+Inf"
    return ""


def ordinal(value: NumberOrString, gender: str = "male") -> str:
    import math

    try:
        number = float(value)
        if not math.isfinite(number):
            return _format_not_finite(number)
        integer = int(value)
    except (TypeError, ValueError):
        return str(value)

    chosen_gender = "male" if gender == "male" else "female"
    last = 0 if integer % 100 in (11, 12, 13) else integer % 10
    ending = P_(f"{last} ({chosen_gender})", _ORDINAL_SUFFIXES[last])
    return f"{integer}{ending}"


def ordinalize(value: NumberOrString, gender: str = "male") -> str:
    return ordinal(value, gender)


def clamp(value, floor=None, ceil=None):
    if floor is not None:
        value = max(value, floor)
    if ceil is not None:
        value = min(value, ceil)
    return value


def intcomma(value: NumberOrString, ndigits: int | None = None) -> str:
    import math

    group = thousands_separator()
    point = decimal_separator()

    try:
        if isinstance(value, str):
            normalized = value.replace(group, "").replace(point, ".")
            numeric = float(normalized)
            if not math.isfinite(numeric):
                return _format_not_finite(numeric)
            value = float(normalized) if "." in normalized else int(normalized)
        else:
            numeric = float(value)
            if not math.isfinite(numeric):
                return _format_not_finite(numeric)
    except (TypeError, ValueError):
        return str(value)

    if ndigits is None:
        rendered = f"{value:,}"
    else:
        rendered = f"{value:,.{ndigits}f}"

    if group != "," or point != ".":
        rendered = rendered.translate(str.maketrans(",.", group + point))
    return rendered


powers = [10**exponent for exponent in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]
human_powers = (
    NS_("thousand", "thousand"),
    NS_("million", "million"),
    NS_("billion", "billion"),
    NS_("trillion", "trillion"),
    NS_("quadrillion", "quadrillion"),
    NS_("quintillion", "quintillion"),
    NS_("sextillion", "sextillion"),
    NS_("septillion", "septillion"),
    NS_("octillion", "octillion"),
    NS_("nonillion", "nonillion"),
    NS_("decillion", "decillion"),
    NS_("googol", "googol"),
)


def intword(value: NumberOrString, format: str = "%.1f") -> str:
    import math

    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return _format_not_finite(numeric)
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    prefix = ""
    if value < 0:
        prefix = "-"
        value = -value

    if value < powers[0]:
        return f"{prefix}{value}"

    position = bisect.bisect_right(powers, value)
    is_largest = position == len(powers)
    position -= 1

    divisor = powers[position]
    scaled = value / divisor
    displayed = float(format % scaled)

    if not is_largest and displayed * divisor == powers[position + 1]:
        position += 1
        displayed = 1.0

    singular, plural = human_powers[position]
    label = _ngettext(singular, plural, math.ceil(displayed))
    text = (format % displayed).replace(".", decimal_separator())
    return f"{prefix}{text} {label}"


def apnumber(value: NumberOrString) -> str:
    import math

    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return _format_not_finite(numeric)
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if 0 <= value < 10:
        return _(_APNUMBER_WORDS[value])
    return str(value)


def fractional(value: NumberOrString) -> str:
    import fractions
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(number):
        return _format_not_finite(number)

    fraction = fractions.Fraction(number).limit_denominator()

    if fraction.denominator == 1:
        return str(fraction.numerator)

    numerator = fraction.numerator
    denominator = fraction.denominator

    if abs(numerator) > denominator:
        whole = int(numerator / denominator)
        remainder = abs(numerator) % denominator
        return f"{whole} {remainder}/{denominator}"

    return f"{numerator}/{denominator}"


def scientific(value: NumberOrString, precision: int = 2) -> str:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(number):
        return _format_not_finite(number)

    if number == 0:
        return "0"

    exponent = int(math.floor(math.log10(abs(number))))
    significand = number / (10**exponent)
    exponent_text = str(exponent).translate(_SUPERSCRIPT_TRANS)
    return f"{significand:.{precision}f} x 10{exponent_text}"


def metric(value: NumberOrString, unit: str = "", precision: int = 1) -> str:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(number):
        return _format_not_finite(number)

    if number == 0:
        prefix_index = 0
    else:
        prefix_index = int(math.floor(math.log10(abs(number)) / 3))
        prefix_index = clamp(prefix_index, min(_METRIC_PREFIXES), max(_METRIC_PREFIXES))

    scaled = number / (1000**prefix_index)
    prefix = _METRIC_PREFIXES[prefix_index]
    return f"{scaled:.{precision}f} {prefix}{unit}"