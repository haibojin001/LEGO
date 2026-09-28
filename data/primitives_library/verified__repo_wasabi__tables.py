import os
from itertools import zip_longest
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Union, cast

from .compat import Literal
from .util import COLORS
from .util import color as _color
from .util import supports_ansi

ALIGN_MAP = {"l": "<", "r": ">", "c": "^"}


def table(
    data: Union[Collection, Dict],
    header: Optional[Iterable] = None,
    footer: Optional[Iterable] = None,
    divider: bool = False,
    widths: Union[Iterable[int], Literal["auto"]] = "auto",
    max_col: int = 30,
    spacing: int = 3,
    aligns: Optional[
        Union[Iterable[Literal["r", "c", "l"]], Literal["r", "c", "l"]]
    ] = None,
    multiline: bool = False,
    env_prefix: str = "WASABI",
    color_values: Optional[Dict] = None,
    fg_colors: Optional[Iterable] = None,
    bg_colors: Optional[Iterable] = None,
) -> str:
    """Format tabular data."""
    if fg_colors is not None or bg_colors is not None:
        colors = dict(COLORS)
        if color_values is not None:
            colors.update(color_values)
        if fg_colors is not None:
            fg_colors = [colors.get(value, value) for value in fg_colors]
        if bg_colors is not None:
            bg_colors = [colors.get(value, value) for value in bg_colors]

    if isinstance(data, dict):
        data = list(data.items())

    if multiline:
        expanded = []
        for index, item in enumerate(data):
            values = [
                value if isinstance(value, (list, tuple)) else [value]
                for value in item
            ]
            expanded.extend(list(zip_longest(*values, fillvalue="")))
            if index < len(data) - 1:
                expanded.append(tuple("" for _ in item))
        data = expanded

    if widths == "auto":
        widths = _get_max_widths(data, header, footer, max_col)

    settings = {
        "widths": widths,
        "spacing": spacing,
        "aligns": aligns,
        "env_prefix": env_prefix,
        "fg_colors": fg_colors,
        "bg_colors": bg_colors,
    }

    separator = row(["-" * width for width in widths], **settings)
    output = []

    if header:
        output.append(row(header, **settings))
        if divider:
            output.append(separator)

    for item in data:
        output.append(row(item, **settings))

    if footer:
        if divider:
            output.append(separator)
        output.append(row(footer, **settings))

    return "\n{}\n".format("\n".join(output))


def row(
    data: Collection,
    widths: Union[Sequence[int], int, Literal["auto"]] = "auto",
    spacing: int = 3,
    aligns: Optional[Union[Sequence[Literal["r", "c", "l"]], str]] = None,
    env_prefix: str = "WASABI",
    fg_colors: Optional[Sequence] = None,
    bg_colors: Optional[Sequence] = None,
) -> str:
    """Format data as a table row."""
    log_friendly = os.getenv("{}_LOG_FRIENDLY".format(env_prefix), False)
    use_colors = (
        supports_ansi()
        and not log_friendly
        and (fg_colors is not None or bg_colors is not None)
    )

    columns: List[str] = []
    resolved_aligns = (
        [aligns for _ in data] if isinstance(aligns, str) else cast(List[str], aligns)
    )

    if not hasattr(widths, "__iter__") and widths != "auto":
        widths = cast(List[int], [widths for _ in range(len(data))])

    for index, value in enumerate(data):
        alignment = ALIGN_MAP.get(
            resolved_aligns[index]
            if resolved_aligns and index < len(resolved_aligns)
            else "l"
        )
        width = len(value) if widths == "auto" else cast(List[int], widths)[index]
        template = "{:%s%d}" % (alignment, width)
        value = template.format(str(value))

        if use_colors:
            foreground = fg_colors[index] if fg_colors is not None else None
            background = bg_colors[index] if bg_colors is not None else None
            value = _color(value, fg=foreground, bg=background)

        columns.append(value)

    return (" " * spacing).join(columns)


def _get_max_widths(data, header, footer, max_col):
    values = list(data)
    if header:
        values.append(header)
    if footer:
        values.append(footer)
    lengths = [[len(str(column)) for column in item] for item in values]
    return [min(max(column), max_col) for column in list(zip(*lengths))]