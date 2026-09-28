"""Bits and bytes related humanization."""

from __future__ import annotations

from math import log

from humanize.i18n import _gettext as _

__lazy_modules__ = {"humanize.i18n", "math"}

suffixes = {
    "decimal": ("kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"),
    "binary": ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB"),
    "gnu": "KMGTPEZYRQ",
}


def naturalsize(
    value: float | str,
    binary: bool = False,
    gnu: bool = False,
    format: str = "%.1f",
) -> str:
    """Return a human-readable representation of a byte quantity."""
    if gnu:
        labels = suffixes["gnu"]
    elif binary:
        labels = suffixes["binary"]
    else:
        labels = suffixes["decimal"]

    divisor = 1024 if gnu or binary else 1000
    number = float(value)
    absolute = abs(number)

    if absolute == 1 and not gnu:
        return _("%d Byte") % int(number)

    if absolute < divisor:
        if gnu:
            return f"{int(number)}B"
        return _("%d Bytes") % int(number)

    power = int(min(log(absolute, divisor), len(labels)))
    displayed = format % (absolute / divisor**power)

    if power < len(labels) and abs(float(displayed)) >= divisor:
        power += 1

    joiner = "" if gnu else " "
    return format % (number / divisor**power) + joiner + _(labels[power - 1])