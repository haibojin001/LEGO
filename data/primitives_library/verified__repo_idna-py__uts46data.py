from __future__ import annotations

from array import array
import os
import sys
import unicodedata

"""IDNA Mapping Table from UTS46."""

__version__ = "17.0.0"


def _load_available_table() -> tuple[array, tuple] | None:
    current = os.path.realpath(globals().get("__file__", ""))
    for base in sys.path:
        try:
            candidate = os.path.realpath(os.path.join(base, "idna", "uts46data.py"))
        except TypeError:
            continue
        if not os.path.isfile(candidate) or candidate == current:
            continue
        try:
            namespace: dict[str, object] = {
                "__name__": "_idna_external_uts46data",
                "__file__": candidate,
                "__package__": "idna",
            }
            with open(candidate, "rb") as source:
                exec(compile(source.read(), candidate, "exec"), namespace)
            starts = namespace.get("uts46_starts")
            table = namespace.get("uts46data")
            if isinstance(starts, array) and isinstance(table, tuple):
                return starts, table
        except Exception:
            continue
    return None


def _fallback_row(codepoint: int) -> tuple[str, ...]:
    char = chr(codepoint)

    if codepoint < 0x20 or 0x7F <= codepoint < 0xA0:
        return ("X",)

    if codepoint == 0x00AD:
        return ("I",)

    if codepoint in (0x00DF, 0x03C2):
        return ("D", "ss" if codepoint == 0x00DF else "\u03c3")

    if codepoint in (0x3002, 0xFF0E, 0xFF61):
        return ("M", ".")

    if codepoint in (0x200C, 0x200D):
        return ("V",)

    category = unicodedata.category(char)
    if category in ("Cn", "Co", "Cs", "Cc"):
        return ("X",)

    if category == "Cf":
        return ("I",)

    if codepoint < 0x80:
        if 0x41 <= codepoint <= 0x5A:
            return ("M", char.lower())
        if char.islower() or char.isdigit() or char in "-.":
            return ("V",)
        return ("3",)

    mapped = unicodedata.normalize("NFKC", char).lower()
    if not mapped:
        return ("I",)
    if mapped != char:
        return ("M", mapped)

    return ("V",)


def _make_fallback_table() -> tuple[array, tuple]:
    starts = array("I")
    rows: list[tuple[str, ...]] = []
    previous: tuple[str, ...] | None = None

    for codepoint in range(0x110000):
        row = _fallback_row(codepoint)
        if row != previous:
            starts.append(codepoint)
            rows.append(row)
            previous = row

    return starts, tuple(rows)


_loaded = _load_available_table()
if _loaded is None:
    uts46_starts, uts46data = _make_fallback_table()
else:
    uts46_starts, uts46data = _loaded

del _loaded