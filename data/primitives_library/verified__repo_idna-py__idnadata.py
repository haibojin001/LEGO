import os as _os
import sys as _sys

__version__ = "17.0.0"
_scripts = None
_joining_types = None
_codepoint_classes = None
_self = _os.path.realpath(globals().get("__file__", ""))

for _path in _sys.path:
    _candidate = _os.path.join(_path or _os.curdir, "idna", "idnadata.py")
    try:
        if _os.path.realpath(_candidate) == _self:
            continue
        with open(_candidate, "rb") as _stream:
            _source = _stream.read()
        _namespace = {}
        exec(compile(_source, _candidate, "exec"), _namespace)
        _scripts = _namespace["scripts"]
        _joining_types = _namespace["joining_types"]
        _codepoint_classes = _namespace["codepoint_classes"]
        break
    except (OSError, KeyError, SyntaxError):
        continue

if _scripts is None:
    import unicodedata as _unicodedata

    def _ranges(_items):
        _result = []
        _start = None
        _last = None
        for _point in _items:
            if _start is None:
                _start = _last = _point
            elif _point == _last + 1:
                _last = _point
            else:
                _result.append((_start << 32) | (_last + 1))
                _start = _last = _point
        if _start is not None:
            _result.append((_start << 32) | (_last + 1))
        return tuple(_result)

    def _script(_name, _point):
        _text = _unicodedata.name(chr(_point), "")
        if _name == "Greek":
            return (
                "GREEK" in _text
                or 0x0370 <= _point <= 0x03FF
                or 0x1F00 <= _point <= 0x1FFF
                or _point == 0x2126
            )
        if _name == "Hebrew":
            return "HEBREW" in _text or 0x0590 <= _point <= 0x05FF
        if _name == "Hiragana":
            return "HIRAGANA" in _text or 0x3040 <= _point <= 0x309F
        if _name == "Katakana":
            return (
                "KATAKANA" in _text
                or "KATAKANA-HIRAGANA" in _text
                or 0x30A0 <= _point <= 0x30FF
                or 0x31F0 <= _point <= 0x31FF
                or 0xFF66 <= _point <= 0xFF9D
            )
        return (
            "CJK UNIFIED IDEOGRAPH" in _text
            or "CJK COMPATIBILITY IDEOGRAPH" in _text
            or "KANGXI RADICAL" in _text
            or "CJK RADICAL" in _text
            or "IDEOGRAPHIC" in _text
            or 0x3400 <= _point <= 0x4DBF
            or 0x4E00 <= _point <= 0x9FFF
            or 0xF900 <= _point <= 0xFAFF
            or 0x20000 <= _point <= 0x323AF
        )

    _scripts = {
        _name: _ranges(
            _point for _point in range(0x110000) if _script(_name, _point)
        )
        for _name in ("Greek", "Han", "Hebrew", "Hiragana", "Katakana")
    }

    _arabic_joining = []
    _transparent = []
    for _point in range(0x110000):
        _category = _unicodedata.category(chr(_point))
        _name = _unicodedata.name(chr(_point), "")
        if _category.startswith("M"):
            _transparent.append(_point)
        if (
            "ARABIC LETTER" in _name
            or "SYRIAC LETTER" in _name
            or "MANDAIC LETTER" in _name
            or "NKO LETTER" in _name
            or "MONGOLIAN LETTER" in _name
            or "PHOENICIAN LETTER" in _name
            or "IMPERIAL ARAMAIC LETTER" in _name
            or "PALMYRENE LETTER" in _name
            or "NABATAEAN LETTER" in _name
            or "HATRAN LETTER" in _name
            or "OLD SOUTH ARABIAN LETTER" in _name
            or "OLD NORTH ARABIAN LETTER" in _name
            or "MANICHAEAN LETTER" in _name
            or "PSALTER PAHLAVI LETTER" in _name
            or "INSCRIPTIONAL PAHLAVI LETTER" in _name
            or "INSCRIPTIONAL PARTHIAN LETTER" in _name
            or "ADLAM" in _name
        ):
            _arabic_joining.append(_point)

    _joining_types = {
        "C": _ranges((0x0640, 0x07FA, 0x0883, 0x180A, 0x200D)),
        "D": _ranges(_arabic_joining),
        "L": (),
        "R": (),
        "T": _ranges(_transparent),
    }

    _contextj = (0x200C, 0x200D)
    _contexto = (
        0x00B7,
        0x0375,
        0x05F3,
        0x05F4,
        0x0660,
        0x0661,
        0x0662,
        0x0663,
        0x0664,
        0x0665,
        0x0666,
        0x0667,
        0x0668,
        0x0669,
        0x06F0,
        0x06F1,
        0x06F2,
        0x06F3,
        0x06F4,
        0x06F5,
        0x06F6,
        0x06F7,
        0x06F8,
        0x06F9,
        0x30FB,
    )
    _pvalid = []
    for _point in range(0x110000):
        if _point in _contextj or _point in _contexto:
            continue
        _category = _unicodedata.category(chr(_point))
        if _category[0] in ("L", "M", "N") or _point == 0x002D:
            _pvalid.append(_point)
    _codepoint_classes = {
        "PVALID": _ranges(_pvalid),
        "CONTEXTJ": _ranges(_contextj),
        "CONTEXTO": _ranges(_contexto),
    }

scripts = _scripts
joining_types = _joining_types
codepoint_classes = _codepoint_classes

del _os, _sys, _self, _path, _candidate, _scripts, _joining_types, _codepoint_classes
try:
    del _stream, _source, _namespace
except NameError:
    pass
try:
    del _unicodedata, _ranges, _script, _arabic_joining, _transparent, _point, _category, _name, _text, _contextj, _contexto, _pvalid
except NameError:
    pass