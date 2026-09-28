"""Lexer for JSON text.

Token format returned by ``tokenize``:

    (token_type, value)

Structural token types are the character itself: "{", "}", "[", "]", ":", ",".
Other token types are "STRING", "NUMBER", "TRUE", "FALSE", and "NULL".
String values are decoded, including JSON escape sequences.
Number values are returned as ``int`` when possible and ``float`` when the
number contains a fraction or exponent.
"""

__all__ = ["tokenize"]


_HEX_DIGITS = "0123456789abcdefABCDEF"
_WHITESPACE = " \t\r\n"


def tokenize(s: str) -> list:
    """Tokenize JSON text.

    Raises:
        ValueError: if the input contains invalid JSON lexical syntax.
    """
    if not isinstance(s, str):
        raise ValueError("tokenize() expects a str")

    tokens = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch in _WHITESPACE:
            i += 1
            continue

        if ch in "{}[]:,":
            tokens.append((ch, ch))
            i += 1
            continue

        if ch == '"':
            value, i = _read_string(s, i)
            tokens.append(("STRING", value))
            continue

        if ch == "-" or ("0" <= ch <= "9"):
            value, i = _read_number(s, i)
            tokens.append(("NUMBER", value))
            continue

        if s.startswith("true", i):
            tokens.append(("TRUE", True))
            i += 4
            continue

        if s.startswith("false", i):
            tokens.append(("FALSE", False))
            i += 5
            continue

        if s.startswith("null", i):
            tokens.append(("NULL", None))
            i += 4
            continue

        _raise_at(s, i, "invalid character")

    return tokens


def _read_string(s: str, start: int) -> tuple:
    parts = []
    i = start + 1
    n = len(s)

    while i < n:
        ch = s[i]

        if ch == '"':
            return "".join(parts), i + 1

        if ord(ch) < 0x20:
            _raise_at(s, i, "unescaped control character in string")

        if ch != "\\":
            parts.append(ch)
            i += 1
            continue

        i += 1
        if i >= n:
            _raise_at(s, i - 1, "unterminated escape sequence")

        esc = s[i]

        if esc == '"':
            parts.append('"')
            i += 1
        elif esc == "\\":
            parts.append("\\")
            i += 1
        elif esc == "/":
            parts.append("/")
            i += 1
        elif esc == "b":
            parts.append("\b")
            i += 1
        elif esc == "f":
            parts.append("\f")
            i += 1
        elif esc == "n":
            parts.append("\n")
            i += 1
        elif esc == "r":
            parts.append("\r")
            i += 1
        elif esc == "t":
            parts.append("\t")
            i += 1
        elif esc == "u":
            codepoint, i = _read_unicode_escape(s, i)
            parts.append(chr(codepoint))
        else:
            _raise_at(s, i, "invalid escape sequence")

    _raise_at(s, start, "unterminated string")


def _read_unicode_escape(s: str, u_pos: int) -> tuple:
    hex_start = u_pos + 1
    hex_end = hex_start + 4

    if hex_end > len(s):
        _raise_at(s, u_pos, "incomplete unicode escape")

    hex_text = s[hex_start:hex_end]
    if not _is_hex4(hex_text):
        _raise_at(s, hex_start, "invalid unicode escape")

    first = int(hex_text, 16)
    i = hex_end

    if 0xD800 <= first <= 0xDBFF:
        if i + 6 <= len(s) and s[i] == "\\" and s[i + 1] == "u":
            low_hex = s[i + 2:i + 6]
            if _is_hex4(low_hex):
                second = int(low_hex, 16)
                if 0xDC00 <= second <= 0xDFFF:
                    combined = (
                        0x10000
                        + ((first - 0xD800) << 10)
                        + (second - 0xDC00)
                    )
                    return combined, i + 6

    return first, i


def _read_number(s: str, start: int) -> tuple:
    i = start
    n = len(s)
    is_float = False

    if s[i] == "-":
        i += 1
        if i >= n or not ("0" <= s[i] <= "9"):
            _raise_at(s, start, "invalid number")

    if s[i] == "0":
        i += 1
        if i < n and "0" <= s[i] <= "9":
            _raise_at(s, i, "leading zero in number")
    elif "1" <= s[i] <= "9":
        i += 1
        while i < n and "0" <= s[i] <= "9":
            i += 1
    else:
        _raise_at(s, start, "invalid number")

    if i < n and s[i] == ".":
        is_float = True
        i += 1
        if i >= n or not ("0" <= s[i] <= "9"):
            _raise_at(s, i - 1, "invalid fractional part in number")
        while i < n and "0" <= s[i] <= "9":
            i += 1

    if i < n and (s[i] == "e" or s[i] == "E"):
        is_float = True
        i += 1
        if i < n and (s[i] == "+" or s[i] == "-"):
            i += 1
        if i >= n or not ("0" <= s[i] <= "9"):
            _raise_at(s, i - 1, "invalid exponent in number")
        while i < n and "0" <= s[i] <= "9":
            i += 1

    text = s[start:i]
    if is_float:
        return float(text), i
    return int(text), i


def _is_hex4(text: str) -> bool:
    return len(text) == 4 and all(ch in _HEX_DIGITS for ch in text)


def _line_col(s: str, pos: int) -> tuple:
    if pos < 0:
        pos = 0
    if pos > len(s):
        pos = len(s)

    line = 1
    col = 1

    for ch in s[:pos]:
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    return line, col


def _raise_at(s: str, pos: int, message: str):
    line, col = _line_col(s, pos)
    raise ValueError(f"{message} at line {line} column {col}")