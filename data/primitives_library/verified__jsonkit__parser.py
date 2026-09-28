from jsonkit.lexer import _HEX_DIGITS

__all__ = ["loads"]

_JSON_WHITESPACE = " \t\r\n"
_JSON_HEX_DIGITS = set(_HEX_DIGITS)
_JSON_HEX_DIGITS.update("0123456789abcdefABCDEF")


def loads(s: str):
    if not isinstance(s, str):
        raise TypeError("loads() expects a str instance")

    parser = _Parser(s)
    return parser.parse()


def _line_col(s: str, pos: int):
    if pos < 0:
        pos = 0
    if pos > len(s):
        pos = len(s)

    line = s.count("\n", 0, pos) + 1
    last_newline = s.rfind("\n", 0, pos)
    if last_newline == -1:
        col = pos + 1
    else:
        col = pos - last_newline
    return line, col


class _Parser:
    __slots__ = ("s", "pos", "length")

    def __init__(self, s: str):
        self.s = s
        self.pos = 0
        self.length = len(s)

    def parse(self):
        value = self.parse_value()
        self._skip_whitespace()
        if self.pos != self.length:
            self._error("Extra data")
        return value

    def current_char(self):
        if self.pos >= self.length:
            return ""
        return self.s[self.pos]

    def parse_value(self):
        self._skip_whitespace()
        current_char = self.current_char()

        if current_char == "{":
            return self.parse_object()
        elif current_char == "[":
            return self.parse_array()
        elif current_char == '"':
            return self.parse_string()
        elif ("0" <= current_char <= "9") or current_char == "-":
            return self.parse_number()
        elif current_char == "t" or current_char == "f" or current_char == "n":
            return self.parse_boolean_or_null()
        elif current_char == "":
            self._error("Expecting value")
        else:
            self._error("Unexpected character")

    def parse_object(self):
        obj = {}
        self._expect("{")
        self._skip_whitespace()

        if self.current_char() == "}":
            self.pos += 1
            return obj

        while True:
            self._skip_whitespace()
            if self.current_char() != '"':
                self._error("Expecting property name enclosed in double quotes")

            key = self.parse_string()

            self._skip_whitespace()
            self._expect(":")

            value = self.parse_value()
            obj[key] = value

            self._skip_whitespace()
            ch = self.current_char()

            if ch == ",":
                self.pos += 1
                continue
            if ch == "}":
                self.pos += 1
                return obj

            self._error("Expecting ',' delimiter or '}'")

    def parse_array(self):
        arr = []
        self._expect("[")
        self._skip_whitespace()

        if self.current_char() == "]":
            self.pos += 1
            return arr

        while True:
            arr.append(self.parse_value())

            self._skip_whitespace()
            ch = self.current_char()

            if ch == ",":
                self.pos += 1
                continue
            if ch == "]":
                self.pos += 1
                return arr

            self._error("Expecting ',' delimiter or ']'")

    def parse_string(self):
        if self.current_char() != '"':
            self._error("Expecting string")

        self.pos += 1
        start = self.pos
        i = self.pos
        out = []
        s = self.s
        n = self.length

        while i < n:
            ch = s[i]

            if ch == '"':
                out.append(s[start:i])
                self.pos = i + 1
                return "".join(out)

            if ch == "\\":
                out.append(s[start:i])
                i += 1

                if i >= n:
                    self.pos = i
                    self._error("Unterminated string escape")

                esc = s[i]

                if esc == '"':
                    out.append('"')
                    i += 1
                elif esc == "\\":
                    out.append("\\")
                    i += 1
                elif esc == "/":
                    out.append("/")
                    i += 1
                elif esc == "b":
                    out.append("\b")
                    i += 1
                elif esc == "f":
                    out.append("\f")
                    i += 1
                elif esc == "n":
                    out.append("\n")
                    i += 1
                elif esc == "r":
                    out.append("\r")
                    i += 1
                elif esc == "t":
                    out.append("\t")
                    i += 1
                elif esc == "u":
                    codepoint, i = self._read_unicode_escape(i + 1)

                    if 0xD800 <= codepoint <= 0xDBFF:
                        if i + 1 < n and s[i] == "\\" and s[i + 1] == "u":
                            low = self._peek_unicode_escape(i + 2)
                            if low is not None and 0xDC00 <= low <= 0xDFFF:
                                i += 6
                                codepoint = (
                                    0x10000
                                    + ((codepoint - 0xD800) << 10)
                                    + (low - 0xDC00)
                                )

                    out.append(chr(codepoint))
                else:
                    self.pos = i
                    self._error("Invalid escape sequence")

                start = i
                continue

            if ord(ch) < 0x20:
                self.pos = i
                self._error("Invalid control character in string")

            i += 1

        self.pos = n
        self._error("Unterminated string")

    def parse_number(self):
        s = self.s
        n = self.length
        start = self.pos

        if self.current_char() == "-":
            self.pos += 1
            if self.pos >= n:
                self._error("Invalid number")

        ch = self.current_char()

        if ch == "0":
            self.pos += 1
            if self.pos < n and "0" <= s[self.pos] <= "9":
                self._error("Invalid number with leading zero")
        elif "1" <= ch <= "9":
            self.pos += 1
            while self.pos < n and "0" <= s[self.pos] <= "9":
                self.pos += 1
        else:
            self._error("Invalid number")

        is_float = False

        if self.pos < n and s[self.pos] == ".":
            is_float = True
            self.pos += 1

            if self.pos >= n or not ("0" <= s[self.pos] <= "9"):
                self._error("Invalid number")

            while self.pos < n and "0" <= s[self.pos] <= "9":
                self.pos += 1

        if self.pos < n and (s[self.pos] == "e" or s[self.pos] == "E"):
            is_float = True
            self.pos += 1

            if self.pos < n and (s[self.pos] == "+" or s[self.pos] == "-"):
                self.pos += 1

            if self.pos >= n or not ("0" <= s[self.pos] <= "9"):
                self._error("Invalid number")

            while self.pos < n and "0" <= s[self.pos] <= "9":
                self.pos += 1

        text = s[start:self.pos]
        try:
            if is_float:
                return float(text)
            return int(text)
        except Exception as exc:
            raise ValueError("Invalid number") from exc

    def parse_boolean_or_null(self):
        if self.s.startswith("true", self.pos):
            self.pos += 4
            return True
        if self.s.startswith("false", self.pos):
            self.pos += 5
            return False
        if self.s.startswith("null", self.pos):
            self.pos += 4
            return None
        self._error("Expecting value")

    def _skip_whitespace(self):
        while self.pos < self.length and self.s[self.pos] in _JSON_WHITESPACE:
            self.pos += 1

    def _expect(self, char: str):
        if self.current_char() != char:
            self._error("Expecting " + repr(char))
        self.pos += 1

    def _read_unicode_escape(self, pos: int):
        if pos + 4 > self.length:
            self.pos = pos
            self._error("Invalid unicode escape")

        for offset in range(4):
            if self.s[pos + offset] not in _JSON_HEX_DIGITS:
                self.pos = pos + offset
                self._error("Invalid unicode escape")

        return int(self.s[pos:pos + 4], 16), pos + 4

    def _peek_unicode_escape(self, pos: int):
        if pos + 4 > self.length:
            return None

        for offset in range(4):
            if self.s[pos + offset] not in _JSON_HEX_DIGITS:
                return None

        return int(self.s[pos:pos + 4], 16)

    def _error(self, message: str):
        line, col = _line_col(self.s, self.pos)
        raise ValueError(f"{message} at line {line} column {col} (char {self.pos})")