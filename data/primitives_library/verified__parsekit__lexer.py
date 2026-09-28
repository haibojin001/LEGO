class Token:
    """A lexical token for arithmetic expressions."""

    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        if kind not in ("NUM", "OP", "LPAREN", "RPAREN"):
            raise ValueError("invalid token kind: %r" % (kind,))
        self.kind = kind
        self.value = value

    def __repr__(self):
        return "Token(%r, %r)" % (self.kind, self.value)

    def __eq__(self, other):
        return (
            isinstance(other, Token)
            and self.kind == other.kind
            and self.value == other.value
        )


def tokenize(s: str) -> list:
    """
    Tokenize an arithmetic expression.

    Recognized tokens:
      - numbers: integers and floats, including forms like 12, 12.3, .3, 12.,
        1e3, 1.2e-3
      - operators: + - * / ^
      - parentheses: ( )

    Whitespace is skipped. Unknown or malformed characters raise ValueError.
    """
    tokens = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        if ch in "+-*/^":
            tokens.append(Token("OP", ch))
            i += 1
            continue

        if ch == "(":
            tokens.append(Token("LPAREN", ch))
            i += 1
            continue

        if ch == ")":
            tokens.append(Token("RPAREN", ch))
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
            start = i
            saw_dot = False
            saw_exp = False

            if ch == ".":
                saw_dot = True
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
            else:
                while i < n and s[i].isdigit():
                    i += 1

                if i < n and s[i] == ".":
                    saw_dot = True
                    i += 1
                    while i < n and s[i].isdigit():
                        i += 1

            if i < n and s[i] in "eE":
                saw_exp = True
                i += 1

                if i < n and s[i] in "+-":
                    i += 1

                exp_start = i
                while i < n and s[i].isdigit():
                    i += 1

                if i == exp_start:
                    raise ValueError("malformed number at position %d" % start)

            if i < n and s[i] == ".":
                raise ValueError("malformed number at position %d" % start)

            raw = s[start:i]
            try:
                value = float(raw) if (saw_dot or saw_exp) else int(raw)
            except ValueError:
                raise ValueError("malformed number at position %d" % start)

            tokens.append(Token("NUM", value))
            continue

        raise ValueError("unknown character %r at position %d" % (ch, i))

    return tokens


__all__ = ["Token", "tokenize"]