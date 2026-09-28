from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

from parsekit.lexer import tokenize
from parsekit.ast_nodes import BinOp, Num, UnaryOp

__all__ = ["parse"]


_NUMBER_RE = re.compile(
    r"""
    (?:
        (?:
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?:[eE][+-]?\d+)?
    )
    \Z
    """,
    re.VERBOSE,
)

_KIND_MAP = {
    "PLUS": "+",
    "ADD": "+",
    "MINUS": "-",
    "SUB": "-",
    "SUBTRACT": "-",
    "DASH": "-",
    "STAR": "*",
    "ASTERISK": "*",
    "TIMES": "*",
    "MUL": "*",
    "MULTIPLY": "*",
    "SLASH": "/",
    "DIV": "/",
    "DIVIDE": "/",
    "CARET": "^",
    "CIRCUMFLEX": "^",
    "POW": "^",
    "POWER": "^",
    "EXP": "^",
    "EXPONENT": "^",
    "LPAREN": "(",
    "L_PAREN": "(",
    "LEFT_PAREN": "(",
    "OPEN_PAREN": "(",
    "LPAR": "(",
    "RPAREN": ")",
    "R_PAREN": ")",
    "RIGHT_PAREN": ")",
    "CLOSE_PAREN": ")",
    "RPAR": ")",
    "NUMBER": "NUMBER",
    "NUM": "NUMBER",
    "INTEGER": "NUMBER",
    "INT": "NUMBER",
    "FLOAT": "NUMBER",
    "DECIMAL": "NUMBER",
    "EOF": "EOF",
    "END": "EOF",
    "ENDMARKER": "EOF",
    "END_MARKER": "EOF",
}

_IGNORED_KINDS = {
    "WS",
    "SPACE",
    "SPACES",
    "WHITESPACE",
    "NEWLINE",
    "NL",
}


def parse(s: str):
    """Parse an arithmetic expression and return an AST."""
    try:
        tokens = list(tokenize(s))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    parser = _Parser(tokens)
    result = parser.parse()
    return result


class _Parser:
    def __init__(self, tokens: Iterable[Any]):
        self._tokens: List[Any] = [
            tok for tok in tokens if not _is_ignored_token(tok)
        ]
        self._pos = 0

    def parse(self):
        if self._at_end():
            raise ValueError("expected expression")
        node = self._parse_expression()
        if not self._at_end():
            raise ValueError(f"unexpected token {_describe_token(self._current())}")
        return node

    def _parse_expression(self):
        return self._parse_additive()

    def _parse_additive(self):
        node = self._parse_multiplicative()
        while True:
            if self._accept("+"):
                node = BinOp(node, "+", self._parse_multiplicative())
            elif self._accept("-"):
                node = BinOp(node, "-", self._parse_multiplicative())
            else:
                return node

    def _parse_multiplicative(self):
        node = self._parse_unary()
        while True:
            if self._accept("*"):
                node = BinOp(node, "*", self._parse_unary())
            elif self._accept("/"):
                node = BinOp(node, "/", self._parse_unary())
            else:
                return node

    def _parse_unary(self):
        if self._accept("-"):
            return UnaryOp("-", self._parse_unary())
        return self._parse_power()

    def _parse_power(self):
        node = self._parse_primary()
        if self._accept("^"):
            node = BinOp(node, "^", self._parse_unary())
        return node

    def _parse_primary(self):
        tok = self._current()
        sym = _token_symbol(tok)

        if sym == "NUMBER":
            self._advance()
            return Num(_number_value(tok))

        if self._accept("("):
            node = self._parse_expression()
            self._expect(")")
            return node

        if self._at_end():
            raise ValueError("expected expression")
        raise ValueError(f"expected expression, got {_describe_token(tok)}")

    def _accept(self, symbol: str) -> bool:
        if _token_symbol(self._current()) == symbol:
            self._advance()
            return True
        return False

    def _expect(self, symbol: str) -> None:
        if not self._accept(symbol):
            if self._at_end():
                raise ValueError(f"expected {symbol!r}")
            raise ValueError(
                f"expected {symbol!r}, got {_describe_token(self._current())}"
            )

    def _advance(self) -> None:
        if self._pos < len(self._tokens):
            self._pos += 1

    def _current(self) -> Optional[Any]:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _at_end(self) -> bool:
        tok = self._current()
        return tok is None or _token_symbol(tok) == "EOF"


def _token_type(tok: Any) -> Any:
    if tok is None:
        return None
    for attr in ("type", "kind", "tag", "name"):
        if hasattr(tok, attr):
            value = getattr(tok, attr)
            if not callable(value):
                return value
    if isinstance(tok, (tuple, list)) and tok:
        return tok[0]
    return tok


def _token_value(tok: Any) -> Any:
    if tok is None:
        return None
    for attr in ("value", "lexeme", "text", "literal"):
        if hasattr(tok, attr):
            value = getattr(tok, attr)
            if not callable(value):
                return value
    if isinstance(tok, (tuple, list)):
        if len(tok) >= 2:
            return tok[1]
        if len(tok) == 1:
            return tok[0]
    if isinstance(tok, (str, int, float)) and not isinstance(tok, bool):
        return tok
    return None


def _canonical_name(obj: Any) -> str:
    if obj is None:
        return ""
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        text = name
    else:
        text = str(obj)
    text = text.strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _mapped_symbol(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped in {"+", "-", "*", "/", "^", "(", ")"}:
            return stripped

    name = _canonical_name(obj)
    if name in {"+", "-", "*", "/", "^", "(", ")"}:
        return name
    return _KIND_MAP.get(name)


def _token_symbol(tok: Any) -> Optional[str]:
    if tok is None:
        return "EOF"

    typ = _token_type(tok)
    val = _token_value(tok)

    sym = _mapped_symbol(typ)
    if sym is not None:
        return sym

    sym = _mapped_symbol(val)
    if sym is not None:
        return sym

    if _looks_like_number(val) or _looks_like_number(typ):
        return "NUMBER"

    return _canonical_name(typ) or _canonical_name(val) or None


def _is_ignored_token(tok: Any) -> bool:
    typ = _token_type(tok)
    val = _token_value(tok)
    return _canonical_name(typ) in _IGNORED_KINDS or _canonical_name(val) in _IGNORED_KINDS


def _looks_like_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(_NUMBER_RE.fullmatch(value.strip()))
    return False


def _number_value(tok: Any) -> Any:
    val = _token_value(tok)
    typ = _token_type(tok)

    for candidate in (val, typ):
        if isinstance(candidate, bool) or candidate is None:
            continue
        if isinstance(candidate, (int, float)):
            return candidate
        if isinstance(candidate, str):
            text = candidate.strip()
            if _NUMBER_RE.fullmatch(text):
                if any(ch in text for ch in ".eE"):
                    return float(text)
                return int(text)

    raise ValueError(f"invalid number token {_describe_token(tok)}")


def _describe_token(tok: Any) -> str:
    if tok is None:
        return "end of input"
    return repr(tok)