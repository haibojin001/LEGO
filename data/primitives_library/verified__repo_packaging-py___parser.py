from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias

from .specifiers import Specifier

if not hasattr(Specifier, "_specifier_regex_str"):
    Specifier._specifier_regex_str = (
        Specifier._operator_regex_str + Specifier._version_regex_str
    )

from ._tokenizer import DEFAULT_RULES, ParserSyntaxError, Tokenizer


class Node:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.value!r})>"

    def serialize(self) -> str:
        raise NotImplementedError

    def __getstate__(self) -> str:
        return self.value

    def _restore_value(self, value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"Cannot restore {self.__class__.__name__} value from {value!r}"
            )
        self.value = value

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            self._restore_value(state)
            return

        if isinstance(state, tuple) and len(state) == 2:
            _, slot_dict = state
            if isinstance(slot_dict, dict) and "value" in slot_dict:
                self._restore_value(slot_dict["value"])
                return

        if isinstance(state, dict) and "value" in state:
            self._restore_value(state["value"])
            return

        raise TypeError(f"Cannot restore {self.__class__.__name__} from {state!r}")


class Variable(Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


class Value(Node):
    __slots__ = ()

    def serialize(self) -> str:
        value = str(self)
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        raise ValueError(
            "Cannot serialize marker value containing both quote characters"
        )


class Op(Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


MarkerLogical = Literal["and", "or"]
MarkerVar = Variable | Value
MarkerItem = tuple[MarkerVar, Op, MarkerVar]
MarkerAtom = MarkerItem | Sequence["MarkerAtom"]
MarkerList: TypeAlias = 'list["MarkerList" | MarkerAtom | MarkerLogical]'


class ParsedRequirement(NamedTuple):
    name: str
    url: str
    extras: list[str]
    specifier: str
    marker: MarkerList | None


def parse_requirement(source: str) -> ParsedRequirement:
    return _parse_requirement(Tokenizer(source, rules=DEFAULT_RULES))


def _parse_requirement(tokenizer: Tokenizer) -> ParsedRequirement:
    tokenizer.consume("WS")

    name_token = tokenizer.expect(
        "IDENTIFIER",
        expected="package name at the start of dependency specifier",
    )
    name = name_token.text
    tokenizer.consume("WS")

    extras = _parse_extras(tokenizer)
    tokenizer.consume("WS")

    url, specifier, marker = _parse_requirement_details(tokenizer)
    tokenizer.expect("END", expected="end of dependency specifier")

    return ParsedRequirement(name, url, extras, specifier, marker)


def _parse_requirement_details(
    tokenizer: Tokenizer,
) -> tuple[str, str, MarkerList | None]:
    specifier = ""
    url = ""
    marker = None

    if tokenizer.check("AT"):
        tokenizer.read()
        tokenizer.consume("WS")

        url_start = tokenizer.position
        url = tokenizer.expect("URL", expected="URL after @").text

        if tokenizer.check("END", peek=True):
            return url, specifier, marker

        tokenizer.expect("WS", expected="whitespace after URL")

        if tokenizer.check("END", peek=True):
            return url, specifier, marker

        marker = _parse_requirement_marker(
            tokenizer,
            span_start=url_start,
            expected="semicolon (after URL and whitespace)",
        )
    else:
        specifier_start = tokenizer.position
        specifier = _parse_specifier(tokenizer)
        tokenizer.consume("WS")

        if tokenizer.check("END", peek=True):
            return url, specifier, marker

        marker = _parse_requirement_marker(
            tokenizer,
            span_start=specifier_start,
            expected=(
                "comma (within version specifier), semicolon (after version specifier)"
                if specifier
                else "semicolon (after name with no version specifier)"
            ),
        )

    return url, specifier, marker


def _parse_requirement_marker(
    tokenizer: Tokenizer,
    *,
    span_start: int,
    expected: str,
) -> MarkerList:
    if not tokenizer.check("SEMICOLON"):
        tokenizer.raise_syntax_error(
            f"Expected {expected} or end",
            span_start=span_start,
            span_end=None,
        )

    tokenizer.read()
    marker = _parse_marker(tokenizer)
    tokenizer.consume("WS")

    return marker


def _parse_extras(tokenizer: Tokenizer) -> list[str]:
    if not tokenizer.check("LEFT_BRACKET", peek=True):
        return []

    with tokenizer.enclosing_tokens(
        "LEFT_BRACKET",
        "RIGHT_BRACKET",
        around="extras",
    ):
        tokenizer.consume("WS")
        extras = _parse_extras_list(tokenizer)
        tokenizer.consume("WS")

    return extras


def _parse_extras_list(tokenizer: Tokenizer) -> list[str]:
    extras: list[str] = []

    if not tokenizer.check("IDENTIFIER"):
        return extras

    extras.append(tokenizer.read().text)

    while True:
        tokenizer.consume("WS")

        if tokenizer.check("IDENTIFIER", peek=True):
            tokenizer.raise_syntax_error("Expected comma between extra names")

        if not tokenizer.check("COMMA"):
            break

        tokenizer.read()
        tokenizer.consume("WS")
        extras.append(
            tokenizer.expect("IDENTIFIER", expected="extra name after comma").text
        )

    return extras


def _parse_specifier(tokenizer: Tokenizer) -> str:
    with tokenizer.enclosing_tokens(
        "LEFT_PARENTHESIS",
        "RIGHT_PARENTHESIS",
        around="version specifier",
    ):
        tokenizer.consume("WS")
        result = _parse_version_many(tokenizer)
        tokenizer.consume("WS")

    return result


def _parse_version_many(tokenizer: Tokenizer) -> str:
    result = ""

    while tokenizer.check("SPECIFIER"):
        start = tokenizer.position
        specifier = tokenizer.read().text
        result += specifier

        if tokenizer.check("VERSION_PREFIX_TRAIL", peek=True):
            message = ".* suffix can only be used with `==` or `!=` operators"
            if specifier.startswith("!=") or (
                specifier.startswith("==") and not specifier.startswith("===")
            ):
                message = (
                    ".* suffix cannot be used with pre-release, post-release, "
                    "dev or local versions"
                )

            tokenizer.raise_syntax_error(
                message,
                span_start=start,
                span_end=tokenizer.position + 1,
            )

        if tokenizer.check("VERSION_LOCAL_LABEL_TRAIL", peek=True):
            tokenizer.raise_syntax_error(
                "Local version label can only be used with `==` or `!=` operators",
                span_start=start,
                span_end=tokenizer.position,
            )

        tokenizer.consume("WS")

        if not tokenizer.check("COMMA"):
            break

        result += tokenizer.read().text
        tokenizer.consume("WS")

    return result


def parse_marker(source: str) -> MarkerList:
    return _parse_full_marker(Tokenizer(source, rules=DEFAULT_RULES))


def _parse_full_marker(tokenizer: Tokenizer) -> MarkerList:
    tokenizer.consume("WS")
    marker = _parse_marker(tokenizer)
    tokenizer.consume("WS")
    tokenizer.expect("END", expected="end of marker expression")
    return marker


def _parse_marker(tokenizer: Tokenizer) -> MarkerList:
    tokenizer.consume("WS")
    expression: MarkerList = [_parse_marker_atom(tokenizer)]
    tokenizer.consume("WS")

    while tokenizer.check("BOOLOP"):
        expression.append(tokenizer.read().text)
        tokenizer.consume("WS")
        expression.append(_parse_marker_atom(tokenizer))
        tokenizer.consume("WS")

    return expression


def _parse_marker_atom(tokenizer: Tokenizer) -> MarkerAtom:
    if tokenizer.check("LEFT_PARENTHESIS", peek=True):
        with tokenizer.enclosing_tokens(
            "LEFT_PARENTHESIS",
            "RIGHT_PARENTHESIS",
            around="marker expression",
        ):
            tokenizer.consume("WS")
            expression = _parse_marker(tokenizer)
            tokenizer.consume("WS")

        return expression

    return _parse_marker_item(tokenizer)


def _parse_marker_item(tokenizer: Tokenizer) -> MarkerItem:
    left = _parse_marker_var(tokenizer)
    tokenizer.consume("WS")
    operator = _parse_marker_op(tokenizer)
    tokenizer.consume("WS")
    right = _parse_marker_var(tokenizer)
    return left, operator, right


def _parse_marker_var(tokenizer: Tokenizer) -> MarkerVar:
    if tokenizer.check("VARIABLE"):
        return Variable(tokenizer.read().text)

    if tokenizer.check("QUOTED_STRING"):
        return Value(ast.literal_eval(tokenizer.read().text))

    tokenizer.raise_syntax_error(
        "Expected a marker variable or quoted string",
        span_start=tokenizer.position,
        span_end=tokenizer.position + 1,
    )
    raise AssertionError("unreachable")


def _parse_marker_op(tokenizer: Tokenizer) -> Op:
    if tokenizer.check("OP"):
        return Op(tokenizer.read().text)

    if tokenizer.check("IN"):
        return Op(tokenizer.read().text)

    if tokenizer.check("NOT"):
        tokenizer.read()
        tokenizer.consume("WS")
        tokenizer.expect("IN", expected="in after not")
        return Op("not in")

    tokenizer.raise_syntax_error(
        "Expected a marker operator, one of "
        "<=, <, !=, ==, >=, >, ~=, ===, in, not in",
        span_start=tokenizer.position,
        span_end=tokenizer.position + 1,
    )
    raise AssertionError("unreachable")