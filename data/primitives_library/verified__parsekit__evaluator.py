import importlib
import re

from parsekit.ast_nodes import BinOp, Num, UnaryOp

__all__ = ["evaluate", "calc"]

_MISSING = object()

_OPERATOR_MAP = {
    "+": "+",
    "PLUS": "+",
    "ADD": "+",
    "ADDITION": "+",
    "-": "-",
    "MINUS": "-",
    "SUB": "-",
    "SUBTRACT": "-",
    "SUBTRACTION": "-",
    "*": "*",
    "STAR": "*",
    "ASTERISK": "*",
    "MUL": "*",
    "MULT": "*",
    "MULTIPLY": "*",
    "MULTIPLICATION": "*",
    "TIMES": "*",
    "/": "/",
    "SLASH": "/",
    "DIV": "/",
    "DIVIDE": "/",
    "DIVISION": "/",
    "^": "^",
    "CARET": "^",
    "POW": "^",
    "POWER": "^",
    "EXP": "^",
    "EXPONENT": "^",
    "EXPONENTIATE": "^",
    "EXPONENTIATION": "^",
}


def _get_attr(obj, *names):
    for name in names:
        try:
            return getattr(obj, name)
        except AttributeError:
            pass
    return _MISSING


def _class_name(obj):
    try:
        return obj.__class__.__name__
    except Exception:
        return ""


def _is_node(obj, cls, name):
    return isinstance(obj, cls) or _class_name(obj) == name


def _is_ast_node(obj):
    return (
        _is_node(obj, Num, "Num")
        or _is_node(obj, UnaryOp, "UnaryOp")
        or _is_node(obj, BinOp, "BinOp")
    )


def _as_float(value):
    if isinstance(value, complex):
        if value.imag == 0:
            return float(value.real)
        raise TypeError("evaluation produced a complex result")
    return float(value)


def _normalize_operator_text(text):
    text = text.strip()
    if not text:
        return None

    candidates = [text, text.upper()]
    upper = text.upper()

    if "." in upper:
        candidates.append(upper.rsplit(".", 1)[-1])
    if ":" in upper:
        candidates.append(upper.rsplit(":", 1)[-1])

    for prefix in ("TOKEN_", "TOK_", "OP_", "KIND_", "TT_"):
        if upper.startswith(prefix):
            candidates.append(upper[len(prefix):])

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in _OPERATOR_MAP:
            return _OPERATOR_MAP[candidate]

    return None


def _operator_symbol(op):
    stack = [op]
    seen = set()

    while stack:
        value = stack.pop()
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)

        if isinstance(value, str):
            symbol = _normalize_operator_text(value)
            if symbol is not None:
                return symbol
            continue

        if isinstance(value, (tuple, list)):
            stack.extend(reversed(value))
            continue

        for attr in ("value", "lexeme", "text", "string", "literal", "kind", "type", "name"):
            try:
                attr_value = getattr(value, attr)
            except AttributeError:
                continue
            if attr_value is not value:
                stack.append(attr_value)

        try:
            text = str(value)
        except Exception:
            text = ""
        if text and text != object.__repr__(value):
            symbol = _normalize_operator_text(text)
            if symbol is not None:
                return symbol

    raise TypeError("unknown operator: {!r}".format(op))


def _field_operator_symbol(value):
    if value is _MISSING:
        return None

    if _is_ast_node(value):
        return None

    if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        return None

    try:
        return _operator_symbol(value)
    except TypeError:
        return None


def _extract_unary(node):
    op = _get_attr(node, "op", "operator")
    operand = _get_attr(node, "operand", "expr", "node", "right")

    if op is _MISSING:
        raise TypeError("UnaryOp node is missing an operator")
    if operand is _MISSING:
        raise TypeError("UnaryOp node is missing an operand")

    op_symbol = _field_operator_symbol(op)
    if op_symbol is not None:
        return op, operand

    operand_symbol = _field_operator_symbol(operand)
    if operand_symbol is not None:
        return operand, op

    return op, operand


def _extract_binop(node):
    left_attr = _get_attr(node, "left", "lhs")
    op_attr = _get_attr(node, "op", "operator")
    right_attr = _get_attr(node, "right", "rhs")

    if left_attr is _MISSING:
        raise TypeError("BinOp node is missing a left operand")
    if right_attr is _MISSING:
        raise TypeError("BinOp node is missing a right operand")
    if op_attr is _MISSING:
        raise TypeError("BinOp node is missing an operator")

    values = (left_attr, op_attr, right_attr)
    operator_indexes = []
    for index, value in enumerate(values):
        if _field_operator_symbol(value) is not None:
            operator_indexes.append(index)

    if 1 in operator_indexes:
        return left_attr, op_attr, right_attr

    if 0 in operator_indexes:
        return op_attr, left_attr, right_attr

    if 2 in operator_indexes:
        return left_attr, right_attr, op_attr

    return left_attr, op_attr, right_attr


def evaluate(node) -> float:
    if _is_node(node, Num, "Num"):
        value = _get_attr(node, "value", "number", "n")
        if value is _MISSING:
            raise TypeError("Num node is missing a numeric value")
        return _as_float(value)

    if _is_node(node, UnaryOp, "UnaryOp"):
        op, operand = _extract_unary(node)

        value = evaluate(operand)
        symbol = _operator_symbol(op)

        if symbol == "+":
            return _as_float(+value)
        if symbol == "-":
            return _as_float(-value)
        raise TypeError("unsupported unary operator: {!r}".format(op))

    if _is_node(node, BinOp, "BinOp"):
        left_node, op, right_node = _extract_binop(node)

        left = evaluate(left_node)
        right = evaluate(right_node)
        symbol = _operator_symbol(op)

        if symbol == "+":
            return _as_float(left + right)
        if symbol == "-":
            return _as_float(left - right)
        if symbol == "*":
            return _as_float(left * right)
        if symbol == "/":
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return _as_float(left / right)
        if symbol == "^":
            return _as_float(left ** right)

        raise TypeError("unsupported binary operator: {!r}".format(op))

    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return _as_float(node)

    raise TypeError("unsupported AST node: {!r}".format(node))


def calc(s: str) -> float:
    try:
        node = _parse_with_package_parser(s)
    except (ImportError, AttributeError, TypeError):
        node = _parse_locally(s)
    return evaluate(node)


def _parse_with_package_parser(s):
    parser_module = importlib.import_module("parsekit.parser")

    parse = getattr(parser_module, "parse", None)
    if callable(parse):
        return parse(s)

    parser_cls = getattr(parser_module, "_Parser")
    parser = parser_cls(s)

    for method_name in ("parse", "_parse", "expression", "expr"):
        method = getattr(parser, method_name, None)
        if callable(method):
            return method()

    if callable(parser):
        return parser()

    raise AttributeError("_Parser has no callable parse method")


_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _new_num(value):
    try:
        return Num(value)
    except TypeError:
        return Num(value=value)


def _new_binop(left, op, right):
    try:
        return BinOp(left=left, op=op, right=right)
    except TypeError:
        pass

    try:
        return BinOp(left, op, right)
    except TypeError:
        return BinOp(op, left, right)


def _new_unaryop(op, operand):
    try:
        return UnaryOp(op=op, operand=operand)
    except TypeError:
        pass

    try:
        return UnaryOp(op, operand)
    except TypeError:
        return UnaryOp(operand, op)


def _parse_locally(s):
    return _LocalParser(s).parse()


class _LocalParser:
    def __init__(self, source):
        if not isinstance(source, str):
            raise TypeError("calc() argument must be a string")
        self._tokens = self._tokenize(source)
        self._index = 0

    def parse(self):
        node = self._expression()
        if self._current()[0] != "EOF":
            raise SyntaxError("unexpected token {!r}".format(self._current()[1]))
        return node

    def _tokenize(self, source):
        tokens = []
        index = 0
        length = len(source)

        while index < length:
            char = source[index]

            if char.isspace():
                index += 1
                continue

            match = _NUMBER_RE.match(source, index)
            if match is not None:
                text = match.group(0)
                tokens.append(("NUMBER", text))
                index = match.end()
                continue

            if char in "+-*/^":
                tokens.append(("OP", char))
                index += 1
                continue

            if char == "(":
                tokens.append(("LPAREN", char))
                index += 1
                continue

            if char == ")":
                tokens.append(("RPAREN", char))
                index += 1
                continue

            raise SyntaxError("unexpected character {!r}".format(char))

        tokens.append(("EOF", ""))
        return tokens

    def _current(self):
        return self._tokens[self._index]

    def _advance(self):
        token = self._current()
        self._index += 1
        return token

    def _match_op(self, *ops):
        token = self._current()
        if token[0] == "OP" and token[1] in ops:
            self._advance()
            return token[1]
        return None

    def _expression(self):
        node = self._term()

        while True:
            op = self._match_op("+", "-")
            if op is None:
                return node
            node = _new_binop(node, op, self._term())

    def _term(self):
        node = self._unary()

        while True:
            op = self._match_op("*", "/")
            if op is None:
                return node
            node = _new_binop(node, op, self._unary())

    def _unary(self):
        op = self._match_op("+", "-")
        if op is not None:
            return _new_unaryop(op, self._unary())
        return self._power()

    def _power(self):
        node = self._primary()
        op = self._match_op("^")
        if op is not None:
            node = _new_binop(node, op, self._unary())
        return node

    def _primary(self):
        token = self._current()

        if token[0] == "NUMBER":
            self._advance()
            return _new_num(float(token[1]))

        if token[0] == "LPAREN":
            self._advance()
            node = self._expression()
            if self._current()[0] != "RPAREN":
                raise SyntaxError("expected ')'")
            self._advance()
            return node

        if token[0] == "EOF":
            raise SyntaxError("unexpected end of input")

        raise SyntaxError("unexpected token {!r}".format(token[1]))