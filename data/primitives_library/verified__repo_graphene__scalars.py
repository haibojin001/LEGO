from typing import Any

from graphql import Undefined
from graphql.language.ast import (
    BooleanValueNode,
    FloatValueNode,
    IntValueNode,
    StringValueNode,
)

from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType


class ScalarOptions(BaseOptions):
    pass


class Scalar(UnmountedType, BaseType):
    @classmethod
    def __init_subclass_with_meta__(cls, **options):
        meta = ScalarOptions(cls)
        super(Scalar, cls).__init_subclass_with_meta__(_meta=meta, **options)

    serialize = None
    parse_value = None
    parse_literal = None

    @classmethod
    def get_type(cls):
        return cls


MAX_INT = 2147483647
MIN_INT = -2147483648


class Int(Scalar):
    @staticmethod
    def coerce_int(value):
        try:
            number = int(value)
        except ValueError:
            try:
                number = int(float(value))
            except ValueError:
                return Undefined

        if MIN_INT <= number <= MAX_INT:
            return number
        return Undefined

    serialize = coerce_int
    parse_value = coerce_int

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, IntValueNode):
            number = int(ast.value)
            if MIN_INT <= number <= MAX_INT:
                return number
        return Undefined


class BigInt(Scalar):
    @staticmethod
    def coerce_int(value):
        try:
            number = int(value)
        except ValueError:
            try:
                number = int(float(value))
            except ValueError:
                return Undefined
        return number

    serialize = coerce_int
    parse_value = coerce_int

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, IntValueNode):
            return int(ast.value)
        return Undefined


class Float(Scalar):
    @staticmethod
    def coerce_float(value: Any) -> float:
        try:
            return float(value)
        except ValueError:
            return Undefined

    serialize = coerce_float
    parse_value = coerce_float

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, (FloatValueNode, IntValueNode)):
            return float(ast.value)
        return Undefined


class String(Scalar):
    @staticmethod
    def coerce_string(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    serialize = coerce_string
    parse_value = coerce_string

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, StringValueNode):
            return ast.value
        return Undefined


class Boolean(Scalar):
    serialize = bool
    parse_value = bool

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, BooleanValueNode):
            return ast.value
        return Undefined


class ID(Scalar):
    serialize = str
    parse_value = str

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, (StringValueNode, IntValueNode)):
            return ast.value
        return Undefined