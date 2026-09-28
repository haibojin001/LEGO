import re
from collections.abc import Callable, Mapping
from typing import Any, Optional, Protocol, Union

from .utils import freeze

__all__ = ("Query", "QueryLike", "where")


def is_sequence(obj):
    return hasattr(obj, "__iter__")


class QueryLike(Protocol):
    def __call__(self, value: Mapping) -> bool:
        ...

    def __hash__(self) -> int:
        ...


class QueryInstance:
    def __init__(
        self,
        test: Callable[[Mapping], bool],
        hashval: Optional[tuple],
    ):
        self._test = test

        if hashval is not None:
            try:
                hash(hashval)
            except TypeError:
                hashval = None

        self._hash = hashval

    def is_cacheable(self) -> bool:
        return self._hash is not None

    def __call__(self, value: Mapping) -> bool:
        return self._test(value)

    def __hash__(self) -> int:
        return hash(self._hash)

    def __repr__(self):
        return "QueryImpl{}".format(self._hash)

    def __eq__(self, other: object):
        if isinstance(other, QueryInstance):
            return self._hash == other._hash

        return False

    def __and__(self, other: "QueryInstance") -> "QueryInstance":
        if self.is_cacheable() and other.is_cacheable():
            hashval = ("and", frozenset((self._hash, other._hash)))
        else:
            hashval = None

        return QueryInstance(lambda value: self(value) and other(value), hashval)

    def __or__(self, other: "QueryInstance") -> "QueryInstance":
        if self.is_cacheable() and other.is_cacheable():
            hashval = ("or", frozenset((self._hash, other._hash)))
        else:
            hashval = None

        return QueryInstance(lambda value: self(value) or other(value), hashval)

    def __invert__(self) -> "QueryInstance":
        hashval = ("not", self._hash) if self.is_cacheable() else None
        return QueryInstance(lambda value: not self(value), hashval)


class Query(QueryInstance):
    def __init__(self) -> None:
        self._path: tuple[Union[str, Callable], ...] = ()

        def empty_query(_):
            raise RuntimeError("Empty query was evaluated")

        super().__init__(empty_query, (None,))

    def __repr__(self):
        return "{}()".format(type(self).__name__)

    def __hash__(self):
        return super().__hash__()

    def __getattr__(self, item: str):
        query = type(self)()
        query._path = self._path + (item,)
        query._hash = ("path", query._path) if self.is_cacheable() else None
        return query

    def __getitem__(self, item: str):
        return self.__getattr__(item)

    def _generate_test(
        self,
        test: Callable[[Any], bool],
        hashval: tuple,
        allow_empty_path: bool = False,
    ) -> QueryInstance:
        if not self._path and not allow_empty_path:
            raise ValueError("Query has no path")

        def run(document):
            try:
                for path_part in self._path:
                    if isinstance(path_part, str):
                        document = document[path_part]
                    else:
                        document = path_part(document)
            except (KeyError, TypeError):
                return False
            else:
                return test(document)

        return QueryInstance(
            lambda value: run(value),
            hashval if self.is_cacheable() else None,
        )

    def __eq__(self, rhs: Any):
        return self._generate_test(
            lambda value: value == rhs,
            ("==", self._path, freeze(rhs)),
        )

    def __ne__(self, rhs: Any):
        return self._generate_test(
            lambda value: value != rhs,
            ("!=", self._path, freeze(rhs)),
        )

    def __lt__(self, rhs: Any) -> QueryInstance:
        return self._generate_test(
            lambda value: value < rhs,
            ("<", self._path, freeze(rhs)),
        )

    def __le__(self, rhs: Any) -> QueryInstance:
        return self._generate_test(
            lambda value: value <= rhs,
            ("<=", self._path, freeze(rhs)),
        )

    def __gt__(self, rhs: Any) -> QueryInstance:
        return self._generate_test(
            lambda value: value > rhs,
            (">", self._path, freeze(rhs)),
        )

    def __ge__(self, rhs: Any) -> QueryInstance:
        return self._generate_test(
            lambda value: value >= rhs,
            (">=", self._path, freeze(rhs)),
        )

    def exists(self) -> QueryInstance:
        return self._generate_test(
            lambda value: True,
            ("exists", self._path),
        )

    def matches(self, regex: str, flags: int = 0) -> QueryInstance:
        def test(value):
            if not isinstance(value, str):
                return False

            return re.match(regex, value, flags) is not None

        return self._generate_test(
            test,
            ("matches", self._path, regex),
        )

    def search(self, regex: str, flags: int = 0) -> QueryInstance:
        def test(value):
            if not isinstance(value, str):
                return False

            return re.search(regex, value, flags) is not None

        return self._generate_test(
            test,
            ("search", self._path, regex),
        )

    def test(self, func: Callable[[Any], bool], *args) -> QueryInstance:
        return self._generate_test(
            lambda value: func(value, *args),
            ("test", self._path, func, args),
        )

    def any(self, cond) -> QueryInstance:
        if callable(cond):
            def test(value):
                return is_sequence(value) and any(cond(element) for element in value)
        else:
            def test(value):
                return is_sequence(value) and any(element in cond for element in value)

        return self._generate_test(
            test,
            ("any", self._path, freeze(cond)),
        )

    def all(self, cond) -> QueryInstance:
        if callable(cond):
            def test(value):
                return is_sequence(value) and all(cond(element) for element in value)
        else:
            def test(value):
                return is_sequence(value) and all(element in value for element in cond)

        return self._generate_test(
            test,
            ("all", self._path, freeze(cond)),
        )

    def one_of(self, items) -> QueryInstance:
        return self._generate_test(
            lambda value: value in items,
            ("one_of", self._path, freeze(items)),
        )

    def fragment(self, document: Mapping) -> QueryInstance:
        def test(value):
            if not isinstance(value, Mapping):
                return False

            for key in document:
                if value.get(key) != document[key]:
                    return False

            return True

        return self._generate_test(
            test,
            ("fragment", self._path, freeze(document)),
        )

    def noop(self) -> QueryInstance:
        return self._generate_test(
            lambda value: True,
            (),
            allow_empty_path=True,
        )

    def map(self, fn: Callable) -> "Query":
        query = type(self)()
        query._path = self._path + (fn,)
        query._hash = ("path", query._path) if self.is_cacheable() else None
        return query


def where(key: str) -> Query:
    return Query()[key]