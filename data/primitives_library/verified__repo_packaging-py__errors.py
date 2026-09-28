from __future__ import annotations

import contextlib
import dataclasses
import sys
import typing

__all__ = ["ExceptionGroup"]


def __dir__() -> list[str]:
    return __all__


if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:

    class ExceptionGroup(Exception):
        message: str
        exceptions: list[Exception]

        def __init__(self, message: str, exceptions: list[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions

        def __repr__(self) -> str:
            name = type(self).__name__
            return f"{name}({self.message!r}, {self.exceptions!r})"


@dataclasses.dataclass(slots=True)
class _ErrorCollector:
    errors: list[Exception] = dataclasses.field(default_factory=list, init=False)

    def finalize(self, msg: str) -> None:
        if self.errors:
            raise ExceptionGroup(msg, self.errors)

    @contextlib.contextmanager
    def on_exit(self, msg: str) -> typing.Generator[_ErrorCollector]:
        yield self
        self.finalize(msg)

    @contextlib.contextmanager
    def collect(self, *err_cls: type[Exception]) -> typing.Generator[None]:
        caught_types = err_cls or (Exception,)
        try:
            yield
        except ExceptionGroup as exc:
            self.errors.extend(exc.exceptions)
        except caught_types as exc:
            self.errors.append(exc)

    def error(self, error: Exception) -> None:
        self.errors.append(error)