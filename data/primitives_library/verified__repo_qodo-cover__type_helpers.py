"""Type helpers for multilspy."""

import inspect
from typing import Callable, Type, TypeVar

R = TypeVar("R", bound=object)


def ensure_all_methods_implemented(
    source_cls: Type[object],
) -> Callable[[Type[R]], Type[R]]:
    """Create a decorator that verifies source class functions are overridden."""

    def check_all_methods_implemented(target_cls: Type[R]) -> Type[R]:
        for name, _ in inspect.getmembers(source_cls, inspect.isfunction):
            if name not in target_cls.__dict__ or not callable(
                target_cls.__dict__[name]
            ):
                raise NotImplementedError(f"{name} is not implemented in {target_cls}")
        return target_cls

    return check_all_methods_implemented