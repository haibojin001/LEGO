from __future__ import annotations

import functools
import typing
from collections import defaultdict

PRE_DUMP = "pre_dump"
POST_DUMP = "post_dump"
PRE_LOAD = "pre_load"
POST_LOAD = "post_load"
VALIDATES = "validates"
VALIDATES_SCHEMA = "validates_schema"


class MarshmallowHook:
    __marshmallow_hook__: dict[str, list[tuple[bool, typing.Any]]] | None = None


def validates(*field_names: str) -> typing.Callable[..., typing.Any]:
    return set_hook(None, VALIDATES, field_names=field_names)


def validates_schema(
    fn: typing.Callable[..., typing.Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
    skip_on_field_errors: bool = True,
) -> typing.Callable[..., typing.Any]:
    return set_hook(
        fn,
        VALIDATES_SCHEMA,
        many=pass_collection,
        pass_original=pass_original,
        skip_on_field_errors=skip_on_field_errors,
    )


def pre_dump(
    fn: typing.Callable[..., typing.Any] | None = None,
    *,
    pass_collection: bool = False,
) -> typing.Callable[..., typing.Any]:
    return set_hook(fn, PRE_DUMP, many=pass_collection)


def post_dump(
    fn: typing.Callable[..., typing.Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
) -> typing.Callable[..., typing.Any]:
    return set_hook(
        fn,
        POST_DUMP,
        many=pass_collection,
        pass_original=pass_original,
    )


def pre_load(
    fn: typing.Callable[..., typing.Any] | None = None,
    *,
    pass_collection: bool = False,
) -> typing.Callable[..., typing.Any]:
    return set_hook(fn, PRE_LOAD, many=pass_collection)


def post_load(
    fn: typing.Callable[..., typing.Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
) -> typing.Callable[..., typing.Any]:
    return set_hook(
        fn,
        POST_LOAD,
        many=pass_collection,
        pass_original=pass_original,
    )


def set_hook(
    fn: typing.Callable[..., typing.Any] | None,
    tag: str,
    *,
    many: bool = False,
    **kwargs: typing.Any,
) -> typing.Callable[..., typing.Any]:
    if fn is None:
        return functools.partial(set_hook, tag=tag, many=many, **kwargs)

    try:
        hooks = fn.__marshmallow_hook__
    except AttributeError:
        hooks = defaultdict(list)
        fn.__marshmallow_hook__ = hooks

    hooks[tag].append((many, kwargs))
    return fn