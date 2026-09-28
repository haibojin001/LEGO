from reprlib import repr
from traceback import format_exception
from typing import cast

from graphql import GraphQLError

from .utils import unwrap_graphql_error


def format_error(error: GraphQLError, debug: bool = False) -> dict:
    formatted = cast(dict, error.formatted)

    if debug:
        if "extensions" not in formatted:
            formatted["extensions"] = {}
        formatted["extensions"]["exception"] = get_error_extension(error)

    return formatted


def get_error_extension(error: GraphQLError) -> dict | None:
    unwrapped_error = unwrap_graphql_error(error)

    if unwrapped_error is None or error.__traceback__ is None:
        return None

    return {
        "stacktrace": get_formatted_error_traceback(unwrapped_error),
        "context": get_formatted_error_context(unwrapped_error),
    }


def get_formatted_error_traceback(error: Exception) -> list[str]:
    lines: list[str] = []

    for formatted_line in format_exception(
        type(error), error, error.__traceback__
    ):
        lines.extend(formatted_line.rstrip().splitlines())

    return lines


def get_formatted_error_context(error: Exception) -> dict | None:
    traceback = error.__traceback__

    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next

    if traceback is None:
        return None

    return {
        name: repr(value)
        for name, value in traceback.tb_frame.f_locals.items()
    }