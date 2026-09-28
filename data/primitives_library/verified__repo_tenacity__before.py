import typing

from tenacity import _utils

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


def before_nothing(retry_state: "RetryCallState") -> None:
    """Before call strategy that does nothing."""
    return None


def before_log(
    logger: _utils.LoggerProtocol, log_level: int
) -> typing.Callable[["RetryCallState"], None]:
    """Before call strategy that logs to some logger the attempt."""

    def log_it(retry_state: "RetryCallState") -> None:
        function_name = retry_state.get_fn_name()
        attempt = _utils.to_ordinal(retry_state.attempt_number)
        logger.log(
            log_level,
            f"Starting call to '{function_name}', this is the {attempt} time calling it.",
        )

    return log_it