import typing

from tenacity import _utils

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


def after_nothing(retry_state: "RetryCallState") -> None:
    """After call strategy that does nothing."""


def after_log(
    logger: _utils.LoggerProtocol,
    log_level: int,
    sec_format: str = "%.3g",
) -> typing.Callable[["RetryCallState"], None]:
    """After call strategy that logs to some logger the finished attempt."""

    def log_it(retry_state: "RetryCallState") -> None:
        function_name = retry_state.get_fn_name()
        elapsed = retry_state.seconds_since_start
        elapsed_text = sec_format % elapsed if elapsed is not None else "?"
        ordinal = _utils.to_ordinal(retry_state.attempt_number)
        logger.log(
            log_level,
            f"Finished call to '{function_name}' after {elapsed_text}(s), "
            f"this was the {ordinal} time calling it.",
        )

    return log_it