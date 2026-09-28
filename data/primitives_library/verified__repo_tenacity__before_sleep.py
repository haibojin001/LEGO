import traceback
import typing

from tenacity import _utils

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


def before_sleep_nothing(retry_state: "RetryCallState") -> None:
    """Before sleep strategy that does nothing."""
    return None


def before_sleep_log(
    logger: _utils.LoggerProtocol,
    log_level: int,
    exc_info: bool = False,
    sec_format: str = "%.3g",
) -> typing.Callable[["RetryCallState"], None]:
    """Before sleep strategy that logs to some logger the attempt."""

    def log_it(retry_state: "RetryCallState") -> None:
        outcome = retry_state.outcome
        if outcome is None:
            raise RuntimeError("log_it() called before outcome was set")

        action = retry_state.next_action
        if action is None:
            raise RuntimeError("log_it() called before next_action was set")

        if outcome.failed:
            error = outcome.exception()
            event = "raised"
            detail = f"{error.__class__.__name__}: {error}"
        else:
            event = "returned"
            detail = outcome.result()

        message = (
            f"Retrying {retry_state.get_fn_name()} "
            f"in {sec_format % action.sleep} seconds as it {event} {detail}."
        )

        if exc_info and outcome.failed:
            error = outcome.exception()
            if error is not None:
                rendered = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
                message = f"{message}\n{rendered.rstrip()}"

        logger.log(log_level, message)

    return log_it