from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta


class Hook:
    """
    Event handler that invokes an arbitrary callback when invoked.
    If the timeout_milliseconds argument is greater than 0,
    the hook will be suspended for n milliseconds after it's being invoked.
    """

    def __init__(
        self,
        callback: Callable[..., object],
        timeout_milliseconds: int = 0,
        *callback_args: object,
    ) -> None:
        self.callback = callback
        self.timeout_milliseconds = timeout_milliseconds
        self.callback_args = callback_args
        self.ready_time = datetime.now()

    def is_ready(self) -> bool:
        """
        Returns whether the hook is ready to invoke its callback or not.
        """
        return datetime.now() >= self.ready_time

    def invoke(self) -> None:
        """
        Run callback, optionally passing a variable number
        of arguments `callback_args`.
        """
        if self.timeout_milliseconds > 0:
            self.ready_time = datetime.now() + timedelta(
                milliseconds=self.timeout_milliseconds
            )

        self.callback(self.callback_args)