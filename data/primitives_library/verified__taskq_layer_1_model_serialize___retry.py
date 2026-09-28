"""Retry policies for taskq."""

from __future__ import annotations

import random


class RetryPolicy:
    """Exponential backoff retry policy with optional additive jitter."""

    def __init__(
        self,
        max_attempts: int = 3,
        base: float = 1.0,
        cap: float = 60.0,
        factor: float = 2.0,
        jitter: float = 0.0,
    ):
        self.max_attempts = max_attempts
        self.base = base
        self.cap = cap
        self.factor = factor
        self.jitter = jitter

    def next_delay(self, attempt: int) -> float:
        """Return the delay, in seconds, before the given retry attempt.

        attempt=1 -> base
        attempt=2 -> min(cap, base * factor)
        attempt=3 -> min(cap, base * factor ** 2)

        If jitter > 0, an additional uniform-random amount in
        [0, jitter * delay] is added.
        """
        if attempt < 1:
            return 0.0

        try:
            delay = self.base * (self.factor ** (attempt - 1))
        except OverflowError:
            delay = self.cap

        delay = min(self.cap, delay)

        if self.jitter > 0:
            delay += random.uniform(0.0, self.jitter * delay)

        return float(delay)

    def should_dead_letter(self, attempts: int) -> bool:
        """Return True when attempts has reached or exceeded max_attempts."""
        return attempts >= self.max_attempts


__all__ = ["RetryPolicy"]