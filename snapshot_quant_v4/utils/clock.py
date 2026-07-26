"""Clock abstraction.

Every time-dependent decision in the engine (WATCH expiry, maximum hold,
cooldown) reads the clock through the :class:`Clock` protocol so that unit
tests can drive time deterministically with :class:`ManualClock` instead of
sleeping.

Two distinct notions of time are deliberately kept separate:

``monotonic_ms``
    Never goes backwards, unaffected by NTP adjustments. Used for *timeouts*
    and latency measurement.
``wall_ms``
    Unix epoch milliseconds. Used only for display and for correlating with
    exchange timestamps.

Feature mathematics never uses either of these: features use the
``exchange_timestamp`` carried by the snapshot itself, which is the only clock
that is consistent with the data.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injectable source of time."""

    def monotonic_ms(self) -> float:
        """Return a monotonically increasing millisecond counter."""

    def wall_ms(self) -> float:
        """Return Unix epoch milliseconds."""


class SystemClock:
    """Production clock backed by :mod:`time`."""

    __slots__ = ()

    def monotonic_ms(self) -> float:
        return time.monotonic_ns() / 1_000_000.0

    def wall_ms(self) -> float:
        return time.time() * 1000.0


class ManualClock:
    """Deterministic clock for tests and replay.

    Parameters
    ----------
    monotonic_ms:
        Initial value of the monotonic counter.
    wall_ms:
        Initial value of the wall counter.
    """

    __slots__ = ("_monotonic_ms", "_wall_ms")

    def __init__(self, monotonic_ms: float = 0.0, wall_ms: float = 0.0) -> None:
        self._monotonic_ms = float(monotonic_ms)
        self._wall_ms = float(wall_ms)

    def monotonic_ms(self) -> float:
        return self._monotonic_ms

    def wall_ms(self) -> float:
        return self._wall_ms

    def advance(self, delta_ms: float) -> None:
        """Advance both counters by ``delta_ms`` milliseconds.

        Raises
        ------
        ValueError
            If ``delta_ms`` is negative, which would violate monotonicity.
        """
        if delta_ms < 0.0:
            raise ValueError("ManualClock cannot move backwards")
        self._monotonic_ms += delta_ms
        self._wall_ms += delta_ms

    def set_wall_ms(self, wall_ms: float) -> None:
        """Set the wall clock without touching the monotonic counter."""
        self._wall_ms = float(wall_ms)
