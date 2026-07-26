"""O(1) sliding-window mean with drift correction.

An incremental sliding mean maintains ``sum += new - evicted``. Over hundreds of
thousands of updates the repeated addition and subtraction of floating point
values accumulates rounding error, and the error never cancels because the
subtraction is not the exact inverse of the earlier addition.

The estimator therefore recomputes the sum exactly every
``recompute_interval`` pushes. That single O(N) pass amortises to O(1) per
update (``N / interval`` work per push, with ``interval`` chosen well above
``N``) and bounds the drift to at most ``interval`` accumulation steps.
"""

from __future__ import annotations

from .ring_buffer import RingBuffer


class RollingMean:
    """Mean of the last ``window`` samples.

    Parameters
    ----------
    window:
        Window length in samples.
    recompute_interval:
        Number of pushes between exact recomputations of the running sum.
    min_samples:
        Number of samples required before :attr:`ready` becomes ``True``.
        Defaults to the full window; a partially filled window produces a mean
        with a much larger standard error and, more importantly, one whose
        effective lookback changes on every update.
    """

    __slots__ = ("_buffer", "_min_samples", "_recompute_interval", "_since_recompute", "_sum")

    def __init__(
        self,
        window: int,
        *,
        recompute_interval: int = 8192,
        min_samples: int | None = None,
    ) -> None:
        if recompute_interval <= 0:
            raise ValueError("recompute_interval must be positive")
        self._buffer = RingBuffer(window)
        self._sum = 0.0
        self._since_recompute = 0
        self._recompute_interval = int(recompute_interval)
        resolved_min = window if min_samples is None else int(min_samples)
        if resolved_min <= 0 or resolved_min > window:
            raise ValueError(f"min_samples must be in 1..{window}, got {resolved_min}")
        self._min_samples = resolved_min

    # -- state ------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def window(self) -> int:
        """Configured window length."""
        return self._buffer.capacity

    @property
    def ready(self) -> bool:
        """``True`` once ``min_samples`` observations are available."""
        return len(self._buffer) >= self._min_samples

    @property
    def fill_ratio(self) -> float:
        """Fraction of the window populated, in ``[0, 1]``."""
        return self._buffer.fill_ratio

    @property
    def value(self) -> float:
        """Current mean, or ``0.0`` when no samples have been seen."""
        count = len(self._buffer)
        if count == 0:
            return 0.0
        return self._sum / count

    # -- mutation ---------------------------------------------------------- #

    def update(self, sample: float) -> float:
        """Push ``sample`` and return the updated mean."""
        evicted = self._buffer.push(sample)
        self._sum += sample
        if evicted is not None:
            self._sum -= evicted
        self._since_recompute += 1
        if self._since_recompute >= self._recompute_interval:
            self._recompute()
        return self.value

    def reset(self) -> None:
        """Clear all state. Called on feed gaps and reconnects."""
        self._buffer.clear()
        self._sum = 0.0
        self._since_recompute = 0

    def _recompute(self) -> None:
        """Recompute the running sum exactly. O(window), amortised O(1)."""
        total = 0.0
        for value in self._buffer.iter_values():
            total += value
        self._sum = total
        self._since_recompute = 0


__all__ = ["RollingMean"]
