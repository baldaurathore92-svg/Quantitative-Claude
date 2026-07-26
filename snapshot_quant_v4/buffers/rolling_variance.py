"""Incremental variance estimators.

Two estimators are provided because the two use cases have genuinely different
numerical requirements.

:class:`WelfordVariance`
    Streaming (unbounded) variance via Welford's recurrence. Numerically stable
    and exact enough for arbitrarily long runs, but it *cannot* forget: samples
    from the opening auction still influence the estimate at 15:00. Used for
    session-level diagnostics.

:class:`RollingVariance`
    Sliding-window variance from running ``sum`` and ``sum of squares``. This is
    what feature normalisation needs, because a feature must be scaled by the
    dispersion of the *recent* market, not of the whole day.

    The sum-of-squares form is O(1) but is the numerically weakest of the
    common formulations: when the mean is large relative to the variance,
    ``sumsq - n * mean^2`` cancels catastrophically. Two defences are applied.
    First, an exact recomputation every ``recompute_interval`` pushes bounds
    the accumulated drift. Second, the variance is floored at zero, since the
    only way the formula can go negative is cancellation error.

    Note that Welford's recurrence has no exact sliding-window inverse, which
    is why the compensated sum-of-squares approach is used here rather than
    "Welford with removal" — the latter is unstable in exactly the regime we
    care about.
"""

from __future__ import annotations

import math

from .ring_buffer import RingBuffer


class RollingVariance:
    """Sample variance of the last ``window`` observations.

    Parameters
    ----------
    window:
        Window length in samples.
    recompute_interval:
        Pushes between exact recomputations of ``sum`` and ``sumsq``.
    min_samples:
        Observations required before :attr:`ready` is ``True``. At least 2.
    """

    __slots__ = (
        "_buffer",
        "_min_samples",
        "_recompute_interval",
        "_since_recompute",
        "_sum",
        "_sumsq",
    )

    def __init__(
        self,
        window: int,
        *,
        recompute_interval: int = 4096,
        min_samples: int | None = None,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if recompute_interval <= 0:
            raise ValueError("recompute_interval must be positive")
        self._buffer = RingBuffer(window)
        self._sum = 0.0
        self._sumsq = 0.0
        self._since_recompute = 0
        self._recompute_interval = int(recompute_interval)
        resolved_min = window if min_samples is None else int(min_samples)
        if resolved_min < 2 or resolved_min > window:
            raise ValueError(f"min_samples must be in 2..{window}, got {resolved_min}")
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
        """``True`` once enough observations are available."""
        return len(self._buffer) >= self._min_samples

    @property
    def fill_ratio(self) -> float:
        """Fraction of the window populated, in ``[0, 1]``."""
        return self._buffer.fill_ratio

    @property
    def mean(self) -> float:
        """Mean of the window, ``0.0`` when empty."""
        count = len(self._buffer)
        if count == 0:
            return 0.0
        return self._sum / count

    @property
    def variance(self) -> float:
        """Unbiased sample variance, floored at zero."""
        count = len(self._buffer)
        if count < 2:
            return 0.0
        centred = self._sumsq - (self._sum * self._sum) / count
        if centred <= 0.0:
            return 0.0
        return centred / (count - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    # -- mutation ---------------------------------------------------------- #

    def update(self, sample: float) -> float:
        """Push ``sample`` and return the updated standard deviation."""
        evicted = self._buffer.push(sample)
        self._sum += sample
        self._sumsq += sample * sample
        if evicted is not None:
            self._sum -= evicted
            self._sumsq -= evicted * evicted
        self._since_recompute += 1
        if self._since_recompute >= self._recompute_interval:
            self._recompute()
        return self.std

    def reset(self) -> None:
        """Clear all state."""
        self._buffer.clear()
        self._sum = 0.0
        self._sumsq = 0.0
        self._since_recompute = 0

    def _recompute(self) -> None:
        """Recompute both accumulators exactly. O(window), amortised O(1)."""
        total = 0.0
        total_sq = 0.0
        for value in self._buffer.iter_values():
            total += value
            total_sq += value * value
        self._sum = total
        self._sumsq = total_sq
        self._since_recompute = 0


class WelfordVariance:
    """Streaming variance using Welford's numerically stable recurrence."""

    __slots__ = ("_count", "_m2", "_mean")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def __len__(self) -> int:
        return self._count

    @property
    def ready(self) -> bool:
        """``True`` once at least two samples have been seen."""
        return self._count >= 2

    @property
    def mean(self) -> float:
        """Running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Unbiased sample variance."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    def update(self, sample: float) -> float:
        """Push ``sample`` and return the updated standard deviation."""
        self._count += 1
        delta = sample - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (sample - self._mean)
        return self.std

    def reset(self) -> None:
        """Clear all state."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0


__all__ = ["RollingVariance", "WelfordVariance"]
