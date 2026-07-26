"""Incremental, O(1) rolling estimators.

Every estimator in this package obeys the same contract:

*   ``update(sample) -> float`` is O(1) (amortised, for the monotonic window and
    the drift-corrected sums).
*   ``reset()`` returns the estimator to its initial state. This is mandatory
    rather than cosmetic: after a feed gap, statistics computed across the gap
    are meaningless, so the engine resets every estimator it owns.
*   ``ready`` reports whether enough samples exist for the estimate to be used.
    Features must consult it instead of silently emitting a value derived from
    two observations.
"""

from __future__ import annotations

from .monotonic_queue import MonotonicWindow
from .ring_buffer import RingBuffer
from .rolling_ema import EMAPair, TimeAwareEMA
from .rolling_mean import RollingMean
from .rolling_variance import RollingVariance, WelfordVariance

__all__ = [
    "EMAPair",
    "MonotonicWindow",
    "RingBuffer",
    "RollingMean",
    "RollingVariance",
    "TimeAwareEMA",
    "WelfordVariance",
]
