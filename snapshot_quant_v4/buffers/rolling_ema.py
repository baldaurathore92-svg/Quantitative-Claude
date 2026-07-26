"""Time-aware exponential moving averages.

Why not a fixed ``alpha``
-------------------------
A conventional EMA applies a constant weight per *sample*. A SnapQuote feed is
event driven and throttled: the interval between two accepted snapshots ranges
from tens of milliseconds during an auction imbalance to several seconds in a
quiet counter. With a constant ``alpha``, the same estimator represents a
half-second lookback in a busy market and a thirty-second lookback in a quiet
one — the feature it produces is not a stationary function of the market.

:class:`TimeAwareEMA` instead derives the weight from the elapsed exchange time,
``alpha = 1 - exp(-ln2 * dt / half_life)``, which fixes the lookback in
*seconds*. The estimator remains a single multiply-add: O(1), no allocation.

:class:`EMAPair` bundles a fast and a slow EMA of the same series, which is the
correct construction for a momentum feature (``fast - slow``). The originally
specified "EMA(current) - EMA(previous)" is the first difference of a single
EMA; it has a signal-to-noise ratio that degrades as the smoothing increases,
and it is not what a momentum estimator is meant to be.
"""

from __future__ import annotations

from ..utils.math_utils import ema_alpha


class TimeAwareEMA:
    """Exponential moving average whose decay is driven by elapsed time.

    Parameters
    ----------
    half_life_ms:
        Time after which an observation's weight has halved.
    max_dt_ms:
        Upper clamp on the reported interval. Protects the estimator from a
        feed stall, which would otherwise produce ``alpha == 1`` and discard the
        entire history in a single update.
    warmup_updates:
        Number of updates before :attr:`ready` becomes ``True``.
    """

    __slots__ = ("_half_life_ms", "_initialised", "_max_dt_ms", "_updates", "_value", "_warmup")

    def __init__(
        self,
        half_life_ms: float,
        *,
        max_dt_ms: float = 5_000.0,
        warmup_updates: int = 1,
    ) -> None:
        if half_life_ms <= 0.0:
            raise ValueError(f"half_life_ms must be positive, got {half_life_ms}")
        if max_dt_ms <= 0.0:
            raise ValueError(f"max_dt_ms must be positive, got {max_dt_ms}")
        if warmup_updates < 1:
            raise ValueError("warmup_updates must be >= 1")
        self._half_life_ms = float(half_life_ms)
        self._max_dt_ms = float(max_dt_ms)
        self._value = 0.0
        self._initialised = False
        self._updates = 0
        self._warmup = int(warmup_updates)

    # -- state ------------------------------------------------------------- #

    @property
    def half_life_ms(self) -> float:
        """Configured half life in milliseconds."""
        return self._half_life_ms

    @property
    def initialised(self) -> bool:
        """``True`` once the first observation has seeded the estimator."""
        return self._initialised

    @property
    def ready(self) -> bool:
        """``True`` once ``warmup_updates`` observations have been applied."""
        return self._updates >= self._warmup

    @property
    def updates(self) -> int:
        """Number of observations applied since the last reset."""
        return self._updates

    @property
    def value(self) -> float:
        """Current estimate, ``0.0`` before initialisation."""
        return self._value

    # -- mutation ---------------------------------------------------------- #

    def update(self, sample: float, dt_ms: float) -> float:
        """Apply one observation separated from the previous one by ``dt_ms``.

        The first observation seeds the estimator directly rather than being
        blended into a zero, which would introduce a downward bias that decays
        only over several half-lives.
        """
        self._updates += 1
        if not self._initialised:
            self._value = sample
            self._initialised = True
            return self._value
        alpha = ema_alpha(dt_ms, self._half_life_ms, self._max_dt_ms)
        self._value += alpha * (sample - self._value)
        return self._value

    def reset(self) -> None:
        """Clear all state."""
        self._value = 0.0
        self._initialised = False
        self._updates = 0


class EMAPair:
    """A fast/slow pair of time-aware EMAs over one series.

    Parameters
    ----------
    fast_half_life_ms:
        Half life of the responsive estimator.
    slow_half_life_ms:
        Half life of the reference estimator. Must be strictly greater than the
        fast half life, otherwise ``spread`` has no defined sign convention.
    max_dt_ms:
        Interval clamp shared by both estimators.
    warmup_updates:
        Updates required before :attr:`ready` is ``True``.
    """

    __slots__ = ("fast", "slow")

    def __init__(
        self,
        fast_half_life_ms: float,
        slow_half_life_ms: float,
        *,
        max_dt_ms: float = 5_000.0,
        warmup_updates: int = 4,
    ) -> None:
        if slow_half_life_ms <= fast_half_life_ms:
            raise ValueError(
                "slow_half_life_ms must exceed fast_half_life_ms "
                f"({slow_half_life_ms} <= {fast_half_life_ms})"
            )
        self.fast = TimeAwareEMA(
            fast_half_life_ms, max_dt_ms=max_dt_ms, warmup_updates=warmup_updates
        )
        self.slow = TimeAwareEMA(
            slow_half_life_ms, max_dt_ms=max_dt_ms, warmup_updates=warmup_updates
        )

    @property
    def ready(self) -> bool:
        """``True`` once both estimators have warmed up."""
        return self.fast.ready and self.slow.ready

    @property
    def spread(self) -> float:
        """``fast - slow``: the momentum of the underlying series."""
        return self.fast.value - self.slow.value

    def update(self, sample: float, dt_ms: float) -> float:
        """Apply one observation to both estimators and return :attr:`spread`."""
        self.fast.update(sample, dt_ms)
        self.slow.update(sample, dt_ms)
        return self.spread

    def reset(self) -> None:
        """Clear both estimators."""
        self.fast.reset()
        self.slow.reset()


__all__ = ["EMAPair", "TimeAwareEMA"]
