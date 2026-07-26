"""Per-instrument rolling statistics shared by features, confidence and regime.

Rationale
---------
Several components need the same handful of rolling quantities: the confidence
model needs a liquidity reference and a volatility reference, the regime
detector needs realised volatility and an efficiency ratio, the threshold policy
needs spread and imbalance dispersion. Computing them once, in one place, avoids
both duplicated logic and duplicated cost.

Update ordering
---------------
The engine calls :meth:`SharedStatistics.update_pre` *before* the features and
:meth:`SharedStatistics.update_post` *after* them. Consequently a feature that
normalises itself by a shared dispersion uses the dispersion that includes the
current snapshot for price-derived series (available pre-feature) and the
*previous* snapshot for feature-derived series such as the imbalance (only
available post-feature). That is intentional: using a dispersion that already
contains the current observation of the same series would be a look-ahead into
the value being normalised.

All estimators are O(1) per update and all of them are reset together on a feed
gap, because a statistic computed across a discontinuity is not a statistic.
"""

from __future__ import annotations

import math

from ..buffers.monotonic_queue import MonotonicWindow
from ..buffers.ring_buffer import RingBuffer
from ..buffers.rolling_ema import EMAPair
from ..buffers.rolling_mean import RollingMean
from ..buffers.rolling_variance import RollingVariance
from ..config import (
    ConfidenceConfig,
    QualityConfig,
    RegimeConfig,
    ThresholdConfig,
)
from ..utils.constants import PRIMARY_DEPTH_LEVELS
from ..utils.math_utils import safe_div
from ..utils.types import Side, Snapshot


class SharedStatistics:
    """Rolling market statistics for one instrument.

    Parameters
    ----------
    tick_size:
        Instrument tick size, used to express every price quantity in ticks.
    confidence, regime, threshold, quality:
        Configuration sections that determine the window lengths. Passing the
        sections rather than raw integers keeps the windows consistent with the
        components that consume them.
    """

    __slots__ = (
        "_current_depth",
        "_depth_levels",
        "_last_dt_ms",
        "_log_depth_mean",
        "_mid_delta_abs",
        "_mid_return_var",
        "_mid_ticks",
        "_mid_trend",
        "_obi_var",
        "_previous_ask_paise",
        "_previous_bid_paise",
        "_previous_mid_ticks",
        "_snapshots",
        "_spread_var",
        "_spread_window",
        "_tick_paise",
        "_tick_size",
        "_touch_stability",
        "_volume_mean",
        "_warmup_snapshots",
    )

    def __init__(
        self,
        *,
        tick_size: float,
        confidence: ConfidenceConfig,
        regime: RegimeConfig,
        threshold: ThresholdConfig,
        quality: QualityConfig,
        depth_levels: int = PRIMARY_DEPTH_LEVELS,
    ) -> None:
        if tick_size <= 0.0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self._tick_size = float(tick_size)
        # Integer tick size in paise, so touch-move comparisons are exact.
        self._tick_paise = max(1, round(tick_size * 100.0))
        self._warmup_snapshots = quality.warmup_snapshots
        self._depth_levels = int(depth_levels)

        self._snapshots = 0
        self._mid_ticks = RingBuffer(regime.window)
        self._mid_delta_abs = RollingMean(regime.window, min_samples=2)
        self._mid_return_var = RollingVariance(
            confidence.volatility_window, min_samples=min(8, confidence.volatility_window)
        )
        self._mid_trend = EMAPair(
            regime.fast_half_life_ms,
            regime.slow_half_life_ms,
            max_dt_ms=regime.max_dt_ms,
            warmup_updates=4,
        )
        self._spread_var = RollingVariance(
            threshold.reference_window, min_samples=min(8, threshold.reference_window)
        )
        self._spread_window = MonotonicWindow(confidence.stability_window)
        self._log_depth_mean = RollingMean(
            confidence.liquidity_window,
            min_samples=min(8, confidence.liquidity_window),
        )
        self._obi_var = RollingVariance(
            threshold.reference_window, min_samples=min(8, threshold.reference_window)
        )
        self._volume_mean = RollingMean(
            confidence.liquidity_window,
            min_samples=min(8, confidence.liquidity_window),
        )
        self._touch_stability = RollingMean(
            confidence.stability_window,
            min_samples=min(4, confidence.stability_window),
        )

        self._previous_mid_ticks: float | None = None
        self._previous_bid_paise = 0
        self._previous_ask_paise = 0
        self._current_depth = 0.0
        self._last_dt_ms = 0.0

    # -- lifecycle --------------------------------------------------------- #

    def reset(self) -> None:
        """Clear every estimator. Called on a feed gap or reconnect."""
        self._snapshots = 0
        self._mid_ticks.clear()
        self._mid_delta_abs.reset()
        self._mid_return_var.reset()
        self._mid_trend.reset()
        self._spread_var.reset()
        self._spread_window.reset()
        self._log_depth_mean.reset()
        self._obi_var.reset()
        self._volume_mean.reset()
        self._touch_stability.reset()
        self._previous_mid_ticks = None
        self._previous_bid_paise = 0
        self._previous_ask_paise = 0
        self._current_depth = 0.0
        self._last_dt_ms = 0.0

    # -- updates ----------------------------------------------------------- #

    def update_pre(self, snapshot: Snapshot) -> None:
        """Update every price-derived estimator. Called before the features."""
        self._snapshots += 1
        self._last_dt_ms = snapshot.dt_ms
        tick = self._tick_size

        mid_ticks = snapshot.mid / tick
        if self._previous_mid_ticks is not None and not snapshot.is_first:
            delta = mid_ticks - self._previous_mid_ticks
            self._mid_delta_abs.update(abs(delta))
            self._mid_return_var.update(delta)
        self._mid_ticks.push(mid_ticks)
        self._mid_trend.update(mid_ticks, snapshot.dt_ms)
        self._previous_mid_ticks = mid_ticks

        self._spread_var.update(snapshot.spread_ticks)
        self._spread_window.update(snapshot.spread_ticks)

        depth = float(
            snapshot.depth_quantity(Side.BID, self._depth_levels)
            + snapshot.depth_quantity(Side.ASK, self._depth_levels)
        )
        self._current_depth = depth
        # A geometric mean (mean of logs) is used as the liquidity reference:
        # depth distributions are heavy tailed, so an arithmetic mean is
        # dominated by occasional very large postings. A true rolling median
        # would need an O(log N) structure per update for no practical gain.
        self._log_depth_mean.update(math.log1p(max(depth, 0.0)))

        if not snapshot.is_first:
            # "Stable" means *orderly*, not motionless. An early version required
            # both touch prices to be unchanged, which drove stability to zero in
            # any trending market -- exactly the regime the engine is supposed to
            # trade -- and collapsed every confidence value with it. A touch that
            # drifts by one tick is normal price discovery; a touch that jumps
            # several ticks between snapshots is the instability worth measuring.
            tick_paise = self._tick_paise
            bid_jump = abs(snapshot.best_bid.price_paise - self._previous_bid_paise)
            ask_jump = abs(snapshot.best_ask.price_paise - self._previous_ask_paise)
            orderly = bid_jump <= tick_paise and ask_jump <= tick_paise
            self._touch_stability.update(1.0 if orderly else 0.0)
            self._volume_mean.update(float(max(snapshot.delta_volume, 0)))
        self._previous_bid_paise = snapshot.best_bid.price_paise
        self._previous_ask_paise = snapshot.best_ask.price_paise

    def update_post(self, directional_imbalance: float) -> None:
        """Record the feature-derived imbalance. Called after the features."""
        self._obi_var.update(directional_imbalance)

    # -- counters ---------------------------------------------------------- #

    @property
    def snapshots(self) -> int:
        """Number of accepted snapshots since the last reset."""
        return self._snapshots

    @property
    def warmup_ratio(self) -> float:
        """Progress towards a fully warmed-up state, in ``[0, 1]``."""
        return min(1.0, self._snapshots / self._warmup_snapshots)

    @property
    def warm(self) -> bool:
        """``True`` once the configured warmup count has been reached."""
        return self._snapshots >= self._warmup_snapshots

    @property
    def last_dt_ms(self) -> float:
        """Interval between the two most recent accepted snapshots."""
        return self._last_dt_ms

    # -- price statistics -------------------------------------------------- #

    @property
    def mid_ticks(self) -> float:
        """Current mid price expressed in ticks, ``0.0`` when empty."""
        if len(self._mid_ticks) == 0:
            return 0.0
        return self._mid_ticks.newest

    @property
    def mid_volatility_ticks(self) -> float:
        """Standard deviation of per-snapshot mid changes, in ticks."""
        return self._mid_return_var.std

    @property
    def mid_volatility_ready(self) -> bool:
        """``True`` once the mid volatility estimate is usable."""
        return self._mid_return_var.ready

    @property
    def trend_ticks(self) -> float:
        """Fast minus slow mid EMA, in ticks: a signed trend displacement."""
        return self._mid_trend.spread

    @property
    def trend_ready(self) -> bool:
        """``True`` once both mid EMAs have warmed up."""
        return self._mid_trend.ready

    def efficiency_ratio(self, lookback: int) -> float:
        """Kaufman efficiency ratio over ``lookback`` snapshots, in ``[0, 1]``.

        ``|net displacement| / path length``. A value near 1 means price moved in
        a straight line (trend); near 0 means it oscillated (range or noise).
        Both terms are maintained incrementally, so this accessor is O(1).
        """
        count = len(self._mid_ticks)
        if count < 3 or not self._mid_delta_abs.ready:
            return 0.0
        age = min(lookback, count - 1)
        net = abs(self._mid_ticks.newest - self._mid_ticks.at(age))
        path = self._mid_delta_abs.value * len(self._mid_delta_abs)
        if path <= 0.0:
            return 0.0
        ratio = net / path
        return ratio if ratio < 1.0 else 1.0

    # -- spread statistics ------------------------------------------------- #

    @property
    def spread_volatility_ticks(self) -> float:
        """Standard deviation of the spread, in ticks."""
        return self._spread_var.std

    @property
    def spread_mean_ticks(self) -> float:
        """Mean spread over the reference window, in ticks."""
        return self._spread_var.mean

    @property
    def spread_range_ticks(self) -> float:
        """Peak-to-trough spread range over the stability window, in ticks."""
        return self._spread_window.range

    @property
    def spread_stats_ready(self) -> bool:
        """``True`` once spread dispersion is usable."""
        return self._spread_var.ready

    # -- liquidity statistics ---------------------------------------------- #

    @property
    def depth_quantity(self) -> float:
        """Current summed quantity across the primary levels of both sides."""
        return self._current_depth

    @property
    def depth_reference(self) -> float:
        """Rolling geometric-mean depth for this instrument."""
        if not self._log_depth_mean.ready:
            return 0.0
        return math.expm1(self._log_depth_mean.value)

    @property
    def relative_depth(self) -> float:
        """Current depth divided by this instrument's own recent depth.

        Returns ``1.0`` before the reference is available, so that a
        just-started engine is not penalised as illiquid.
        """
        reference = self.depth_reference
        if reference <= 0.0:
            return 1.0
        return safe_div(self._current_depth, reference, default=1.0)

    @property
    def depth_reference_ready(self) -> bool:
        """``True`` once the liquidity reference is usable."""
        return self._log_depth_mean.ready

    @property
    def mean_delta_volume(self) -> float:
        """Mean traded quantity per snapshot interval."""
        return self._volume_mean.value

    # -- stability and imbalance ------------------------------------------- #

    @property
    def touch_stability(self) -> float:
        """Fraction of recent snapshots in which neither touch price moved."""
        if not self._touch_stability.ready:
            return 0.5
        return self._touch_stability.value

    @property
    def imbalance_volatility(self) -> float:
        """Standard deviation of the directional imbalance feature."""
        return self._obi_var.std

    @property
    def imbalance_stats_ready(self) -> bool:
        """``True`` once imbalance dispersion is usable."""
        return self._obi_var.ready


__all__ = ["SharedStatistics"]
