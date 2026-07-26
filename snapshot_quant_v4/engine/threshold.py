"""Adaptive signal threshold.

Construction
------------
::

    raw = base * (1 + a * spread_dispersion
                    + b * mid_dispersion
                    + c * imbalance_dispersion
                    + d * instability)
    entry = clamp(raw, min_entry, max_entry)
    watch = entry * watch_ratio
    exit  = entry * exit_ratio

Because the composite score is normalised into ``[-1, +1]``, the threshold lives
in the same units and can be reasoned about directly. There is no hardcoded
magic number such as 3.5 anywhere: that value belongs to an unnormalised score
and is meaningless here.

Each dispersion term is made dimensionless before use:

*   spread and mid dispersion are divided by the instrument's own mean spread in
    ticks, so a two-tick swing means the same thing on every instrument;
*   imbalance dispersion is divided by ``imbalance_vol_reference``, a fraction of
    the imbalance's own ``[-1, +1]`` range;
*   instability is ``1 - touch_stability``, already dimensionless.

Why the clamp is mandatory
--------------------------
Every dispersion term tends to zero in a quiet book. An unclamped adaptive
threshold therefore converges towards ``base * 1`` at best and, with any
multiplicative formulation, towards zero at worst — precisely when the market is
least informative. ``min_entry`` is the structural floor that keeps the engine
from trading numerical dust; ``max_entry`` prevents a single volatility spike
from disabling the engine for the rest of the session.

Warmup behaviour is explicit: until the dispersion estimators are ready, the
threshold is pinned to ``max_entry``. A threshold derived from a half-filled
variance estimator is not conservative, it is arbitrary, so the engine refuses to
trade rather than guessing.

Cost coupling
-------------
The threshold deliberately knows nothing about transaction costs. Cost gating
lives in :mod:`engine.execution` and is optional, so the signal path behaves
identically whether or not a cost model is configured.
"""

from __future__ import annotations

from ..config import ThresholdConfig
from ..utils.math_utils import clamp, safe_div
from ..utils.types import ThresholdResult
from .stats import SharedStatistics


class AdaptiveThreshold:
    """Computes entry, watch and exit thresholds from market dispersion.

    Stateless with respect to history: all inputs come from
    :class:`~engine.stats.SharedStatistics`, so a single instance could be shared
    between instruments. One per instrument is still used for symmetry with the
    other engine components.

    Parameters
    ----------
    config:
        Threshold coefficients and clamps.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ThresholdConfig) -> None:
        self._config = config

    def compute(self, stats: SharedStatistics) -> ThresholdResult:
        """Return the current thresholds.

        Parameters
        ----------
        stats:
            Shared statistics, already updated for this snapshot.
        """
        config = self._config
        if not (stats.spread_stats_ready and stats.mid_volatility_ready):
            return self._build(
                config.max_entry,
                floor_applied=False,
                detail="dispersion estimators warming up; threshold pinned to max",
            )

        spread_reference = max(stats.spread_mean_ticks, 1.0)
        spread_dispersion = safe_div(stats.spread_volatility_ticks, spread_reference)
        mid_dispersion = safe_div(stats.mid_volatility_ticks, spread_reference)
        imbalance_dispersion = (
            safe_div(stats.imbalance_volatility, config.imbalance_vol_reference)
            if stats.imbalance_stats_ready
            else 0.0
        )
        instability = clamp(1.0 - stats.touch_stability, 0.0, 1.0)

        multiplier = (
            1.0
            + config.spread_vol_coeff * spread_dispersion
            + config.mid_vol_coeff * mid_dispersion
            + config.obi_vol_coeff * imbalance_dispersion
            + config.instability_coeff * instability
        )
        raw = config.base * multiplier
        entry = clamp(raw, config.min_entry, config.max_entry)
        return self._build(
            entry,
            floor_applied=raw < config.min_entry,
            detail=(
                f"raw={raw:.3f} spr={spread_dispersion:.2f} mid={mid_dispersion:.2f} "
                f"obi={imbalance_dispersion:.2f} instab={instability:.2f}"
            ),
        )

    def _build(
        self, entry: float, *, floor_applied: bool, detail: str
    ) -> ThresholdResult:
        """Derive the watch and exit levels from an entry threshold."""
        config = self._config
        return ThresholdResult(
            entry=entry,
            watch=entry * config.watch_ratio,
            exit=entry * config.exit_ratio,
            floor_applied=floor_applied,
            detail=detail,
        )


__all__ = ["AdaptiveThreshold"]
