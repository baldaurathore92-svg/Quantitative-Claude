"""Pressure momentum: fast minus slow EMA of the order-book imbalance.

Why not "EMA(current) - EMA(previous)"
-------------------------------------
The first difference of a single EMA is not a momentum estimator. Its magnitude
shrinks as the smoothing constant grows, its variance is dominated by the most
recent sample, and its scale depends on the update interval. The standard and
correct construction is the difference between two EMAs of the *same* series with
different half-lives: the fast estimator tracks the current pressure, the slow
one provides the baseline, and their difference is a signed rate of change with a
well-defined time scale.

Both estimators are time-aware, so the half-lives are expressed in milliseconds
of exchange time rather than in snapshot counts. On an event-driven feed those
are very different things.

Normalisation
-------------
The raw momentum is unbounded in principle, so it is scaled by its own recent
dispersion and squashed with ``tanh``. The dispersion is read *before* the
current observation is added to the estimator: normalising a value by a
dispersion that already contains it is a look-ahead into the quantity being
normalised, and it systematically understates unusual readings. A floor on the
dispersion prevents a quiet market from making every micro-fluctuation saturate
the feature.
"""

from __future__ import annotations

from ..buffers.rolling_ema import EMAPair
from ..buffers.rolling_variance import RollingVariance
from ..config import MomentumConfig
from ..utils.constants import F_MOMENTUM, F_WEIGHTED_OBI
from ..utils.math_utils import robust_scale
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext


class MomentumFeature(Feature):
    """Signed rate of change of the weighted order-book imbalance."""

    name = F_MOMENTUM
    kind = FeatureKind.DIRECTIONAL
    depends_on = (F_WEIGHTED_OBI,)

    __slots__ = ("_config", "_dispersion", "_emas")

    def __init__(self, config: MomentumConfig) -> None:
        self._config = config
        self._emas = EMAPair(
            config.fast_half_life_ms,
            config.slow_half_life_ms,
            max_dt_ms=config.max_dt_ms,
            warmup_updates=config.warmup_updates,
        )
        self._dispersion = RollingVariance(
            config.variance_window, min_samples=min(8, config.variance_window)
        )

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the normalised imbalance momentum."""
        imbalance = context.require(F_WEIGHTED_OBI)
        if not imbalance.valid:
            return self._invalid("imbalance unavailable")

        momentum = self._emas.update(imbalance.value, context.snapshot.dt_ms)
        dispersion = self._dispersion.std if self._dispersion.ready else 0.0
        self._dispersion.update(momentum)

        if not self._emas.ready:
            return self._invalid("momentum EMAs still warming up")

        config = self._config
        value = robust_scale(
            momentum, dispersion, config.dispersion_floor, config.dispersion_k
        )
        return self._value(
            raw=momentum,
            value=value,
            detail=(
                f"momentum={momentum:+.3f} fast={self._emas.fast.value:+.3f} "
                f"slow={self._emas.slow.value:+.3f}"
            ),
        )

    def reset(self) -> None:
        """Clear the EMAs and the dispersion estimator."""
        self._emas.reset()
        self._dispersion.reset()

    @property
    def raw_momentum(self) -> float:
        """Current unnormalised momentum, exposed for the acceleration feature."""
        return self._emas.spread


__all__ = ["MomentumFeature"]
