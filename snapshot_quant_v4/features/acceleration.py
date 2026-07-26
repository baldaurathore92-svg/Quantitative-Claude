"""Pressure acceleration: the time derivative of momentum.

The derivative is taken with respect to *time*, not with respect to snapshot
count::

    acceleration = (momentum_now - momentum_previous) / seconds_elapsed

Dividing by the snapshot index instead would make the feature depend on the
feed's update rate: the same market behaviour would register as a large
acceleration during a quiet minute (few updates, large per-update change) and a
small one during a busy minute. Since the raw momentum comes from time-aware
EMAs, dividing by elapsed seconds keeps the whole chain dimensionally consistent
and yields a value in "imbalance units per second".

A second difference of a smoothed series is inherently noisy, so the derivative
is smoothed once with a short time-aware EMA before normalisation. The
normalisation follows the same rule as momentum: scale by the feature's own
recent dispersion, read before the current sample is added, with a floor.

The interval used for the derivative is clamped below by ``min_dt_ms``. Without
that clamp two snapshots arriving in the same millisecond would divide by
approximately zero and produce an unbounded spike.
"""

from __future__ import annotations

from ..buffers.rolling_ema import TimeAwareEMA
from ..buffers.rolling_variance import RollingVariance
from ..config import AccelerationConfig
from ..utils.constants import F_ACCELERATION, F_MOMENTUM
from ..utils.math_utils import robust_scale
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext


class AccelerationFeature(Feature):
    """Smoothed, normalised time derivative of pressure momentum."""

    name = F_ACCELERATION
    kind = FeatureKind.DIRECTIONAL
    depends_on = (F_MOMENTUM,)

    __slots__ = ("_config", "_dispersion", "_previous_momentum", "_smoother")

    def __init__(self, config: AccelerationConfig) -> None:
        self._config = config
        self._smoother = TimeAwareEMA(
            config.smooth_half_life_ms, max_dt_ms=config.max_dt_ms, warmup_updates=3
        )
        self._dispersion = RollingVariance(
            config.variance_window, min_samples=min(8, config.variance_window)
        )
        self._previous_momentum: float | None = None

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the normalised acceleration of the imbalance."""
        momentum_feature = context.require(F_MOMENTUM)
        if not momentum_feature.valid:
            self._previous_momentum = None
            return self._invalid("momentum unavailable")

        momentum = momentum_feature.raw
        previous = self._previous_momentum
        self._previous_momentum = momentum
        if previous is None:
            return self._invalid("no previous momentum")

        config = self._config
        dt_ms = context.snapshot.dt_ms
        if dt_ms < config.min_dt_ms:
            dt_ms = config.min_dt_ms
        derivative = (momentum - previous) / (dt_ms / 1000.0)
        smoothed = self._smoother.update(derivative, context.snapshot.dt_ms)

        dispersion = self._dispersion.std if self._dispersion.ready else 0.0
        self._dispersion.update(smoothed)

        if not self._smoother.ready:
            return self._invalid("acceleration smoother still warming up")

        value = robust_scale(
            smoothed, dispersion, config.dispersion_floor, config.dispersion_k
        )
        return self._value(
            raw=smoothed,
            value=value,
            detail=f"accel={smoothed:+.3f}/s raw={derivative:+.3f}/s",
        )

    def reset(self) -> None:
        """Clear the smoother, the dispersion estimator and the previous value."""
        self._smoother.reset()
        self._dispersion.reset()
        self._previous_momentum = None


__all__ = ["AccelerationFeature"]
