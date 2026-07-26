"""Spread width as a market-quality feature.

The spread has a magnitude but no direction: a four-tick spread is not bullish or
bearish. Feeding it into a signed composite score would be a category error, so
it is classified as :attr:`~snapshot_quant_v4.utils.types.FeatureKind.QUALITY`
and is consumed by the confidence model rather than by the score.

The value is a *tightness* score in ``[0, 1]``: one tick or better maps to 1.0,
``wide_ticks`` or worse maps to 0.0, linearly in between. Expressing it in ticks
is what makes a single configuration valid across instruments; a basis-point
threshold would be four times stricter on a 600 rupee stock than on a 150 rupee
stock for the same one-tick spread.

The raw value is the spread in ticks, and the feature additionally reports its
own recent dispersion so that an operator can see whether a wide spread is
unusual for the instrument or simply normal.
"""

from __future__ import annotations

from ..buffers.rolling_variance import RollingVariance
from ..config import SpreadConfig
from ..utils.constants import F_SPREAD
from ..utils.math_utils import linear_scale
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext


class SpreadFeature(Feature):
    """Normalised spread tightness in ``[0, 1]``."""

    name = F_SPREAD
    kind = FeatureKind.QUALITY

    __slots__ = ("_config", "_dispersion")

    def __init__(self, config: SpreadConfig) -> None:
        self._config = config
        self._dispersion = RollingVariance(
            config.variance_window, min_samples=min(8, config.variance_window)
        )

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the tightness score for the current spread."""
        spread_ticks = context.snapshot.spread_ticks
        config = self._config
        # Dispersion is read before the update so that the reported statistic
        # describes the recent past rather than including the observation being
        # described.
        dispersion = self._dispersion.std if self._dispersion.ready else 0.0
        self._dispersion.update(spread_ticks)

        tightness = linear_scale(-spread_ticks, -config.wide_ticks, -config.tight_ticks)
        return self._value(
            raw=spread_ticks,
            value=tightness,
            detail=f"spread={spread_ticks:.2f}t sd={dispersion:.2f}t",
        )

    def reset(self) -> None:
        """Clear the dispersion estimator."""
        self._dispersion.reset()


__all__ = ["SpreadFeature"]
