"""Microprice tilt.

Definition
----------
The microprice weights each touch price by the *opposite* queue::

    MP = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)

A heavy bid pushes the fair value towards the ask, which is the intended
behaviour: the side with more resting size is the side less likely to be the one
that has to cross.

Normalisation and an identity worth knowing
-------------------------------------------
The tilt ``MP - mid`` is bounded by half the spread, so dividing by half the
spread yields an exactly bounded value in ``[-1, +1]`` with no clipping and no
arbitrary scale constant. Expanding that ratio gives::

    (MP - mid) / (spread / 2) == (bid_qty - ask_qty) / (bid_qty + ask_qty)

In other words, the normalised microprice tilt **is** the level-one order-book
imbalance — algebraically, not approximately. That has a direct consequence for
the composite score: microprice and the L1/L2 weighted imbalance are strongly
collinear, and if both were given independent weight the engine would count one
observation twice. The composite therefore caps the *joint* weight of this pair
(``composite.collinear_groups``). The two features are still kept separate
because they are not identical: the weighted imbalance also carries level two,
and the microprice carries the spread-scaled interpretation used by the
execution model.

The raw value is reported in ticks, which is the unit an operator can sanity
check against the book.
"""

from __future__ import annotations

from ..buffers.rolling_ema import TimeAwareEMA
from ..config import MicropriceConfig
from ..utils.constants import F_MICROPRICE
from ..utils.math_utils import clamp_unit, safe_div, to_ticks
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext


class MicropriceFeature(Feature):
    """Normalised microprice tilt relative to the mid price."""

    name = F_MICROPRICE
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_config", "_smoother")

    def __init__(self, config: MicropriceConfig) -> None:
        self._config = config
        self._smoother = TimeAwareEMA(
            config.smooth_half_life_ms, max_dt_ms=config.max_dt_ms, warmup_updates=1
        )

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the smoothed, half-spread-normalised microprice tilt."""
        snapshot = context.snapshot
        bid = snapshot.best_bid
        ask = snapshot.best_ask
        queue_total = bid.quantity + ask.quantity
        if queue_total <= 0:
            return self._invalid("empty touch queues")
        half_spread = snapshot.spread * 0.5
        if half_spread <= 0.0:
            return self._invalid("non-positive spread")

        microprice = (bid.price * ask.quantity + ask.price * bid.quantity) / queue_total
        tilt = microprice - snapshot.mid
        tilt_ticks = to_ticks(tilt, snapshot.tick_size)
        normalised = clamp_unit(safe_div(tilt, half_spread))
        smoothed = self._smoother.update(normalised, snapshot.dt_ms)

        return self._value(
            raw=tilt_ticks,
            value=clamp_unit(smoothed),
            detail=f"tilt={tilt_ticks:+.2f}t mp={microprice:.2f}",
        )

    def reset(self) -> None:
        """Clear the smoother."""
        self._smoother.reset()


__all__ = ["MicropriceFeature"]
