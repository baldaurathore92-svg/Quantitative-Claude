"""Relative depth decay from level one to level two.

Construction
------------
For each side, the decay is the log quantity ratio per tick of distance::

    decay = log((q2 + 1) / (q1 + 1)) / distance_in_ticks(L1 -> L2)

The logarithm makes the measure relative (a fall from 10000 to 5000 is the same
as from 200 to 100) and the ``+1`` offset keeps it finite when a level empties,
which happens routinely at the touch. Dividing by the tick distance is essential:
the gap between L1 and L2 is one tick in a liquid book and can be four ticks in a
thin one, and an undivided ratio would confuse "steep" with "far apart".

The directional feature is ``decay_bid - decay_ask``. It is positive when the ask
ladder thins away from the mid faster than the bid ladder does, that is when
there is less size to absorb an upward move than a downward one.

Why level two only for the signal
---------------------------------
The signal uses the L1 to L2 transition because that is the part of the ladder a
few-tick move must actually consume. A two-point estimate is however noisy, so
the deeper levels are used to compute an independent five-point log-linear
regression slope; the *agreement* between the two estimates modulates confidence
and never the score. That keeps the primary signal where the user specified it
while still extracting information from the levels the feed already delivers.
"""

from __future__ import annotations

from ..config import DepthSlopeConfig
from ..utils.constants import F_DEPTH_SLOPE
from ..utils.math_utils import clamp, log_ratio, safe_div, sign, tanh_scale
from ..utils.types import FeatureKind, FeatureValue, Side, Snapshot
from .base import Feature, FeatureContext

#: Minimum tick distance between two levels before the ratio is considered
#: meaningful. Guards against a malformed ladder with duplicated prices.
_MIN_DISTANCE_TICKS = 1e-6


class DepthSlopeFeature(Feature):
    """Asymmetry between the bid-side and ask-side depth decay."""

    name = F_DEPTH_SLOPE
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_config",)

    def __init__(self, config: DepthSlopeConfig) -> None:
        self._config = config

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the normalised bid-minus-ask depth decay asymmetry."""
        snapshot = context.snapshot
        bid_decay = self._pair_decay(snapshot, Side.BID)
        ask_decay = self._pair_decay(snapshot, Side.ASK)
        if bid_decay is None or ask_decay is None:
            return self._invalid("level two missing on at least one side")

        asymmetry = bid_decay - ask_decay
        value = tanh_scale(asymmetry, self._config.scale)
        local_confidence = self._deep_agreement(snapshot, value)

        return self._value(
            raw=asymmetry,
            value=value,
            local_confidence=local_confidence,
            detail=f"bid={bid_decay:+.2f} ask={ask_decay:+.2f}",
        )

    def reset(self) -> None:
        """No internal state: the feature is a pure function of one snapshot."""

    # -- components -------------------------------------------------------- #

    def _pair_decay(self, snapshot: Snapshot, side: Side) -> float | None:
        """Per-tick log decay from level one to level two, or ``None``."""
        levels = snapshot.levels(side)
        if len(levels) < 2:
            return None
        first, second = levels[0], levels[1]
        if first.quantity <= 0 or second.price_paise <= 0:
            return None
        distance_ticks = abs(second.price - first.price) / snapshot.tick_size
        if distance_ticks < _MIN_DISTANCE_TICKS:
            return None
        return log_ratio(float(second.quantity), float(first.quantity)) / distance_ticks

    def _regression_slope(self, snapshot: Snapshot, side: Side) -> float | None:
        """Ordinary-least-squares slope of ``log1p(quantity)`` against ticks.

        Bounded by five iterations, so this is O(1) with a small constant rather
        than an O(N) scan.
        """
        levels = snapshot.levels(side)
        limit = min(self._config.deep_levels, len(levels))
        if limit < 3:
            return None
        touch_price = levels[0].price
        tick = snapshot.tick_size

        count = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        for index in range(limit):
            level = levels[index]
            if level.quantity <= 0 or level.price_paise <= 0:
                break
            x = abs(level.price - touch_price) / tick
            y = log_ratio(float(level.quantity), 0.0, offset=1.0)
            count += 1
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y
        if count < 3:
            return None
        denominator = count * sum_xx - sum_x * sum_x
        if denominator <= 0.0:
            return None
        return safe_div(count * sum_xy - sum_x * sum_y, denominator)

    def _deep_agreement(self, snapshot: Snapshot, primary_value: float) -> float:
        """Confidence factor from the five-level regression estimate.

        Returns ``1.0`` when the deep estimate is unavailable or disabled, so
        that a thin ladder is not penalised twice (the shared confidence model
        already handles low liquidity).
        """
        if not self._config.use_deep_levels:
            return 1.0
        bid_slope = self._regression_slope(snapshot, Side.BID)
        ask_slope = self._regression_slope(snapshot, Side.ASK)
        if bid_slope is None or ask_slope is None:
            return 1.0
        deep_value = tanh_scale(bid_slope - ask_slope, self._config.scale)
        agreement = deep_value * sign(primary_value)
        penalty = (1.0 - clamp(agreement, -1.0, 1.0)) * 0.5
        return clamp(1.0 - 0.4 * penalty, 0.0, 1.0)


__all__ = ["DepthSlopeFeature"]
