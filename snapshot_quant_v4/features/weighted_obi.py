"""Distance-weighted order-book imbalance.

Weighting
---------
Each level's quantity is weighted by its distance from the mid, measured **in
ticks**::

    distance_ticks = |level_price - mid| / tick_size
    weight         = exp(-distance_ticks / decay_ticks)

Measuring the distance in ticks rather than in rupees is not a stylistic choice.
A rupee-denominated decay constant of the form ``mid * 0.005`` evaluates to
roughly 250 ticks on a 2500 rupee instrument, while the five published levels of
a liquid book span perhaps five ticks. Every weight then rounds to
``exp(-0.02) ~ 0.98``, the weighting collapses into a plain sum, and the same
formula behaves completely differently on a 150 rupee instrument. The tick form
is scale-free by construction: two ticks away is two ticks away at any price.

Primary versus secondary levels
-------------------------------
The directional signal comes from the primary levels only (L1 and L2 by
default). This is a deliberate restriction: those are the levels that a
few-tick move actually has to consume.

The deeper levels and the exchange-published whole-book totals
(``total_buy_quantity`` / ``total_sell_quantity``) are used **only to modulate
confidence**, never to move the score:

*   deeper levels agreeing with L1/L2 means the imbalance is structural rather
    than a single posting at the touch;
*   the whole-book totals are genuine exchange data rather than an estimate, but
    they include orders parked far from the market that can sit untouched all
    day. They are therefore a weak corroborator, not a signal, and they are
    ignored entirely when the two totals differ by more than
    ``aggregate_max_ratio``, which is the signature of one-sided junk depth.

This is not "full order-book imbalance": the per-level book beyond five levels
is not published by a snapshot feed and is not estimated anywhere.
"""

from __future__ import annotations

import math

from ..config import WeightedObiConfig
from ..utils.constants import F_WEIGHTED_OBI
from ..utils.math_utils import clamp, clamp_unit, safe_div
from ..utils.types import FeatureKind, FeatureValue, Side, Snapshot
from .base import Feature, FeatureContext


class WeightedObiFeature(Feature):
    """Exponentially distance-weighted imbalance over the primary levels."""

    name = F_WEIGHTED_OBI
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_config",)

    def __init__(self, config: WeightedObiConfig) -> None:
        self._config = config

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the primary imbalance, with deeper data folded into confidence."""
        snapshot = context.snapshot
        config = self._config

        bid_mass, ask_mass = self._weighted_mass(snapshot, 0, config.primary_levels)
        total = bid_mass + ask_mass
        if total <= 0.0:
            return self._invalid("no weighted mass on the primary levels")
        imbalance = clamp_unit((bid_mass - ask_mass) / total)

        deep_agreement = self._deep_agreement(snapshot, imbalance)
        aggregate_agreement = self._aggregate_agreement(snapshot, imbalance)
        local_confidence = self._blend_agreement(deep_agreement, aggregate_agreement)

        return self._value(
            raw=imbalance,
            value=imbalance,
            local_confidence=local_confidence,
            detail=(
                f"obi={imbalance:+.3f} deep={self._format(deep_agreement)} "
                f"agg={self._format(aggregate_agreement)}"
            ),
        )

    def reset(self) -> None:
        """No internal state: the feature is a pure function of one snapshot."""

    # -- components -------------------------------------------------------- #

    def _weighted_mass(
        self, snapshot: Snapshot, start_level: int, end_level: int
    ) -> tuple[float, float]:
        """Return ``(bid_mass, ask_mass)`` for levels ``[start_level, end_level)``."""
        decay = self._config.decay_ticks
        tick = snapshot.tick_size
        mid = snapshot.mid
        bid_mass = 0.0
        ask_mass = 0.0

        for index, level in enumerate(snapshot.levels(Side.BID)):
            if index < start_level:
                continue
            if index >= end_level:
                break
            if level.quantity <= 0 or level.price_paise <= 0:
                break
            distance_ticks = (mid - level.price) / tick
            bid_mass += level.quantity * math.exp(-abs(distance_ticks) / decay)

        for index, level in enumerate(snapshot.levels(Side.ASK)):
            if index < start_level:
                continue
            if index >= end_level:
                break
            if level.quantity <= 0 or level.price_paise <= 0:
                break
            distance_ticks = (level.price - mid) / tick
            ask_mass += level.quantity * math.exp(-abs(distance_ticks) / decay)

        return bid_mass, ask_mass

    def _deep_agreement(self, snapshot: Snapshot, primary: float) -> float | None:
        """Signed agreement of levels beyond the primary window, in ``[-1, +1]``.

        Returns ``None`` when there is no usable depth beyond the primary levels,
        which is common in thin instruments and must not be read as
        disagreement.
        """
        config = self._config
        if config.deep_levels <= config.primary_levels:
            return None
        bid_mass, ask_mass = self._weighted_mass(
            snapshot, config.primary_levels, config.deep_levels
        )
        total = bid_mass + ask_mass
        if total <= 0.0:
            return None
        deep = clamp_unit((bid_mass - ask_mass) / total)
        return deep if primary >= 0.0 else -deep

    def _aggregate_agreement(self, snapshot: Snapshot, primary: float) -> float | None:
        """Signed agreement of the whole-book totals, in ``[-1, +1]``.

        Returns ``None`` when the totals are unavailable or so lopsided that they
        are dominated by depth parked far from the market.
        """
        buy = snapshot.total_buy_quantity
        sell = snapshot.total_sell_quantity
        if buy <= 0.0 or sell <= 0.0:
            return None
        ratio = max(buy, sell) / min(buy, sell)
        if ratio > self._config.aggregate_max_ratio:
            return None
        aggregate = clamp_unit(safe_div(buy - sell, buy + sell))
        return aggregate if primary >= 0.0 else -aggregate

    def _blend_agreement(
        self, deep_agreement: float | None, aggregate_agreement: float | None
    ) -> float:
        """Convert agreement values into a multiplicative confidence factor.

        Perfect agreement leaves confidence untouched; perfect disagreement
        removes the configured weight of that corroborator. The mapping is
        linear in the agreement and stays in ``[1 - w, 1]``, so a corroborator
        can never inflate confidence above what the shared model assigned.
        """
        config = self._config
        factor = 1.0
        if deep_agreement is not None:
            penalty = (1.0 - deep_agreement) * 0.5
            factor *= 1.0 - config.deep_agreement_weight * penalty
        if aggregate_agreement is not None:
            penalty = (1.0 - aggregate_agreement) * 0.5
            factor *= 1.0 - config.aggregate_agreement_weight * penalty
        return clamp(factor, 0.0, 1.0)

    @staticmethod
    def _format(agreement: float | None) -> str:
        """Format an optional agreement value for the console."""
        return "n/a" if agreement is None else f"{agreement:+.2f}"


__all__ = ["WeightedObiFeature"]
