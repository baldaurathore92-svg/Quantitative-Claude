"""Refill proxy: trade-confirmed consumption followed by replenishment.

What this is not
----------------
This is **not** iceberg detection. Hidden quantity is not published by any retail
feed, and a snapshot cannot distinguish a genuine reserve order from a
market maker who simply reposts quickly. What is observable is a two-step
pattern: a price level loses a material share of its size *while trades print*,
and then that same price recovers its size. That pattern is what this feature
scores, and it is scored as a proxy, not as a detection.

The three ideas that make it defensible
---------------------------------------
1.  **Price-keyed tracking.** Both the consumption and the recovery are measured
    at one fixed price, looked up by
    :meth:`~snapshot_quant_v4.utils.types.Snapshot.level_at`. Comparing ladder
    index zero across snapshots would confuse a one-tick move of the touch with
    a size collapse.

2.  **Volume confirmation separates consumption from cancellation.** A quantity
    drop at the touch has two completely different causes with opposite
    information content: the size was *traded* (aggressive flow, and the refill
    that follows is real support) or the size was *cancelled* (liquidity fading
    away). The only snapshot-observable discriminator is the change in the day's
    cumulative traded volume. A drop is accepted as consumption only when
    ``delta_volume`` explains at least ``volume_confirmation_ratio`` of it.

    Honest limitation: ``delta_volume`` is not attributable to a side. It
    confirms that trades of a comparable size occurred in the interval, not that
    they occurred against this particular queue. Snapshot data cannot do better,
    and this feature does not pretend otherwise.

3.  **Growth alone is never an event.** A level that simply grows produces
    nothing. Without a preceding, volume-confirmed consumption there is no
    pending state, so there is nothing to refill. This is the specific failure
    mode that makes naive "refill" implementations fire continuously in a
    thickening book.

Scoring
-------
Each resolved episode emits an impulse into a per-side score that decays
exponentially in exchange time: a successful refill adds a positive impulse
proportional to how completely the level recovered, while an episode that expires
or whose price disappears adds a negative impulse (consumed and *not*
replenished, which is weakness rather than absence of information). The
directional value is ``bid_score - ask_score``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import RefillConfig
from ..utils.constants import F_REFILL, LN2
from ..utils.math_utils import clamp, clamp_unit
from ..utils.types import FeatureKind, FeatureValue, Side, Snapshot
from .base import Feature, FeatureContext

#: Bound on the accumulated per-side score, so a burst of episodes cannot
#: saturate the feature for an unbounded period.
_SCORE_LIMIT = 1.5


@dataclass(slots=True)
class _Episode:
    """A volume-confirmed consumption awaiting resolution."""

    price_paise: int
    quantity_before: int
    drop: int
    started_ms: int


class _DecayingScore:
    """An impulse accumulator that decays exponentially in exchange time.

    A plain EMA is unsuitable here: episodes are discrete, sparse events, and
    blending an impulse of 1.0 into an EMA with a small alpha would register
    almost nothing. This structure instead adds the impulse directly and then
    decays it, which gives the intended "recent refill activity" semantics with
    a single multiply per snapshot.
    """

    __slots__ = ("_half_life_ms", "_max_dt_ms", "_value")

    def __init__(self, half_life_ms: float, max_dt_ms: float) -> None:
        if half_life_ms <= 0.0:
            raise ValueError("half_life_ms must be positive")
        self._half_life_ms = float(half_life_ms)
        self._max_dt_ms = float(max_dt_ms)
        self._value = 0.0

    @property
    def value(self) -> float:
        """Current accumulated score."""
        return self._value

    def decay(self, dt_ms: float) -> None:
        """Apply exponential decay for an elapsed interval."""
        if self._value == 0.0:
            return
        effective = clamp(dt_ms, 0.0, self._max_dt_ms)
        if effective <= 0.0:
            return
        self._value *= math.exp(-LN2 * effective / self._half_life_ms)

    def add(self, impulse: float) -> None:
        """Add an impulse, keeping the accumulator bounded."""
        self._value = clamp(self._value + impulse, -_SCORE_LIMIT, _SCORE_LIMIT)

    def reset(self) -> None:
        """Clear the accumulator."""
        self._value = 0.0


class RefillFeature(Feature):
    """Scores volume-confirmed consumption followed by replenishment."""

    name = F_REFILL
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_config", "_episodes", "_last_detail", "_scores")

    def __init__(self, config: RefillConfig) -> None:
        self._config = config
        self._scores: dict[Side, _DecayingScore] = {
            Side.BID: _DecayingScore(config.decay_half_life_ms, config.max_dt_ms),
            Side.ASK: _DecayingScore(config.decay_half_life_ms, config.max_dt_ms),
        }
        self._episodes: dict[Side, _Episode | None] = {Side.BID: None, Side.ASK: None}
        self._last_detail = ""

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the bid-minus-ask refill score."""
        previous = context.previous
        snapshot = context.snapshot
        if previous is None:
            # A gap or the first snapshot: any episode in flight was tracked
            # against a book state we can no longer trust.
            self._episodes[Side.BID] = None
            self._episodes[Side.ASK] = None
            return self._invalid("no previous snapshot")

        events: list[str] = []
        for side in (Side.BID, Side.ASK):
            score = self._scores[side]
            score.decay(snapshot.dt_ms)
            impulse, label = self._process_side(side, snapshot, previous)
            if impulse != 0.0:
                score.add(impulse)
                events.append(f"{side.value.lower()}:{label}")

        bid_score = self._scores[Side.BID].value
        ask_score = self._scores[Side.ASK].value
        raw = bid_score - ask_score
        if events:
            self._last_detail = " ".join(events)

        return self._value(
            raw=raw,
            value=clamp_unit(raw),
            detail=(
                f"bid={bid_score:+.2f} ask={ask_score:+.2f}"
                + (f" [{self._last_detail}]" if self._last_detail else "")
            ),
        )

    def reset(self) -> None:
        """Clear scores and any in-flight episodes."""
        for score in self._scores.values():
            score.reset()
        self._episodes[Side.BID] = None
        self._episodes[Side.ASK] = None
        self._last_detail = ""

    # -- components -------------------------------------------------------- #

    def _process_side(
        self, side: Side, snapshot: Snapshot, previous: Snapshot
    ) -> tuple[float, str]:
        """Advance the state machine for one side.

        Returns
        -------
        tuple[float, str]
            The impulse to add (zero when nothing resolved) and a short label
            describing the event for the console.
        """
        episode = self._episodes[side]
        if episode is not None:
            return self._resolve(side, episode, snapshot)
        self._detect(side, snapshot, previous)
        return 0.0, ""

    def _resolve(
        self, side: Side, episode: _Episode, snapshot: Snapshot
    ) -> tuple[float, str]:
        """Check whether an in-flight episode has completed, failed or expired."""
        config = self._config
        level = snapshot.level_at(side, episode.price_paise)
        age_ms = float(snapshot.exchange_timestamp_ms - episode.started_ms)

        if level is None:
            self._episodes[side] = None
            return -config.failure_weight, "price-gone"

        recovery_target = episode.quantity_before * config.recovery_fraction
        if level.quantity >= recovery_target:
            self._episodes[side] = None
            completeness = min(
                1.0, level.quantity / max(episode.quantity_before, 1)
            )
            return completeness, f"refilled@{episode.price_paise / 100.0:.2f}"

        if age_ms > config.window_ms:
            self._episodes[side] = None
            return -config.failure_weight, "no-refill"

        return 0.0, ""

    def _detect(self, side: Side, snapshot: Snapshot, previous: Snapshot) -> None:
        """Look for a new, volume-confirmed consumption at the previous touch."""
        config = self._config
        previous_level = previous.best_bid if side is Side.BID else previous.best_ask
        quantity_before = previous_level.quantity
        if quantity_before <= 0:
            return

        current = snapshot.level_at(side, previous_level.price_paise)
        current_quantity = current.quantity if current is not None else 0
        drop = quantity_before - current_quantity
        if drop < quantity_before * config.consumption_fraction:
            return

        # The discriminator: were there trades large enough to explain the drop?
        # If not, the size was cancelled rather than traded, and a subsequent
        # reappearance carries no information about genuine support.
        required_volume = drop * config.volume_confirmation_ratio
        if snapshot.delta_volume < required_volume:
            return

        self._episodes[side] = _Episode(
            price_paise=previous_level.price_paise,
            quantity_before=quantity_before,
            drop=drop,
            started_ms=snapshot.exchange_timestamp_ms,
        )


__all__ = ["RefillFeature"]
