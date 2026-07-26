"""Last-traded-price confirmation.

Purpose
-------
The book can lean bullish while trades keep printing lower. Acting on the book
alone in that situation is how a snapshot strategy gets run over. This feature
produces a signed measure of where *executions* have been going, so that the
state machine can refuse an entry whose book signal is contradicted by the tape.

Two components
--------------
1.  **Drift.** The last traded price against its own time-aware EMA, expressed in
    ticks and squashed with ``tanh``. This is the dominant term.
2.  **Position within the spread.** ``(ltp - mid) / (spread / 2)``, bounded in
    ``[-1, +1]``. A print at the offer is weak evidence of buy-side initiative.

    This is emphatically **not** an aggressor classification. Without the trade
    tape and its matching side, the initiator of a trade is unknowable; a print
    at the offer can be a passive sell into an incoming bid improvement, and a
    print at the mid is uninformative. The component therefore carries a small,
    configurable weight (``position_weight``) and is documented as a weak proxy.

Staleness
---------
The last traded price is a *level*, not an event: it persists in the payload long
after the trade occurred. If the day's cumulative volume has not moved for
``stale_ms``, the price refers to an old trade and the feature reports reduced
local confidence rather than pretending to be current. Interval measurement uses
the exchange timestamp, so a slow consumer cannot make a fresh print look stale.
"""

from __future__ import annotations

from ..buffers.rolling_ema import TimeAwareEMA
from ..config import LtpConfirmationConfig
from ..utils.constants import F_LTP_CONFIRMATION
from ..utils.math_utils import clamp_unit, safe_div, tanh_scale, to_ticks
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext

#: Local confidence applied while it is unknown whether any trade has printed.
_UNKNOWN_FRESHNESS_CONFIDENCE = 0.6
#: Local confidence applied once the tape is demonstrably stale.
_STALE_CONFIDENCE = 0.3


class LtpConfirmationFeature(Feature):
    """Signed agreement between traded prices and the current book."""

    name = F_LTP_CONFIRMATION
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_config", "_last_trade_ms", "_reference")

    def __init__(self, config: LtpConfirmationConfig) -> None:
        self._config = config
        self._reference = TimeAwareEMA(
            config.half_life_ms, max_dt_ms=config.max_dt_ms, warmup_updates=2
        )
        self._last_trade_ms: int | None = None

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the normalised tape-versus-book agreement."""
        snapshot = context.snapshot
        config = self._config
        last_price = snapshot.last_traded_price
        if last_price <= 0.0:
            return self._invalid("no traded price yet")

        if snapshot.delta_volume > 0:
            self._last_trade_ms = snapshot.exchange_timestamp_ms

        # The drift is measured against the reference *before* the reference
        # absorbs the current print; otherwise the estimator is being compared
        # with itself and the drift is systematically understated.
        had_reference = self._reference.initialised
        reference_price = self._reference.value
        self._reference.update(last_price, snapshot.dt_ms)
        if not had_reference or not self._reference.ready:
            return self._invalid("traded-price reference still warming up")

        drift_ticks = to_ticks(last_price - reference_price, snapshot.tick_size)
        drift_component = tanh_scale(drift_ticks, config.scale_ticks)

        half_spread = snapshot.spread * 0.5
        position_component = clamp_unit(
            safe_div(last_price - snapshot.mid, half_spread)
        )

        weight = config.position_weight
        value = clamp_unit(
            (1.0 - weight) * drift_component + weight * position_component
        )

        local_confidence, freshness = self._freshness(snapshot.exchange_timestamp_ms)
        return self._value(
            raw=drift_ticks,
            value=value,
            local_confidence=local_confidence,
            detail=(
                f"drift={drift_ticks:+.2f}t pos={position_component:+.2f} {freshness}"
            ),
        )

    def reset(self) -> None:
        """Clear the reference price and the freshness tracker."""
        self._reference.reset()
        self._last_trade_ms = None

    # -- components -------------------------------------------------------- #

    def _freshness(self, now_ms: int) -> tuple[float, str]:
        """Return the local confidence and a label describing tape freshness."""
        if self._last_trade_ms is None:
            return _UNKNOWN_FRESHNESS_CONFIDENCE, "tape=unknown"
        age_ms = float(now_ms - self._last_trade_ms)
        if age_ms > self._config.stale_ms:
            return _STALE_CONFIDENCE, f"tape=stale({age_ms:.0f}ms)"
        return 1.0, "tape=fresh"

    def opposes(self, direction_sign: int, value: float) -> bool:
        """Return ``True`` when the tape materially contradicts ``direction_sign``.

        Exposed as a method so that the veto rule lives with the feature that
        defines it, while the state machine remains responsible for *acting* on
        the veto.
        """
        if direction_sign == 0:
            return False
        threshold = self._config.opposition_threshold
        if direction_sign > 0:
            return value <= -threshold
        return value >= threshold


__all__ = ["LtpConfirmationFeature"]
