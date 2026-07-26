"""Trading state machine.

States
------
``WARMUP`` -> ``NEUTRAL`` -> ``WATCH_LONG`` / ``WATCH_SHORT`` -> ``LONG`` /
``SHORT`` -> ``EXIT_LONG`` / ``EXIT_SHORT`` -> ``COOLDOWN`` -> ``NEUTRAL``

Entry requires *all* of the following, deliberately conjunctive:

*   the smoothed composite score exceeds the adaptive entry threshold;
*   aggregate confidence reaches ``min_entry_confidence``;
*   the last-traded-price feature does not contradict the direction;
*   the market-quality gate is open;
*   the conditions hold for ``entry_confirmations`` consecutive snapshots;
*   the direction is re-armed, meaning the score has crossed zero since the last
    exit in that direction.

Exit occurs on the first of: stop, target, composite reversal beyond the exit
threshold, confidence collapse, maximum holding time, or a collapse in market
quality (feed gap, spread blow-out, liquidity failure).

Design notes
------------
**All timing is monotonic.** Timeouts use an injected
:class:`~snapshot_quant_v4.utils.clock.Clock` rather than exchange timestamps, so
a stalled or replayed feed cannot make a two second watch window last an hour.
Tests drive it with ``ManualClock``.

**The re-arm rule prevents churn.** Without it, a score oscillating around the
threshold re-enters the same direction immediately after every exit. The
requirement that the score pass back through zero is a deterministic,
parameter-free way to demand that the market actually changed its mind.

**Confirmation counting is on consecutive snapshots.** A single snapshot at the
threshold is frequently a transient artefact of one large order arriving and
being cancelled.

**The stop is evaluated on the exit side of the book**, delegated to the
execution model. Evaluating a stop on the mid price reports fills that a real
order would not have received.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LtpConfirmationConfig, StateMachineConfig
from ..utils.clock import Clock
from ..utils.constants import F_LTP_CONFIRMATION
from ..utils.logging_utils import get_logger
from ..utils.math_utils import sign
from ..utils.types import (
    BlockReason,
    CompositeResult,
    Direction,
    FeatureMap,
    FillQuote,
    PnLReport,
    Position,
    QualityReport,
    Snapshot,
    StateTransition,
    ThresholdResult,
    TradeState,
)
from .execution import ExecutionModel

_LOGGER = get_logger(__name__)

#: Quality failures severe enough to close an open position rather than merely
#: block new entries.
_CRITICAL_BLOCKS: frozenset[BlockReason] = frozenset(
    {
        BlockReason.FEED_GAP,
        BlockReason.PRICE_GAP,
        BlockReason.SPREAD_TOO_WIDE,
        BlockReason.LIQUIDITY_BELOW_THRESHOLD,
        BlockReason.BOOK_TOO_THIN,
    }
)


@dataclass(frozen=True, slots=True)
class StateMachineOutput:
    """Result of one state-machine update."""

    state: TradeState
    transition: StateTransition
    position: Position | None
    pnl: PnLReport | None
    entry_quote: FillQuote | None
    exit_quote: FillQuote | None
    reasons: tuple[str, ...]


class TradingStateMachine:
    """Deterministic entry and exit logic for one instrument.

    Parameters
    ----------
    config:
        Entry, exit and timing rules.
    ltp_config:
        Supplies the opposition threshold used by the tape veto.
    execution:
        Injected execution model, used for quotes, stops and mark-to-market.
    clock:
        Injected clock, so timeouts are testable.
    """

    __slots__ = (
        "_clock",
        "_config",
        "_confirmation_sign",
        "_confirmations",
        "_cooldown_until_ms",
        "_execution",
        "_ltp_config",
        "_position",
        "_quantity",
        "_rearm_block_sign",
        "_state",
        "_watch_deadline_ms",
    )

    def __init__(
        self,
        config: StateMachineConfig,
        ltp_config: LtpConfirmationConfig,
        execution: ExecutionModel,
        clock: Clock,
        *,
        quantity: int = 1,
    ) -> None:
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        self._config = config
        self._ltp_config = ltp_config
        self._execution = execution
        self._clock = clock
        self._quantity = int(quantity)
        self._state = TradeState.WARMUP
        self._position: Position | None = None
        self._watch_deadline_ms = 0.0
        self._confirmations = 0
        self._confirmation_sign = 0
        self._cooldown_until_ms = 0.0
        self._rearm_block_sign = 0

    # -- state ------------------------------------------------------------- #

    @property
    def state(self) -> TradeState:
        """Current state."""
        return self._state

    @property
    def position(self) -> Position | None:
        """The open position, or ``None``."""
        return self._position

    def reset(self) -> None:
        """Return to ``WARMUP`` and drop all counters.

        Any open position must be closed by the caller *before* resetting, via
        :meth:`force_flat`; this method does not silently discard one.
        """
        self._state = TradeState.WARMUP
        self._watch_deadline_ms = 0.0
        self._confirmations = 0
        self._confirmation_sign = 0
        self._cooldown_until_ms = 0.0
        self._rearm_block_sign = 0

    # -- main transition --------------------------------------------------- #

    def update(
        self,
        *,
        snapshot: Snapshot,
        composite: CompositeResult,
        threshold: ThresholdResult,
        quality: QualityReport,
        features: FeatureMap,
    ) -> StateMachineOutput:
        """Advance the machine by one snapshot."""
        now_ms = self._clock.monotonic_ms()
        previous_state = self._state
        score = composite.smoothed if composite.valid else 0.0
        confidence = composite.confidence if composite.valid else 0.0
        self._update_rearm(score)

        if self._state in (TradeState.LONG, TradeState.SHORT):
            return self._handle_open(
                snapshot=snapshot,
                previous_state=previous_state,
                score=score,
                confidence=confidence,
                threshold=threshold,
                quality=quality,
                now_ms=now_ms,
            )

        if self._state in (TradeState.EXIT_LONG, TradeState.EXIT_SHORT):
            return self._handle_post_exit(previous_state, now_ms)

        if self._state is TradeState.COOLDOWN:
            if now_ms >= self._cooldown_until_ms:
                return self._transition(
                    previous_state, TradeState.NEUTRAL, ("cooldown elapsed",)
                )
            return self._stay(previous_state, ("cooldown active",))

        if self._state is TradeState.WARMUP:
            if BlockReason.WARMUP in quality.reasons:
                return self._stay(previous_state, ("warming up",))
            return self._transition(
                previous_state, TradeState.NEUTRAL, ("warmup complete",)
            )

        return self._handle_flat(
            snapshot=snapshot,
            previous_state=previous_state,
            composite=composite,
            score=score,
            confidence=confidence,
            threshold=threshold,
            quality=quality,
            features=features,
            now_ms=now_ms,
        )

    def force_flat(
        self, snapshot: Snapshot, reason: str
    ) -> StateMachineOutput:
        """Close any open position immediately and reset to ``WARMUP``.

        Called by the engine on a feed gap or a reconnect. Holding a position
        whose supporting statistics have just been invalidated is not a
        defensible state, so the position is closed at the current book and the
        machine restarts its warmup.
        """
        previous_state = self._state
        now_ms = self._clock.monotonic_ms()
        position = self._position
        pnl = None
        exit_quote = None
        if position is not None and position.is_open:
            exit_quote, pnl = self._execution.mark_to_market(
                position, snapshot, monotonic_ms=now_ms
            )
            _LOGGER.warning(
                "%s: forced flat (%s) at %.2f, net %.2f",
                snapshot.symbol,
                reason,
                exit_quote.price,
                pnl.net_rupees,
            )
        self._position = None
        self.reset()
        transition = StateTransition(
            previous=previous_state,
            current=TradeState.WARMUP,
            reasons=(f"forced flat: {reason}",),
        )
        return StateMachineOutput(
            state=TradeState.WARMUP,
            transition=transition,
            position=None,
            pnl=pnl,
            entry_quote=None,
            exit_quote=exit_quote,
            reasons=transition.reasons,
        )

    # -- open position ----------------------------------------------------- #

    def _handle_open(
        self,
        *,
        snapshot: Snapshot,
        previous_state: TradeState,
        score: float,
        confidence: float,
        threshold: ThresholdResult,
        quality: QualityReport,
        now_ms: float,
    ) -> StateMachineOutput:
        """Evaluate exit conditions for an open position."""
        position = self._position
        if position is None or not position.is_open:
            # Defensive: the state says a position is open but none exists. Fail
            # towards flat rather than towards trading.
            _LOGGER.error(
                "%s: state %s without a position; forcing NEUTRAL",
                snapshot.symbol,
                self._state,
            )
            self._state = TradeState.NEUTRAL
            return self._stay(previous_state, ("inconsistent position state",))

        exit_reason = self._exit_reason(
            position=position,
            snapshot=snapshot,
            score=score,
            confidence=confidence,
            threshold=threshold,
            quality=quality,
            now_ms=now_ms,
        )
        exit_quote, pnl = self._execution.mark_to_market(
            position, snapshot, monotonic_ms=now_ms
        )
        if exit_reason is None:
            return StateMachineOutput(
                state=self._state,
                transition=StateTransition(previous_state, self._state, ()),
                position=position,
                pnl=pnl,
                entry_quote=position.entry_quote,
                exit_quote=exit_quote,
                reasons=(f"holding {self._execution.unrealised_ticks(position, snapshot):+.1f}t",),
            )

        exit_state = (
            TradeState.EXIT_LONG
            if position.direction is Direction.LONG
            else TradeState.EXIT_SHORT
        )
        self._state = exit_state
        self._cooldown_until_ms = now_ms + self._config.cooldown_ms
        self._rearm_block_sign = position.direction.signum
        self._position = None
        self._confirmations = 0
        self._confirmation_sign = 0
        reasons = (f"exit: {exit_reason}", f"net {pnl.net_rupees:+.2f}")
        _LOGGER.info(
            "%s: %s exit (%s) entry=%.2f exit=%.2f gross=%.1ft net=%.2f",
            snapshot.symbol,
            position.direction.name,
            exit_reason,
            position.entry_price,
            exit_quote.price,
            pnl.gross_ticks,
            pnl.net_rupees,
        )
        return StateMachineOutput(
            state=exit_state,
            transition=StateTransition(previous_state, exit_state, reasons),
            position=None,
            pnl=pnl,
            entry_quote=position.entry_quote,
            exit_quote=exit_quote,
            reasons=reasons,
        )

    def _exit_reason(
        self,
        *,
        position: Position,
        snapshot: Snapshot,
        score: float,
        confidence: float,
        threshold: ThresholdResult,
        quality: QualityReport,
        now_ms: float,
    ) -> str | None:
        """Return the first applicable exit reason, or ``None`` to keep holding.

        The stop is checked before the minimum holding time: a minimum hold is a
        protection against noise, not a licence to sit through a stop.
        """
        config = self._config
        if self._execution.stop_hit(position, snapshot):
            return "stop"

        holding_ms = now_ms - position.entry_monotonic_ms
        if holding_ms < config.min_hold_ms:
            return None

        if config.use_target and self._execution.target_hit(position, snapshot):
            return "target"
        if holding_ms >= config.max_hold_ms:
            return "time"

        signum = position.direction.signum
        if score * signum <= -threshold.exit * config.reversal_ratio:
            return "composite reversal"
        if confidence < config.exit_confidence:
            return f"confidence {confidence:.2f}"
        if config.exit_on_quality_loss:
            critical = [
                reason.value for reason in quality.reasons if reason in _CRITICAL_BLOCKS
            ]
            if critical:
                return "quality: " + "+".join(critical)
        return None

    def _handle_post_exit(
        self, previous_state: TradeState, now_ms: float
    ) -> StateMachineOutput:
        """Move out of the transient exit state into cooldown or neutral."""
        if self._config.cooldown_ms > 0.0 and now_ms < self._cooldown_until_ms:
            return self._transition(
                previous_state, TradeState.COOLDOWN, ("entering cooldown",)
            )
        return self._transition(previous_state, TradeState.NEUTRAL, ("flat",))

    # -- flat states ------------------------------------------------------- #

    def _handle_flat(
        self,
        *,
        snapshot: Snapshot,
        previous_state: TradeState,
        composite: CompositeResult,
        score: float,
        confidence: float,
        threshold: ThresholdResult,
        quality: QualityReport,
        features: FeatureMap,
        now_ms: float,
    ) -> StateMachineOutput:
        """Handle ``NEUTRAL`` and the two ``WATCH`` states."""
        config = self._config

        if not composite.valid:
            self._confirmations = 0
            return self._to_neutral(previous_state, "composite not usable")
        if not quality.tradable:
            self._confirmations = 0
            blocked = "+".join(reason.value for reason in quality.reasons)
            return self._to_neutral(previous_state, f"blocked: {blocked}")

        direction_sign = sign(score)
        if direction_sign == 0:
            self._confirmations = 0
            return self._to_neutral(previous_state, "no directional score")

        if self._is_rearm_blocked(direction_sign):
            self._confirmations = 0
            return self._to_neutral(
                previous_state, "awaiting zero-cross re-arm after exit"
            )

        magnitude = abs(score)
        if magnitude < threshold.watch or confidence < config.min_watch_confidence:
            self._confirmations = 0
            return self._to_neutral(
                previous_state,
                f"below watch ({magnitude:.2f}<{threshold.watch:.2f} "
                f"conf {confidence:.2f})",
            )

        watch_state = (
            TradeState.WATCH_LONG if direction_sign > 0 else TradeState.WATCH_SHORT
        )
        if self._state is not watch_state:
            self._state = watch_state
            self._watch_deadline_ms = now_ms + config.watch_timeout_ms
            self._confirmations = 0
            self._confirmation_sign = direction_sign
            return StateMachineOutput(
                state=watch_state,
                transition=StateTransition(
                    previous_state, watch_state, (f"watch armed at {score:+.2f}",)
                ),
                position=None,
                pnl=None,
                entry_quote=None,
                exit_quote=None,
                reasons=(f"watch armed at {score:+.2f}",),
            )

        if now_ms > self._watch_deadline_ms:
            self._confirmations = 0
            return self._to_neutral(previous_state, "watch expired")

        entry_blocked = self._entry_block_reason(
            snapshot=snapshot,
            direction_sign=direction_sign,
            magnitude=magnitude,
            confidence=confidence,
            threshold=threshold,
            features=features,
        )
        if entry_blocked is not None:
            self._confirmations = 0
            return self._stay(previous_state, (entry_blocked,))

        if self._confirmation_sign != direction_sign:
            self._confirmation_sign = direction_sign
            self._confirmations = 0
        self._confirmations += 1
        if self._confirmations < config.entry_confirmations:
            return self._stay(
                previous_state,
                (
                    f"confirming {self._confirmations}/{config.entry_confirmations}",
                ),
            )

        return self._enter(
            snapshot=snapshot,
            previous_state=previous_state,
            direction_sign=direction_sign,
            score=score,
            confidence=confidence,
            now_ms=now_ms,
        )

    def _entry_block_reason(
        self,
        *,
        snapshot: Snapshot,
        direction_sign: int,
        magnitude: float,
        confidence: float,
        threshold: ThresholdResult,
        features: FeatureMap,
    ) -> str | None:
        """Return why an entry is not allowed yet, or ``None`` if it is."""
        config = self._config
        if magnitude < threshold.entry:
            return f"score {magnitude:.2f} below entry {threshold.entry:.2f}"
        if confidence < config.min_entry_confidence:
            return (
                f"confidence {confidence:.2f} below "
                f"{config.min_entry_confidence:.2f}"
            )
        if config.require_ltp_confirmation:
            ltp = features.get(F_LTP_CONFIRMATION)
            if ltp is None or not ltp.valid:
                return "awaiting traded-price confirmation"
            opposition = self._ltp_config.opposition_threshold
            if direction_sign > 0 and ltp.value <= -opposition:
                return f"traded price contradicts long ({ltp.value:+.2f})"
            if direction_sign < 0 and ltp.value >= opposition:
                return f"traded price contradicts short ({ltp.value:+.2f})"
        allowed, detail = self._execution.clears_cost(snapshot, self._entry_quantity())
        if not allowed:
            return f"cost gate: {detail}"
        return None

    def _entry_quantity(self) -> int:
        """Configured traded quantity for this instrument."""
        return self._quantity

    def _enter(
        self,
        *,
        snapshot: Snapshot,
        previous_state: TradeState,
        direction_sign: int,
        score: float,
        confidence: float,
        now_ms: float,
    ) -> StateMachineOutput:
        """Open a position and move into ``LONG`` or ``SHORT``."""
        direction = Direction.LONG if direction_sign > 0 else Direction.SHORT
        position = self._execution.open_position(
            snapshot, direction, self._quantity, monotonic_ms=now_ms
        )
        self._position = position
        self._state = TradeState.LONG if direction_sign > 0 else TradeState.SHORT
        self._confirmations = 0
        reasons = (
            f"entry {direction.name} at {position.entry_price:.2f}",
            f"score {score:+.2f} conf {confidence:.2f}",
            f"stop {position.stop_price:.2f} target {position.target_price:.2f}",
        )
        _LOGGER.info(
            "%s: %s entry at %.2f (score %+.2f, confidence %.2f)",
            snapshot.symbol,
            direction.name,
            position.entry_price,
            score,
            confidence,
        )
        return StateMachineOutput(
            state=self._state,
            transition=StateTransition(previous_state, self._state, reasons),
            position=position,
            pnl=None,
            entry_quote=position.entry_quote,
            exit_quote=None,
            reasons=reasons,
        )

    # -- re-arm ------------------------------------------------------------ #

    def _update_rearm(self, score: float) -> None:
        """Clear the re-arm block once the score has crossed back through zero."""
        if self._rearm_block_sign == 0:
            return
        if sign(score) != self._rearm_block_sign:
            self._rearm_block_sign = 0

    def _is_rearm_blocked(self, direction_sign: int) -> bool:
        """Whether the requested direction is still blocked after an exit."""
        if not self._config.require_zero_cross_rearm:
            return False
        return self._rearm_block_sign != 0 and self._rearm_block_sign == direction_sign

    # -- transition helpers ------------------------------------------------ #

    def _to_neutral(
        self, previous_state: TradeState, reason: str
    ) -> StateMachineOutput:
        """Move to ``NEUTRAL`` with a reason, or stay there quietly."""
        if self._state is TradeState.NEUTRAL:
            return self._stay(previous_state, (reason,))
        return self._transition(previous_state, TradeState.NEUTRAL, (reason,))

    def _transition(
        self,
        previous_state: TradeState,
        new_state: TradeState,
        reasons: tuple[str, ...],
    ) -> StateMachineOutput:
        """Commit a state change with no position side effects."""
        self._state = new_state
        return StateMachineOutput(
            state=new_state,
            transition=StateTransition(previous_state, new_state, reasons),
            position=self._position,
            pnl=None,
            entry_quote=None,
            exit_quote=None,
            reasons=reasons,
        )

    def _stay(
        self, previous_state: TradeState, reasons: tuple[str, ...]
    ) -> StateMachineOutput:
        """Remain in the current state."""
        return StateMachineOutput(
            state=self._state,
            transition=StateTransition(previous_state, self._state, ()),
            position=self._position,
            pnl=None,
            entry_quote=None,
            exit_quote=None,
            reasons=reasons,
        )


__all__ = ["StateMachineOutput", "TradingStateMachine"]
