"""State-machine tests.

The state machine is driven directly with synthetic composite, threshold, quality
and feature inputs. That keeps each test about one transition rule instead of
about the market conditions needed to provoke it, and it lets time be controlled
exactly through :class:`ManualClock`.
"""

from __future__ import annotations

import pytest
from snapshot_quant_v4.config import (
    CostConfig,
    ExecutionConfig,
    LtpConfirmationConfig,
    StateMachineConfig,
)
from snapshot_quant_v4.engine.execution import ExecutionModel
from snapshot_quant_v4.engine.state_machine import TradingStateMachine
from snapshot_quant_v4.utils.clock import ManualClock
from snapshot_quant_v4.utils.constants import F_LTP_CONFIRMATION
from snapshot_quant_v4.utils.types import (
    BlockReason,
    CompositeResult,
    Direction,
    ThresholdResult,
    TradeState,
)

from conftest import (
    BASE_PAISE,
    TICK_PAISE,
    feature_value,
    make_snapshot,
    quality_report,
    validate_one,
)

LTP_CONFIG = LtpConfirmationConfig(opposition_threshold=0.35)


def composite(score: float, *, confidence: float = 0.9, valid: bool = True) -> CompositeResult:
    """Build a composite result with a chosen score and confidence."""
    return CompositeResult(
        score=score,
        smoothed=score,
        confidence=confidence,
        weight_mass=1.0,
        used_features=5,
        contributions=(),
        valid=valid,
    )


def thresholds(entry: float = 0.30) -> ThresholdResult:
    """Build a threshold result derived from an entry level."""
    return ThresholdResult(
        entry=entry,
        watch=entry * 0.6,
        exit=entry * 0.5,
        floor_applied=False,
        detail="",
    )


def features(ltp: float = 0.5) -> dict:
    """Build the minimal feature map the machine consults."""
    return {F_LTP_CONFIRMATION: feature_value(F_LTP_CONFIRMATION, ltp, confidence=0.8)}


def build_machine(
    *,
    clock: ManualClock,
    state_config: StateMachineConfig | None = None,
    execution_config: ExecutionConfig | None = None,
) -> TradingStateMachine:
    """Construct a state machine with injected clock and execution model."""
    resolved_state = (
        state_config
        if state_config is not None
        else StateMachineConfig(
            entry_confirmations=1,
            min_watch_confidence=0.2,
            min_entry_confidence=0.5,
            cooldown_ms=1_000.0,
            min_hold_ms=0.0,
            max_hold_ms=10_000.0,
            stop_ticks=4.0,
            target_ticks=8.0,
        )
    )
    resolved_execution = (
        execution_config
        if execution_config is not None
        else ExecutionConfig(
            entry_aggression_ticks=1.0,
            exit_slippage_ticks=0.0,
            cost=CostConfig(enabled=False),
        )
    )
    execution = ExecutionModel(resolved_execution, resolved_state)
    return TradingStateMachine(resolved_state, LTP_CONFIG, execution, clock, quantity=1)


def warm_up(machine: TradingStateMachine, snapshot) -> None:
    """Move the machine out of ``WARMUP`` into ``NEUTRAL``."""
    machine.update(
        snapshot=snapshot,
        composite=composite(0.0),
        threshold=thresholds(),
        quality=quality_report(),
        features=features(),
    )
    assert machine.state is TradeState.NEUTRAL


class TestWarmup:
    def test_stays_in_warmup_while_the_gate_reports_warmup(self) -> None:
        machine = build_machine(clock=ManualClock())
        snapshot = validate_one(make_snapshot())
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(
                tradable=False, reasons=(BlockReason.WARMUP,)
            ),
            features=features(),
        )
        assert output.state is TradeState.WARMUP
        assert "warming up" in output.reasons[0]

    def test_leaves_warmup_when_the_gate_opens(self) -> None:
        machine = build_machine(clock=ManualClock())
        warm_up(machine, validate_one(make_snapshot()))


class TestEntry:
    def test_watch_then_entry_on_a_strong_score(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)

        watch = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=0.6),
        )
        assert watch.state is TradeState.WATCH_LONG

        clock.advance(100.0)
        entry = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=0.6),
        )
        assert entry.state is TradeState.LONG
        assert entry.position is not None
        assert entry.position.direction is Direction.LONG
        # Entry crosses the spread and pays one tick of aggression on top.
        assert entry.position.entry_price == pytest.approx(800.10)
        assert entry.position.stop_price == pytest.approx(800.10 - 4 * 0.05)
        assert entry.position.target_price == pytest.approx(800.10 + 8 * 0.05)

    def test_short_entry_sells_the_bid(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        machine.update(
            snapshot=snapshot,
            composite=composite(-0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=-0.6),
        )
        clock.advance(100.0)
        entry = machine.update(
            snapshot=snapshot,
            composite=composite(-0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=-0.6),
        )
        assert entry.state is TradeState.SHORT
        assert entry.position is not None
        assert entry.position.entry_price == pytest.approx(799.90)

    def test_opposing_traded_price_vetoes_entry(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=-0.9),
        )
        clock.advance(100.0)
        blocked = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=-0.9),
        )
        assert blocked.state is TradeState.WATCH_LONG
        assert any("contradicts long" in reason for reason in blocked.reasons)

    def test_invalid_traded_price_feature_blocks_entry(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        invalid = {
            F_LTP_CONFIRMATION: feature_value(
                F_LTP_CONFIRMATION, 0.0, confidence=0.0, valid=False
            )
        }
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=invalid,
        )
        clock.advance(100.0)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=invalid,
        )
        assert output.state is TradeState.WATCH_LONG
        assert any("traded-price confirmation" in reason for reason in output.reasons)

    def test_low_confidence_blocks_entry(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9, confidence=0.3),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        clock.advance(50.0)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9, confidence=0.3),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.WATCH_LONG
        assert any("confidence" in reason for reason in output.reasons)

    def test_blocked_quality_prevents_watch(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.95),
            threshold=thresholds(),
            quality=quality_report(
                tradable=False, reasons=(BlockReason.SPREAD_TOO_WIDE,)
            ),
            features=features(),
        )
        assert output.state is TradeState.NEUTRAL

    def test_watch_expires_after_the_timeout(self) -> None:
        clock = ManualClock()
        machine = build_machine(
            clock=clock,
            state_config=StateMachineConfig(
                watch_timeout_ms=500.0,
                entry_confirmations=5,
                min_watch_confidence=0.2,
                min_entry_confidence=0.5,
            ),
        )
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert machine.state is TradeState.WATCH_LONG
        clock.advance(600.0)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.NEUTRAL
        assert "watch expired" in output.reasons

    def test_confirmations_are_required_on_consecutive_snapshots(self) -> None:
        clock = ManualClock()
        machine = build_machine(
            clock=clock,
            state_config=StateMachineConfig(
                entry_confirmations=3,
                min_watch_confidence=0.2,
                min_entry_confidence=0.5,
                watch_timeout_ms=10_000.0,
            ),
        )
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        for _ in range(2):
            machine.update(
                snapshot=snapshot,
                composite=composite(0.9),
                threshold=thresholds(),
                quality=quality_report(),
                features=features(),
            )
            clock.advance(50.0)
        assert machine.state is TradeState.WATCH_LONG
        # A snapshot below the entry threshold resets the counter.
        machine.update(
            snapshot=snapshot,
            composite=composite(0.25),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        clock.advance(50.0)
        for _ in range(2):
            machine.update(
                snapshot=snapshot,
                composite=composite(0.9),
                threshold=thresholds(),
                quality=quality_report(),
                features=features(),
            )
            clock.advance(50.0)
        assert machine.state is TradeState.WATCH_LONG
        final = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert final.state is TradeState.LONG


class TestExit:
    def _enter_long(self, clock: ManualClock, machine: TradingStateMachine, snapshot):
        warm_up(machine, snapshot)
        for _ in range(2):
            machine.update(
                snapshot=snapshot,
                composite=composite(0.9),
                threshold=thresholds(),
                quality=quality_report(),
                features=features(),
            )
            clock.advance(50.0)
        assert machine.state is TradeState.LONG
        return machine.position

    def test_stop_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        # Entry at 800.10 with a four-tick stop means a stop price of 799.90; the
        # bid must fall to or below it.
        crashed = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE - 8 * TICK_PAISE,
                ask_paise=BASE_PAISE - 6 * TICK_PAISE,
            )
        )
        output = machine.update(
            snapshot=crashed,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("stop" in reason for reason in output.reasons)
        assert output.pnl is not None
        assert output.pnl.gross_ticks < 0.0

    def test_target_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        rallied = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE + 12 * TICK_PAISE,
                ask_paise=BASE_PAISE + 14 * TICK_PAISE,
            )
        )
        output = machine.update(
            snapshot=rallied,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("target" in reason for reason in output.reasons)
        assert output.pnl is not None
        assert output.pnl.gross_ticks > 0.0

    def test_composite_reversal_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(-0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(ltp=-0.5),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("reversal" in reason for reason in output.reasons)

    def test_time_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        clock.advance(20_000.0)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("time" in reason for reason in output.reasons)

    def test_confidence_collapse_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9, confidence=0.05),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("confidence" in reason for reason in output.reasons)

    def test_critical_quality_loss_exit(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(tradable=False, reasons=(BlockReason.FEED_GAP,)),
            features=features(),
        )
        assert output.state is TradeState.EXIT_LONG
        assert any("quality" in reason for reason in output.reasons)

    def test_minimum_hold_defers_a_soft_exit_but_not_a_stop(self) -> None:
        clock = ManualClock()
        machine = build_machine(
            clock=clock,
            state_config=StateMachineConfig(
                entry_confirmations=1,
                min_watch_confidence=0.2,
                min_entry_confidence=0.5,
                min_hold_ms=5_000.0,
                max_hold_ms=60_000.0,
                stop_ticks=4.0,
            ),
        )
        snapshot = validate_one(make_snapshot())
        self._enter_long(clock, machine, snapshot)
        # A soft reversal inside the minimum hold is ignored.
        held = machine.update(
            snapshot=snapshot,
            composite=composite(-0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert held.state is TradeState.LONG
        # A stop inside the minimum hold is not.
        crashed = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE - 10 * TICK_PAISE,
                ask_paise=BASE_PAISE - 8 * TICK_PAISE,
            )
        )
        stopped = machine.update(
            snapshot=crashed,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert stopped.state is TradeState.EXIT_LONG


class TestCooldownAndRearm:
    def _enter_and_exit(self, clock: ManualClock, machine: TradingStateMachine, snapshot):
        warm_up(machine, snapshot)
        for _ in range(2):
            machine.update(
                snapshot=snapshot,
                composite=composite(0.9),
                threshold=thresholds(),
                quality=quality_report(),
                features=features(),
            )
            clock.advance(50.0)
        clock.advance(20_000.0)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert machine.state is TradeState.EXIT_LONG

    def test_exit_leads_to_cooldown_then_neutral(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_and_exit(clock, machine, snapshot)

        cooling = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert cooling.state is TradeState.COOLDOWN
        clock.advance(2_000.0)
        flat = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert flat.state is TradeState.NEUTRAL

    def test_same_direction_is_blocked_until_the_score_crosses_zero(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        self._enter_and_exit(clock, machine, snapshot)
        clock.advance(5_000.0)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )  # EXIT -> COOLDOWN/NEUTRAL
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        blocked = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert blocked.state is TradeState.NEUTRAL
        assert any("re-arm" in reason for reason in blocked.reasons)

        # A negative score clears the block; the next positive score may enter.
        machine.update(
            snapshot=snapshot,
            composite=composite(-0.1),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        clock.advance(50.0)
        reentered = machine.update(
            snapshot=snapshot,
            composite=composite(0.9),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert reentered.state is TradeState.LONG


class TestForceFlat:
    def test_force_flat_closes_and_returns_to_warmup(self) -> None:
        clock = ManualClock()
        machine = build_machine(clock=clock)
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        for _ in range(2):
            machine.update(
                snapshot=snapshot,
                composite=composite(0.9),
                threshold=thresholds(),
                quality=quality_report(),
                features=features(),
            )
            clock.advance(50.0)
        assert machine.state is TradeState.LONG

        output = machine.force_flat(snapshot, "feed gap")
        assert output.state is TradeState.WARMUP
        assert machine.position is None
        assert output.pnl is not None
        assert output.exit_quote is not None

    def test_force_flat_without_a_position_is_safe(self) -> None:
        machine = build_machine(clock=ManualClock())
        snapshot = validate_one(make_snapshot())
        output = machine.force_flat(snapshot, "reconnect")
        assert output.state is TradeState.WARMUP
        assert output.pnl is None


class TestCostGate:
    def test_cost_gate_can_block_entry(self) -> None:
        clock = ManualClock()
        state_config = StateMachineConfig(
            entry_confirmations=1,
            min_watch_confidence=0.2,
            min_entry_confidence=0.5,
            target_ticks=1.0,
        )
        execution_config = ExecutionConfig(
            entry_aggression_ticks=1.0,
            enforce_cost_gate=True,
            min_edge_multiple=2.0,
            cost=CostConfig(enabled=True),
        )
        machine = build_machine(
            clock=clock,
            state_config=state_config,
            execution_config=execution_config,
        )
        snapshot = validate_one(make_snapshot())
        warm_up(machine, snapshot)
        machine.update(
            snapshot=snapshot,
            composite=composite(0.95),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        clock.advance(50.0)
        output = machine.update(
            snapshot=snapshot,
            composite=composite(0.95),
            threshold=thresholds(),
            quality=quality_report(),
            features=features(),
        )
        assert output.state is TradeState.WATCH_LONG
        assert any("cost gate" in reason for reason in output.reasons)

    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(ValueError):
            TradingStateMachine(
                StateMachineConfig(),
                LTP_CONFIG,
                ExecutionModel(ExecutionConfig(), StateMachineConfig()),
                ManualClock(),
                quantity=0,
            )
