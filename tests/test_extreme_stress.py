"""Full-pipeline fail-safe contracts for deterministic extreme-stress streams.

These fixtures deliberately include invalid books and discontinuous feeds. The
success criterion is bounded, observable failure behaviour -- rejection,
flat/reset, warmup and explicit partial-fill handling -- not profitability.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise

import pytest
from snapshot_quant_v4.adapter import (
    ExtremeStressConfig,
    ExtremeStressPattern,
    ExtremeStressSource,
)
from snapshot_quant_v4.config import config_from_mapping
from snapshot_quant_v4.engine.quant_engine import EngineRegistry
from snapshot_quant_v4.utils.clock import ManualClock
from snapshot_quant_v4.utils.types import (
    BlockReason,
    EngineOutput,
    EngineStats,
    FeatureKind,
    MarketDataSource,
    RawSnapshot,
    RejectReason,
    TradeState,
)

TOKEN = "3045"
SYMBOL = "EXTREME_STRESS"
TICK_PAISE = 5
COUNT = 260
EVENT = 120
ALL_PATTERNS = tuple(ExtremeStressPattern)


@dataclass(frozen=True, slots=True)
class StressRun:
    """Raw inputs, accepted outputs and final counters for one pattern."""

    pattern: ExtremeStressPattern
    snapshots: tuple[RawSnapshot, ...]
    events: tuple[EngineOutput | None, ...]
    outputs: tuple[EngineOutput, ...]
    stats: EngineStats


def stress_config(
    pattern: ExtremeStressPattern,
    *,
    allow_partial_fill: bool = True,
):
    """Short-window profile that holds a baseline long until stress arrives."""
    partial = pattern is ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE
    return config_from_mapping(
        {
            "symbols": [
                {
                    "token": TOKEN,
                    "symbol": SYMBOL,
                    "exchange_type": 1,
                    "tick_size": 0.05,
                    "quantity": 5_000 if partial else 1,
                }
            ],
            "validation": {
                "max_spread_ticks": 25.0,
                "max_snapshot_gap_ms": 1_000.0,
                "max_staleness_ms": 8_000.0,
                "max_price_gap_ticks": 40.0,
                "require_tick_alignment": True,
            },
            "quality": {
                "warmup_snapshots": 110,
                "max_signal_spread_ticks": 4.0,
                "min_depth_levels": 2,
                "min_relative_depth": 0.2,
                "gap_block_ms": 1_000.0,
            },
            "confidence": {
                "liquidity_window": 60,
                "volatility_window": 60,
                "stability_window": 20,
            },
            "regime": {
                "window": 40,
                "min_samples": 20,
                "min_dwell_snapshots": 2,
                "confirm_snapshots": 2,
            },
            "threshold": {
                "reference_window": 60,
                "base": 0.22,
                "min_entry": 0.15,
                "max_entry": 0.40,
                "spread_vol_coeff": 0.0,
                "mid_vol_coeff": 0.0,
                "obi_vol_coeff": 0.0,
                "instability_coeff": 0.0,
            },
            "composite": {
                "min_features": 3,
                "min_confidence": 0.05,
                "smoothing_half_life_ms": 300.0,
            },
            "state_machine": {
                "entry_confirmations": 1,
                "min_watch_confidence": 0.05,
                "min_entry_confidence": 0.10,
                "exit_confidence": 0.02,
                "watch_timeout_ms": 30_000.0,
                "min_hold_ms": 0.0,
                "max_hold_ms": 1_000_000.0,
                "cooldown_ms": 200.0,
                "stop_ticks": 8.0,
                "target_ticks": 20.0,
                "use_target": False,
            },
            "execution": {
                "entry_aggression_ticks": 2.0 if partial else 0.0,
                "exit_slippage_ticks": 2.0 if partial else 0.0,
                "use_depth_walk": partial,
                "allow_partial_fill": allow_partial_fill,
                "cost": {"enabled": False},
            },
            "features": {
                "momentum": {"variance_window": 40, "warmup_updates": 4},
                "acceleration": {"variance_window": 40},
                "queue_persistence": {"window": 15, "min_samples": 5},
                "spread": {"variance_window": 40},
            },
            "runtime": {"log_file": None, "render_fps": 0.0},
        }
    )


def collect(pattern: ExtremeStressPattern) -> tuple[RawSnapshot, ...]:
    """Materialise a full exact stream."""
    source = ExtremeStressSource(ExtremeStressConfig(pattern=pattern))
    source.start()
    snapshots = tuple(source.snapshots())
    assert source.emitted == COUNT
    return snapshots


def run_pattern(
    pattern: ExtremeStressPattern,
    *,
    allow_partial_fill: bool = True,
) -> StressRun:
    """Run every raw event through production validation and decision logic."""
    snapshots = collect(pattern)
    clock = ManualClock()
    registry = EngineRegistry(
        stress_config(pattern, allow_partial_fill=allow_partial_fill),
        clock=clock,
    )
    outputs: list[EngineOutput] = []
    events: list[EngineOutput | None] = []
    for raw in snapshots:
        clock.advance(200.0)
        output = registry.process(raw)
        events.append(output)
        if output is not None:
            outputs.append(output)
    return StressRun(
        pattern=pattern,
        snapshots=snapshots,
        events=tuple(events),
        outputs=tuple(outputs),
        stats=registry.statistics()[TOKEN],
    )


def deterministic_outputs(outputs: tuple[EngineOutput, ...]) -> tuple[EngineOutput, ...]:
    """Remove only measured compute latency from an output replay."""
    return tuple(replace(output, compute_us=0.0) for output in outputs)


def event_output(run: StressRun, raw_index: int) -> EngineOutput:
    """Find the accepted output corresponding to one raw event."""
    output = run.events[raw_index]
    assert output is not None
    return output


def entry_count(outputs: tuple[EngineOutput, ...]) -> int:
    """Count position-opening transitions."""
    return sum(
        output.transition.changed and output.state in (TradeState.LONG, TradeState.SHORT)
        for output in outputs
    )


@pytest.mark.parametrize("pattern", ALL_PATTERNS, ids=lambda pattern: pattern.value)
def test_source_is_restartable_exact_and_satisfies_the_protocol(
    pattern: ExtremeStressPattern,
) -> None:
    source = ExtremeStressSource(ExtremeStressConfig(pattern=pattern))
    assert isinstance(source, MarketDataSource)
    assert source.pattern is pattern
    source.start()
    first = tuple(source.snapshots())
    source.start()
    second = tuple(source.snapshots())
    assert first == second
    assert len(first) == COUNT
    assert source.emitted == COUNT
    assert all(raw.token == TOKEN and raw.symbol == SYMBOL for raw in first)
    assert all(len(raw.bids) == len(raw.asks) == 5 for raw in first)
    assert all(
        later.volume_traded_today > earlier.volume_traded_today
        for earlier, later in pairwise(first)
    )


def test_source_stop_then_restart_recovers_the_complete_stream() -> None:
    source = ExtremeStressSource()
    source.start()
    iterator = source.snapshots()
    assert next(iterator).sequence_number == 1
    source.stop()
    assert list(iterator) == []
    source.start()
    assert len(tuple(source.snapshots())) == COUNT


def test_exact_adversarial_transformations_are_present() -> None:
    flash = collect(ExtremeStressPattern.FLASH_CRASH)
    flash_moves = [
        (flash[index].bids[0].price_paise - flash[index - 1].bids[0].price_paise)
        // TICK_PAISE
        for index in range(EVENT, EVENT + 4)
    ]
    assert flash_moves == [-19, -19, -19, -19]

    vacuum = collect(ExtremeStressPattern.LIQUIDITY_VACUUM)[EVENT]
    assert tuple(level.quantity for level in vacuum.bids) == (5, 4, 3, 2, 1)
    assert tuple(level.quantity for level in vacuum.asks) == (5, 4, 3, 2, 1)

    spread = collect(ExtremeStressPattern.SPREAD_EXPLOSION)[EVENT]
    assert (spread.asks[0].price_paise - spread.bids[0].price_paise) // TICK_PAISE == 12

    whipsaw = collect(ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW)
    offsets = [
        (raw.bids[0].price_paise - 80_000) // TICK_PAISE for raw in whipsaw[:40]
    ]
    assert offsets == [*range(21), *range(19, 0, -1)]

    gaps = collect(ExtremeStressPattern.GAP_REVERSAL)
    gap_moves = [
        (gaps[index].bids[0].price_paise - gaps[index - 1].bids[0].price_paise)
        // TICK_PAISE
        for index in (EVENT, EVENT + 1, EVENT + 2)
    ]
    assert gap_moves == [61, 1, -119]

    evaporated = collect(ExtremeStressPattern.DEPTH_EVAPORATION)[EVENT]
    assert all(level.quantity == 0 and level.orders == 0 for level in evaporated.bids[1:])
    assert all(level.quantity == 0 and level.orders == 0 for level in evaporated.asks[1:])

    locked = collect(ExtremeStressPattern.LIMIT_LOCK)
    assert all(
        raw.bids[0].price_paise == raw.asks[0].price_paise
        for raw in locked[EVENT : EVENT + 6]
    )

    halted = collect(ExtremeStressPattern.HALT_AND_REOPEN)
    assert (
        halted[EVENT].exchange_timestamp_ms - halted[EVENT - 1].exchange_timestamp_ms
        == 5_200
    )
    assert (
        halted[EVENT].bids[0].price_paise - halted[EVENT - 1].bids[0].price_paise
    ) // TICK_PAISE == 31

    dropped = collect(ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED)
    assert dropped[EVENT].sequence_number == EVENT + 11
    assert dropped[EVENT + 1].exchange_timestamp_ms < dropped[EVENT].exchange_timestamp_ms
    assert dropped[EVENT + 2].sequence_number < dropped[EVENT].sequence_number

    partial = collect(ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE)
    assert sum(level.quantity for level in partial[EVENT - 1].asks) == 150
    assert (
        partial[EVENT].bids[0].price_paise - partial[EVENT - 1].bids[0].price_paise
    ) // TICK_PAISE == -19

    disaster = collect(ExtremeStressPattern.COMBINED_DISASTER)
    assert (
        disaster[EVENT].asks[0].price_paise - disaster[EVENT].bids[0].price_paise
    ) // TICK_PAISE == 12
    assert disaster[EVENT + 4].bids[0].price_paise == disaster[EVENT + 4].asks[0].price_paise
    assert disaster[EVENT + 7].exchange_timestamp_ms < disaster[EVENT + 3].exchange_timestamp_ms
    assert (
        disaster[EVENT + 8].bids[0].price_paise
        - disaster[EVENT + 3].bids[0].price_paise
    ) // TICK_PAISE == -65


@pytest.mark.parametrize("pattern", ALL_PATTERNS, ids=lambda pattern: pattern.value)
def test_full_pipeline_is_deterministic_finite_bounded_and_accounted(
    pattern: ExtremeStressPattern,
) -> None:
    first = run_pattern(pattern)
    replay = run_pattern(pattern)
    assert first.snapshots == replay.snapshots
    assert deterministic_outputs(first.outputs) == deterministic_outputs(replay.outputs)
    assert first.stats == replay.stats
    assert first.stats.accepted + first.stats.rejected == COUNT
    assert first.stats.accepted == len(first.outputs)
    assert first.stats.blocked == sum(not output.quality.tradable for output in first.outputs)
    assert first.stats.resets <= first.stats.gaps

    for output in first.outputs:
        assert output.transition.current is output.state
        assert math.isfinite(output.compute_us) and 0.0 < output.compute_us < 50_000.0
        assert -1.0 <= output.composite.score <= 1.0
        assert -1.0 <= output.composite.smoothed <= 1.0
        assert 0.0 <= output.composite.confidence <= 1.0
        assert 0.0 <= output.threshold.exit <= output.threshold.watch
        assert output.threshold.watch <= output.threshold.entry <= 1.0
        assert 0.0 <= output.quality.book_quality <= 1.0
        assert 0.0 <= output.quality.liquidity_score <= 1.0
        assert math.isfinite(output.regime.efficiency_ratio)
        assert math.isfinite(output.regime.volatility_ticks)
        assert math.isfinite(output.regime.trend_ticks)
        for feature in output.features.values():
            assert math.isfinite(feature.raw)
            assert math.isfinite(feature.value)
            assert math.isfinite(feature.local_confidence)
            assert math.isfinite(feature.confidence)
            assert 0.0 <= feature.local_confidence <= 1.0
            assert 0.0 <= feature.confidence <= 1.0
            lower = -1.0 if feature.kind is FeatureKind.DIRECTIONAL else 0.0
            assert lower <= feature.value <= 1.0
        if output.position is not None:
            assert output.position.is_open
            assert output.state in (TradeState.LONG, TradeState.SHORT)
        for quote in (output.entry_quote, output.exit_quote):
            if quote is not None:
                assert 0 <= quote.filled_quantity <= quote.requested_quantity
                assert quote.complete == (
                    quote.filled_quantity == quote.requested_quantity
                )
        if output.pnl is not None:
            expected_gross = (
                (output.pnl.exit_price - output.pnl.entry_price)
                * output.pnl.direction.signum
                * output.pnl.quantity
            )
            assert output.pnl.gross_rupees == pytest.approx(expected_gross)
            assert output.pnl.net_rupees == pytest.approx(
                output.pnl.gross_rupees - output.pnl.cost_rupees
            )
            assert output.pnl.holding_ms >= 0.0


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (ExtremeStressPattern.FLASH_CRASH, EngineStats(260, 0, 109, 0, 0, {})),
        (ExtremeStressPattern.LIQUIDITY_VACUUM, EngineStats(260, 0, 119, 0, 0, {})),
        (ExtremeStressPattern.SPREAD_EXPLOSION, EngineStats(260, 0, 119, 0, 0, {})),
        (
            ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW,
            EngineStats(260, 0, 109, 0, 0, {}),
        ),
        (ExtremeStressPattern.GAP_REVERSAL, EngineStats(260, 0, 220, 2, 2, {})),
        (ExtremeStressPattern.DEPTH_EVAPORATION, EngineStats(260, 0, 119, 0, 0, {})),
        (
            ExtremeStressPattern.LIMIT_LOCK,
            EngineStats(254, 6, 218, 1, 1, {RejectReason.LOCKED_BOOK.value: 6}),
        ),
        (ExtremeStressPattern.HALT_AND_REOPEN, EngineStats(260, 0, 218, 1, 1, {})),
        (
            ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED,
            EngineStats(
                259,
                1,
                218,
                2,
                1,
                {RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1},
            ),
        ),
        (
            ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE,
            EngineStats(260, 0, 109, 0, 0, {}),
        ),
        (
            ExtremeStressPattern.COMBINED_DISASTER,
            EngineStats(
                256,
                4,
                222,
                2,
                1,
                {
                    RejectReason.LOCKED_BOOK.value: 3,
                    RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1,
                },
            ),
        ),
    ],
    ids=lambda value: value.value if isinstance(value, ExtremeStressPattern) else None,
)
def test_every_pattern_has_exact_observable_counter_contract(
    pattern: ExtremeStressPattern,
    expected: EngineStats,
) -> None:
    assert run_pattern(pattern).stats == expected


def test_flash_crash_is_accepted_without_a_gap_and_hits_the_stop() -> None:
    run = run_pattern(ExtremeStressPattern.FLASH_CRASH)
    assert run.stats.accepted == COUNT
    assert run.stats.rejected == run.stats.gaps == run.stats.resets == 0
    assert run.stats.blocked >= 109
    crashes = [
        event_output(run, index) for index in range(EVENT, EVENT + 4)
    ]
    crash = next(output for output in crashes if output.state is TradeState.EXIT_LONG)
    assert crash.state is TradeState.EXIT_LONG
    assert crash.position is None
    assert crash.pnl is not None and crash.pnl.gross_ticks < 0.0
    assert any("stop" in reason for reason in crash.reasons)


@pytest.mark.parametrize(
    ("pattern", "reason"),
    [
        (ExtremeStressPattern.LIQUIDITY_VACUUM, BlockReason.LIQUIDITY_BELOW_THRESHOLD),
        (ExtremeStressPattern.SPREAD_EXPLOSION, BlockReason.SPREAD_TOO_WIDE),
        (ExtremeStressPattern.DEPTH_EVAPORATION, BlockReason.INSUFFICIENT_DEPTH_LEVELS),
    ],
)
def test_accepted_quality_disasters_exit_and_block_reentry(
    pattern: ExtremeStressPattern,
    reason: BlockReason,
) -> None:
    run = run_pattern(pattern)
    stressed = event_output(run, EVENT)
    assert reason in stressed.quality.reasons
    assert not stressed.quality.tradable
    assert stressed.state is TradeState.EXIT_LONG
    assert stressed.position is None
    assert all(
        output.state not in (TradeState.LONG, TradeState.SHORT)
        for output in run.outputs[EVENT : EVENT + 10]
    )


def test_high_confidence_whipsaw_exercises_both_sides_with_bounded_churn() -> None:
    run = run_pattern(ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW)
    states = Counter(output.state for output in run.outputs)
    entries = [
        output
        for output in run.outputs
        if output.transition.changed
        and output.state in (TradeState.LONG, TradeState.SHORT)
    ]
    assert states[TradeState.LONG] > 0
    assert states[TradeState.SHORT] > 0
    assert 2 <= len(entries) <= 14
    assert min(output.composite.confidence for output in entries) >= 0.45
    assert run.stats.gaps == run.stats.resets == run.stats.rejected == 0


def assert_gap_reset_and_rewarm(run: StressRun, expected_resets: int) -> None:
    """Assert each visible reset output is flat and starts a fresh warmup."""
    reset_indexes = [
        index
        for index, output in enumerate(run.outputs)
        if "! state reset after feed gap" in output.reasons
    ]
    assert len(reset_indexes) == expected_resets
    assert run.stats.resets == expected_resets
    for index in reset_indexes:
        reset = run.outputs[index]
        assert reset.state is TradeState.WARMUP
        assert reset.position is None
        recovery = run.outputs[index : index + 24]
        assert recovery
        assert all(output.state is TradeState.WARMUP for output in recovery)
        assert all(BlockReason.WARMUP in output.quality.reasons for output in recovery)
        assert entry_count(tuple(recovery)) == 0


def test_gap_reversal_forces_two_separate_resets_and_rewarmups() -> None:
    run = run_pattern(ExtremeStressPattern.GAP_REVERSAL)
    assert run.stats.gaps == 2
    assert run.stats.rejected == 0
    assert_gap_reset_and_rewarm(run, 2)


@pytest.mark.parametrize(
    ("pattern", "rejects", "reasons", "gaps"),
    [
        (
            ExtremeStressPattern.LIMIT_LOCK,
            6,
            {RejectReason.LOCKED_BOOK.value: 6},
            1,
        ),
        (
            ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED,
            1,
            {RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1},
            2,
        ),
        (
            ExtremeStressPattern.COMBINED_DISASTER,
            4,
            {
                RejectReason.LOCKED_BOOK.value: 3,
                RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1,
            },
            2,
        ),
    ],
)
def test_invalid_feed_patterns_have_exact_rejection_and_reset_accounting(
    pattern: ExtremeStressPattern,
    rejects: int,
    reasons: dict[str, int],
    gaps: int,
) -> None:
    run = run_pattern(pattern)
    assert run.stats.accepted == COUNT - rejects
    assert run.stats.rejected == rejects
    assert run.stats.rejects_by_reason == reasons
    assert run.stats.gaps == gaps
    assert_gap_reset_and_rewarm(run, 1)


def test_forward_sequence_skip_is_observed_then_out_of_order_feed_resets() -> None:
    run = run_pattern(ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED)
    forward_skip = run.events[EVENT]
    backwards_timestamp = run.events[EVENT + 1]
    sequence_regression = run.events[EVENT + 2]
    assert forward_skip is not None
    assert "! state reset after feed gap" not in forward_skip.reasons
    assert backwards_timestamp is None
    assert sequence_regression is not None
    assert "! state reset after feed gap" in sequence_regression.reasons
    assert sequence_regression.state is TradeState.WARMUP


def test_halt_and_reopen_is_accepted_as_a_gap_then_rewarms() -> None:
    run = run_pattern(ExtremeStressPattern.HALT_AND_REOPEN)
    assert run.stats.accepted == COUNT
    assert run.stats.rejected == 0
    assert run.stats.gaps == 1
    assert_gap_reset_and_rewarm(run, 1)


def test_partial_fill_is_explicit_slipped_sized_and_fully_exitable() -> None:
    run = run_pattern(ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE)
    entry = next(output for output in run.outputs if output.entry_quote is not None)
    assert entry.position is not None
    assert entry.entry_quote is not None
    assert not entry.entry_quote.complete
    assert entry.entry_quote.requested_quantity == 5_000
    assert entry.entry_quote.filled_quantity == 150
    assert entry.entry_quote.levels_consumed == 5
    assert entry.entry_quote.price > entry.entry_quote.reference_price
    assert entry.position.quantity == 150

    exit_output = event_output(run, EVENT)
    assert exit_output.state is TradeState.EXIT_LONG
    assert exit_output.exit_quote is not None and exit_output.exit_quote.complete
    assert exit_output.exit_quote.filled_quantity == 150
    assert exit_output.exit_quote.price < exit_output.exit_quote.reference_price
    assert exit_output.pnl is not None and exit_output.pnl.quantity == 150


def test_partial_fill_disabled_blocks_pipeline_entry_without_throwing() -> None:
    run = run_pattern(
        ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE,
        allow_partial_fill=False,
    )
    assert entry_count(run.outputs) == 0
    assert all(output.position is None for output in run.outputs)
    assert any(
        any("insufficient visible entry depth" in reason for reason in output.reasons)
        for output in run.outputs
    )


def test_combined_disaster_exposes_every_layer_and_remains_flat_after_reset() -> None:
    run = run_pattern(ExtremeStressPattern.COMBINED_DISASTER)
    first = event_output(run, EVENT)
    assert BlockReason.SPREAD_TOO_WIDE in first.quality.reasons
    assert BlockReason.LIQUIDITY_BELOW_THRESHOLD in first.quality.reasons
    assert first.state is TradeState.EXIT_LONG
    assert first.position is None
    assert run.stats.rejects_by_reason == {
        RejectReason.LOCKED_BOOK.value: 3,
        RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1,
    }
    assert_gap_reset_and_rewarm(run, 1)
