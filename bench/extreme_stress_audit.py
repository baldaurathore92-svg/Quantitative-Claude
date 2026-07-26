"""Observable fail-safe audit for all deterministic extreme-stress patterns.

This is a software safety audit, not a backtest or return estimate. It drives
adversarial market, feed and execution fixtures through the production pipeline
and reports rejection, blocking, flat/reset, warmup, partial-fill, PnL and
latency evidence. Profit is never a pass criterion.

Usage::

    python bench/extreme_stress_audit.py
    python bench/extreme_stress_audit.py --count 260 --seed 20260726
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    RawSnapshot,
    RejectReason,
    TradeState,
)

_TOKEN = "3045"
_SYMBOL = "EXTREME_STRESS"
_TICK_SIZE = 0.05
_DEFAULT_COUNT = 260
_STRESS_INDEX = 120
_WARMUP_SNAPSHOTS = 110

_EXPECTED_STATS: dict[ExtremeStressPattern, EngineStats] = {
    ExtremeStressPattern.FLASH_CRASH: EngineStats(260, 0, 109, 0, 0, {}),
    ExtremeStressPattern.LIQUIDITY_VACUUM: EngineStats(260, 0, 119, 0, 0, {}),
    ExtremeStressPattern.SPREAD_EXPLOSION: EngineStats(260, 0, 119, 0, 0, {}),
    ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW: EngineStats(260, 0, 109, 0, 0, {}),
    ExtremeStressPattern.GAP_REVERSAL: EngineStats(260, 0, 220, 2, 2, {}),
    ExtremeStressPattern.DEPTH_EVAPORATION: EngineStats(260, 0, 119, 0, 0, {}),
    ExtremeStressPattern.LIMIT_LOCK: EngineStats(
        254,
        6,
        218,
        1,
        1,
        {RejectReason.LOCKED_BOOK.value: 6},
    ),
    ExtremeStressPattern.HALT_AND_REOPEN: EngineStats(260, 0, 218, 1, 1, {}),
    ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED: EngineStats(
        259,
        1,
        218,
        2,
        1,
        {RejectReason.NON_MONOTONIC_TIMESTAMP.value: 1},
    ),
    ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE: EngineStats(260, 0, 109, 0, 0, {}),
    ExtremeStressPattern.COMBINED_DISASTER: EngineStats(
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
}


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """One materialised input stream and its accepted engine outputs."""

    snapshots: tuple[RawSnapshot, ...]
    events: tuple[EngineOutput | None, ...]
    outputs: tuple[EngineOutput, ...]
    stats: EngineStats


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Observable metrics and invariant findings for one stress pattern."""

    pattern: ExtremeStressPattern
    snapshots: int
    stats: EngineStats
    block_reasons: dict[str, int]
    state_counts: dict[str, int]
    regime_counts: dict[str, int]
    transitions: dict[str, int]
    entries: int
    exits: int
    reset_evidence: int
    resets_from_open_position: int
    partial_entries: int
    entry_slippage_mean_ticks: float
    entry_slippage_max_ticks: float
    exit_slippage_mean_ticks: float
    exit_slippage_max_ticks: float
    reported_gross_ticks: float
    reported_net_rupees: float
    latency_p50_us: float
    latency_p90_us: float
    latency_p99_us: float
    latency_max_us: float
    raw_digest: str
    output_digest: str
    raw_deterministic: bool
    output_deterministic: bool
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether determinism and all software invariants passed."""
        return (
            self.raw_deterministic
            and self.output_deterministic
            and not self.violations
        )


def audit_config(pattern: ExtremeStressPattern):
    """Profile that establishes a long shortly before the stress event."""
    partial = pattern is ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE
    return config_from_mapping(
        {
            "symbols": [
                {
                    "token": _TOKEN,
                    "symbol": _SYMBOL,
                    "exchange_type": 1,
                    "tick_size": _TICK_SIZE,
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
                "warmup_snapshots": _WARMUP_SNAPSHOTS,
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
                "allow_partial_fill": True,
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


def build_snapshots(
    pattern: ExtremeStressPattern,
    count: int,
    seed: int,
) -> tuple[RawSnapshot, ...]:
    """Materialise one deterministic source before timing engine work."""
    source = ExtremeStressSource(
        ExtremeStressConfig(pattern=pattern, count=count, seed=seed)
    )
    source.start()
    return tuple(source.snapshots())


def process(
    pattern: ExtremeStressPattern,
    snapshots: tuple[RawSnapshot, ...],
) -> PipelineRun:
    """Run one materialised stream through the production pipeline."""
    clock = ManualClock()
    registry = EngineRegistry(audit_config(pattern), clock=clock)
    outputs: list[EngineOutput] = []
    events: list[EngineOutput | None] = []
    for snapshot in snapshots:
        clock.advance(200.0)
        output = registry.process(snapshot)
        events.append(output)
        if output is not None:
            outputs.append(output)
    return PipelineRun(
        snapshots,
        tuple(events),
        tuple(outputs),
        registry.statistics()[_TOKEN],
    )


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty sample."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def raw_digest(snapshots: tuple[RawSnapshot, ...]) -> str:
    """Stable digest of the complete raw event stream."""
    return hashlib.sha256(repr(snapshots).encode()).hexdigest()


def output_digest(outputs: tuple[EngineOutput, ...]) -> str:
    """Stable digest excluding only nondeterministic measured compute time."""
    deterministic = tuple(replace(output, compute_us=0.0) for output in outputs)
    return hashlib.sha256(repr(deterministic).encode()).hexdigest()


def _event_output(run: PipelineRun, raw_index: int) -> EngineOutput | None:
    return run.events[raw_index]


def _entry_outputs(run: PipelineRun) -> list[EngineOutput]:
    return [
        output
        for output in run.outputs
        if output.transition.changed
        and output.state in (TradeState.LONG, TradeState.SHORT)
    ]


def _exit_outputs(run: PipelineRun) -> list[EngineOutput]:
    return [
        output
        for output in run.outputs
        if output.transition.changed
        and output.state in (TradeState.EXIT_LONG, TradeState.EXIT_SHORT)
    ]


def invariant_violations(
    pattern: ExtremeStressPattern,
    run: PipelineRun,
    *,
    expected_count: int,
) -> list[str]:
    """Check generic bounds plus the named fail-safe contract for one pattern."""
    failures: list[str] = []

    def fail(message: str) -> None:
        if len(failures) < 30:
            failures.append(message)

    stats = run.stats
    outputs = run.outputs
    if len(run.snapshots) != expected_count:
        fail(f"source emitted {len(run.snapshots)}, expected {expected_count}")
    if stats.accepted + stats.rejected != expected_count:
        fail("accepted + rejected does not equal source count")
    if stats.accepted != len(outputs):
        fail("accepted counter does not equal output count")
    if stats.blocked != sum(not output.quality.tradable for output in outputs):
        fail("blocked counter does not equal non-tradable outputs")
    if stats.resets > stats.gaps:
        fail("reset count exceeds gap count")
    if expected_count == _DEFAULT_COUNT and stats != _EXPECTED_STATS[pattern]:
        fail(f"counter contract differs: got {stats}, expected {_EXPECTED_STATS[pattern]}")

    reset_indexes: list[int] = []
    for index, output in enumerate(outputs):
        if output.transition.current is not output.state:
            fail(f"output {index}: transition current differs from state")
        if not (
            math.isfinite(output.compute_us)
            and math.isfinite(output.composite.score)
            and math.isfinite(output.composite.smoothed)
            and math.isfinite(output.composite.confidence)
            and math.isfinite(output.regime.efficiency_ratio)
            and math.isfinite(output.regime.volatility_ticks)
            and math.isfinite(output.regime.trend_ticks)
        ):
            fail(f"output {index}: non-finite headline value")
        if not -1.0 <= output.composite.score <= 1.0:
            fail(f"output {index}: composite score out of bounds")
        if not -1.0 <= output.composite.smoothed <= 1.0:
            fail(f"output {index}: smoothed score out of bounds")
        if not 0.0 <= output.composite.confidence <= 1.0:
            fail(f"output {index}: confidence out of bounds")
        if not (
            0.0
            <= output.threshold.exit
            <= output.threshold.watch
            <= output.threshold.entry
            <= 1.0
        ):
            fail(f"output {index}: threshold ordering invalid")
        if not (
            0.0 <= output.quality.book_quality <= 1.0
            and 0.0 <= output.quality.liquidity_score <= 1.0
        ):
            fail(f"output {index}: quality value out of bounds")
        for feature in output.features.values():
            lower = -1.0 if feature.kind is FeatureKind.DIRECTIONAL else 0.0
            if not (
                math.isfinite(feature.raw)
                and math.isfinite(feature.value)
                and math.isfinite(feature.local_confidence)
                and math.isfinite(feature.confidence)
            ):
                fail(f"output {index}: {feature.name} is non-finite")
            if not lower <= feature.value <= 1.0:
                fail(f"output {index}: {feature.name} value out of bounds")
            if not (
                0.0 <= feature.local_confidence <= 1.0
                and 0.0 <= feature.confidence <= 1.0
            ):
                fail(f"output {index}: {feature.name} confidence out of bounds")
        if output.position is not None:
            if not output.position.is_open:
                fail(f"output {index}: reported position is not open")
            if output.state not in (TradeState.LONG, TradeState.SHORT):
                fail(f"output {index}: position reported outside an open state")
        for quote in (output.entry_quote, output.exit_quote):
            if quote is not None:
                if not 0 <= quote.filled_quantity <= quote.requested_quantity:
                    fail(f"output {index}: quote quantity is invalid")
                if quote.complete != (
                    quote.filled_quantity == quote.requested_quantity
                ):
                    fail(f"output {index}: quote complete flag is inconsistent")
        if output.pnl is not None:
            pnl = output.pnl
            expected_gross = (
                (pnl.exit_price - pnl.entry_price)
                * pnl.direction.signum
                * pnl.quantity
            )
            if not math.isclose(pnl.gross_rupees, expected_gross, abs_tol=1e-8):
                fail(f"output {index}: gross PnL identity failed")
            if not math.isclose(
                pnl.net_rupees,
                pnl.gross_rupees - pnl.cost_rupees,
                abs_tol=1e-8,
            ):
                fail(f"output {index}: net PnL identity failed")
        if "! state reset after feed gap" in output.reasons:
            reset_indexes.append(index)
            if output.state is not TradeState.WARMUP or output.position is not None:
                fail(f"output {index}: reset did not produce flat WARMUP")
            recovery = outputs[index : index + 24]
            if any(item.state is not TradeState.WARMUP for item in recovery):
                fail(f"output {index}: entered before 24-event re-warmup evidence")

    if len(reset_indexes) != stats.resets:
        fail("visible reset evidence does not equal reset counter")

    stressed = _event_output(run, _STRESS_INDEX)

    def require_reset(raw_index: int, label: str) -> None:
        output = _event_output(run, raw_index)
        if output is None:
            fail(f"{label}: reset event was rejected")
        elif "! state reset after feed gap" not in output.reasons:
            fail(f"{label}: accepted event omitted visible reset evidence")
        elif output.state is not TradeState.WARMUP or output.position is not None:
            fail(f"{label}: reset event was not flat WARMUP")

    quality_contracts = {
        ExtremeStressPattern.LIQUIDITY_VACUUM: BlockReason.LIQUIDITY_BELOW_THRESHOLD,
        ExtremeStressPattern.SPREAD_EXPLOSION: BlockReason.SPREAD_TOO_WIDE,
        ExtremeStressPattern.DEPTH_EVAPORATION: BlockReason.INSUFFICIENT_DEPTH_LEVELS,
    }
    required_reason = quality_contracts.get(pattern)
    if required_reason is not None:
        if stressed is None:
            fail("stress event was unexpectedly rejected")
        elif required_reason not in stressed.quality.reasons:
            fail(f"stress event omitted quality block {required_reason.value}")
        elif stressed.state is not TradeState.EXIT_LONG:
            fail("quality disaster did not close the established long")

    if pattern is ExtremeStressPattern.FLASH_CRASH:
        crashes = [
            output
            for raw_index in range(_STRESS_INDEX, _STRESS_INDEX + 4)
            if (output := _event_output(run, raw_index)) is not None
        ]
        if not any(
            output.state is TradeState.EXIT_LONG
            and any("stop" in reason for reason in output.reasons)
            for output in crashes
        ):
            fail("flash crash did not produce a stop exit within four legs")

    if pattern is ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW:
        entries = _entry_outputs(run)
        directions = {output.state for output in entries}
        if directions != {TradeState.LONG, TradeState.SHORT}:
            fail("whipsaw did not exercise both directions")
        if not 2 <= len(entries) <= 14:
            fail(f"whipsaw entry churn is not bounded: {len(entries)}")
        if entries and min(output.composite.confidence for output in entries) < 0.45:
            fail("whipsaw entry confidence fell below the declared 0.45 floor")

    if pattern is ExtremeStressPattern.GAP_REVERSAL:
        require_reset(_STRESS_INDEX, "first price gap")
        require_reset(_STRESS_INDEX + 2, "reversal price gap")
        if stats.gaps != 2 or stats.resets != 2:
            fail("gap reversal did not record two independent gap resets")

    if pattern is ExtremeStressPattern.LIMIT_LOCK:
        if any(
            _event_output(run, raw_index) is not None
            for raw_index in range(_STRESS_INDEX, _STRESS_INDEX + 6)
        ):
            fail("one or more locked-book events were accepted")
        require_reset(_STRESS_INDEX + 6, "limit-lock recovery")

    if pattern is ExtremeStressPattern.HALT_AND_REOPEN:
        require_reset(_STRESS_INDEX, "halt/reopen")

    if pattern is ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED:
        forward_skip = _event_output(run, _STRESS_INDEX)
        backwards_time = _event_output(run, _STRESS_INDEX + 1)
        if forward_skip is None:
            fail("forward sequence skip was unexpectedly rejected")
        elif "! state reset after feed gap" in forward_skip.reasons:
            fail("forward sequence skip unexpectedly triggered a reset")
        if backwards_time is not None:
            fail("backwards-timestamp event was not rejected")
        require_reset(_STRESS_INDEX + 2, "sequence regression")

    if pattern is ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE:
        entries = _entry_outputs(run)
        if not entries or entries[0].entry_quote is None:
            fail("partial-fill pattern produced no entry quote")
        else:
            quote = entries[0].entry_quote
            position = entries[0].position
            if quote.complete or quote.requested_quantity != 5_000:
                fail("partial entry was not reported as requested")
            if quote.filled_quantity != 150 or position is None or position.quantity != 150:
                fail("partial entry position was not sized to 150 executed units")
        exits = _exit_outputs(run)
        if not exits or exits[0].exit_quote is None or not exits[0].exit_quote.complete:
            fail("partial entry did not receive a complete modelled exit")
        partial_stress = _event_output(run, _STRESS_INDEX)
        if partial_stress is None or partial_stress.state is not TradeState.EXIT_LONG:
            fail("partial-fill slippage event did not stop and exit the long")

    if pattern is ExtremeStressPattern.COMBINED_DISASTER:
        if stressed is None:
            fail("combined disaster's first accepted stress event is missing")
        elif not {
            BlockReason.SPREAD_TOO_WIDE,
            BlockReason.LIQUIDITY_BELOW_THRESHOLD,
        }.issubset(stressed.quality.reasons):
            fail("combined disaster omitted spread/liquidity block evidence")
        if any(
            _event_output(run, raw_index) is not None
            for raw_index in range(_STRESS_INDEX + 4, _STRESS_INDEX + 8)
        ):
            fail("combined disaster accepted a locked/out-of-order event")
        require_reset(_STRESS_INDEX + 8, "combined-disaster recovery")

    return failures


def _adverse_entry_ticks(output: EngineOutput) -> float:
    quote = output.entry_quote
    position = output.position
    if quote is None or position is None:
        return 0.0
    return (
        (quote.price - quote.reference_price)
        * position.direction.signum
        / output.snapshot.tick_size
    )


def _adverse_exit_ticks(output: EngineOutput) -> float:
    quote = output.exit_quote
    pnl = output.pnl
    if quote is None or pnl is None:
        return 0.0
    return (
        (quote.reference_price - quote.price)
        * pnl.direction.signum
        / output.snapshot.tick_size
    )


def audit_pattern(
    pattern: ExtremeStressPattern,
    count: int,
    seed: int,
) -> AuditResult:
    """Run, replay and summarise one named pattern."""
    snapshots = build_snapshots(pattern, count, seed)
    run = process(pattern, snapshots)
    replay_snapshots = build_snapshots(pattern, count, seed)
    replay = process(pattern, replay_snapshots)

    entries = _entry_outputs(run)
    exits = _exit_outputs(run)
    reset_indexes = [
        index
        for index, output in enumerate(run.outputs)
        if "! state reset after feed gap" in output.reasons
    ]
    resets_from_open = sum(
        index > 0 and run.outputs[index - 1].position is not None
        for index in reset_indexes
    )
    block_reasons = Counter(
        reason.value
        for output in run.outputs
        if not output.quality.tradable
        for reason in output.quality.reasons
    )
    state_counts = Counter(output.state.value for output in run.outputs)
    regime_counts = Counter(output.regime.regime.value for output in run.outputs)
    transitions = Counter(
        f"{output.transition.previous.value}->{output.transition.current.value}"
        for output in run.outputs
        if output.transition.changed
    )
    entry_slippage = [_adverse_entry_ticks(output) for output in entries]
    exit_slippage = [_adverse_exit_ticks(output) for output in exits]
    latencies = [output.compute_us for output in run.outputs]
    first_raw_digest = raw_digest(snapshots)
    replay_raw_digest = raw_digest(replay_snapshots)
    first_output_digest = output_digest(run.outputs)
    replay_output_digest = output_digest(replay.outputs)
    violations = invariant_violations(pattern, run, expected_count=count)
    if first_raw_digest != replay_raw_digest:
        violations.append("raw event stream differs on exact replay")
    if first_output_digest != replay_output_digest:
        violations.append("accepted output stream differs on exact replay")
    if run.stats != replay.stats:
        violations.append("engine counters differ on exact replay")

    return AuditResult(
        pattern=pattern,
        snapshots=len(snapshots),
        stats=run.stats,
        block_reasons=dict(block_reasons),
        state_counts=dict(state_counts),
        regime_counts=dict(regime_counts),
        transitions=dict(transitions),
        entries=len(entries),
        exits=len(exits),
        reset_evidence=len(reset_indexes),
        resets_from_open_position=resets_from_open,
        partial_entries=sum(
            output.entry_quote is not None and not output.entry_quote.complete
            for output in entries
        ),
        entry_slippage_mean_ticks=(
            statistics.fmean(entry_slippage) if entry_slippage else 0.0
        ),
        entry_slippage_max_ticks=max(entry_slippage, default=0.0),
        exit_slippage_mean_ticks=(
            statistics.fmean(exit_slippage) if exit_slippage else 0.0
        ),
        exit_slippage_max_ticks=max(exit_slippage, default=0.0),
        reported_gross_ticks=sum(
            output.pnl.gross_ticks for output in exits if output.pnl is not None
        ),
        reported_net_rupees=sum(
            output.pnl.net_rupees for output in exits if output.pnl is not None
        ),
        latency_p50_us=percentile(latencies, 0.50),
        latency_p90_us=percentile(latencies, 0.90),
        latency_p99_us=percentile(latencies, 0.99),
        latency_max_us=max(latencies),
        raw_digest=first_raw_digest[:16],
        output_digest=first_output_digest[:16],
        raw_deterministic=first_raw_digest == replay_raw_digest,
        output_deterministic=first_output_digest == replay_output_digest,
        violations=tuple(violations),
    )


def format_counts(counts: dict[str, int]) -> str:
    """Format a histogram in descending count then name order."""
    return ", ".join(
        f"{name}={count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ) or "none"


def print_result(result: AuditResult) -> None:
    """Print one dense, operator-readable pattern report."""
    stats = result.stats
    print(f"\n=== {result.pattern.value} ===")
    print(
        f"ticks={result.snapshots} accepted={stats.accepted} rejected={stats.rejected} "
        f"blocked={stats.blocked} gaps={stats.gaps} resets={stats.resets}"
    )
    print(f"reject reasons: {format_counts(dict(stats.rejects_by_reason))}")
    print(f"block reasons:  {format_counts(result.block_reasons)}")
    print(f"states:         {format_counts(result.state_counts)}")
    print(f"regimes:        {format_counts(result.regime_counts)}")
    print(f"transitions:    {format_counts(result.transitions)}")
    print(
        f"safety evidence: entries={result.entries} normal_exits={result.exits} "
        f"visible_resets={result.reset_evidence} "
        f"resets_after_open={result.resets_from_open_position} "
        f"partial_entries={result.partial_entries}"
    )
    print(
        f"modelled slippage: entry mean/max={result.entry_slippage_mean_ticks:.2f}/"
        f"{result.entry_slippage_max_ticks:.2f}t exit mean/max="
        f"{result.exit_slippage_mean_ticks:.2f}/{result.exit_slippage_max_ticks:.2f}t"
    )
    print(
        f"reported normal-exit PnL: gross={result.reported_gross_ticks:+.1f}t "
        f"net=Rs{result.reported_net_rupees:+.2f} "
        "(forced-gap PnL is intentionally not exposed by EngineOutput)"
    )
    print(
        f"latency: p50={result.latency_p50_us:.1f}us p90={result.latency_p90_us:.1f}us "
        f"p99={result.latency_p99_us:.1f}us max={result.latency_max_us:.1f}us"
    )
    print(
        f"determinism: raw={'PASS' if result.raw_deterministic else 'FAIL'} "
        f"({result.raw_digest}) output={'PASS' if result.output_deterministic else 'FAIL'} "
        f"({result.output_digest})"
    )
    print(
        f"invariants: {'PASS' if not result.violations else 'FAIL'} "
        f"violations={len(result.violations)}"
    )
    for violation in result.violations:
        print(f"  - {violation}")


def print_comparison(results: list[AuditResult]) -> None:
    """Print direct cross-pattern fail-safe and latency comparison."""
    print("\n=== EXTREME-STRESS COMPARISON ===")
    print(
        f"{'pattern':<28} {'acc':>4} {'rej':>4} {'blk':>4} {'gap':>4} {'rst':>4} "
        f"{'ent':>4} {'exit':>4} {'part':>4} {'p99us':>8} {'check':>6}"
    )
    for result in results:
        stats = result.stats
        print(
            f"{result.pattern.value:<28} {stats.accepted:4d} {stats.rejected:4d} "
            f"{stats.blocked:4d} {stats.gaps:4d} {stats.resets:4d} "
            f"{result.entries:4d} {result.exits:4d} {result.partial_entries:4d} "
            f"{result.latency_p99_us:8.1f} "
            f"{'PASS' if result.passed else 'FAIL':>6}"
        )


def run(count: int, seed: int) -> list[AuditResult]:
    """Audit all patterns in stable enum order."""
    return [audit_pattern(pattern, count, seed) for pattern in ExtremeStressPattern]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; non-zero means determinism or an invariant failed."""
    parser = argparse.ArgumentParser(
        prog="bench.extreme_stress_audit",
        description="Audit all deterministic adversarial fail-safe patterns.",
    )
    parser.add_argument("--count", type=int, default=_DEFAULT_COUNT, help="ticks per pattern")
    parser.add_argument("--seed", type=int, default=20_260_726, help="source seed")
    args = parser.parse_args(argv)
    if args.count < _STRESS_INDEX + 12:
        print(
            f"count must be at least {_STRESS_INDEX + 12} for stress and recovery",
            file=sys.stderr,
        )
        return 2

    logging.getLogger("snapshot_quant_v4").setLevel(logging.CRITICAL)
    print("Snapshot Quant V4 deterministic EXTREME_STRESS audit")
    print("Synthetic safety fixtures only; profit is not a pass criterion.")
    results = run(args.count, args.seed)
    for result in results:
        print_result(result)
    print_comparison(results)
    failed = [result for result in results if not result.passed]
    violations = sum(len(result.violations) for result in results)
    print(
        f"\nOVERALL: {'PASS' if not failed else 'FAIL'} "
        f"({len(results) - len(failed)}/{len(results)} patterns, "
        f"{violations} violations)"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
