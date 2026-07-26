"""Observable full-pipeline audit for four deterministic market scenarios.

This is a software-behaviour audit, not a backtest and not evidence of expected
market returns. It sends every synthetic tick through the production validator,
statistics, features, regime, quality, confidence, composite, threshold, state
machine and execution model, then reports distributions rather than hiding the
result behind pass/fail assertions.

Usage::

    python bench/scenario_audit.py
    python bench/scenario_audit.py --count 1000 --seed 20260726
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapshot_quant_v4.adapter import MarketScenario, ScenarioConfig, ScenarioSource
from snapshot_quant_v4.config import config_from_mapping
from snapshot_quant_v4.engine.quant_engine import EngineRegistry
from snapshot_quant_v4.utils.clock import ManualClock
from snapshot_quant_v4.utils.types import EngineOutput, EngineStats, FeatureKind, RawSnapshot

_TOKEN = "3045"
_SYMBOL = "SBIN"
_TICK_SIZE = 0.05
_WARMUP_CUTOFF = 60


@dataclass(frozen=True, slots=True)
class FeatureAudit:
    """Mean mature value, confidence and weighted contribution for a feature."""

    name: str
    mean_value: float
    mean_confidence: float
    mean_contribution: float


@dataclass(frozen=True, slots=True)
class AuditResult:
    """All metrics produced for one named scenario."""

    scenario: MarketScenario
    snapshots: int
    stats: EngineStats
    price_delta_ticks: float
    tradable: int
    regime_counts: dict[str, int]
    state_counts: dict[str, int]
    transitions: dict[str, int]
    score_mean: float
    score_min: float
    score_max: float
    positive_ratio: float
    negative_ratio: float
    confidence_mean: float
    threshold_mean: float
    efficiency_mean: float
    volatility_mean: float
    feature_audits: tuple[FeatureAudit, ...]
    entries: int
    exits: int
    realised_gross_ticks: float
    realised_net_rupees: float
    latency_p50_us: float
    latency_p90_us: float
    latency_p99_us: float
    latency_max_us: float
    decision_digest: str
    deterministic: bool
    violations: tuple[str, ...]


def audit_config():
    """Build the same short-window research profile used by scenario tests."""
    return config_from_mapping(
        {
            "symbols": [
                {
                    "token": _TOKEN,
                    "symbol": _SYMBOL,
                    "exchange_type": 1,
                    "tick_size": _TICK_SIZE,
                    "quantity": 1,
                }
            ],
            "quality": {"warmup_snapshots": 25, "min_relative_depth": 0.2},
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
            "threshold": {"reference_window": 60, "base": 0.22, "min_entry": 0.15},
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
                "max_hold_ms": 120_000.0,
                "cooldown_ms": 200.0,
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
    scenario: MarketScenario,
    count: int,
    seed: int,
) -> list[RawSnapshot]:
    """Materialise one deterministic scenario before timing the engine."""
    source = ScenarioSource(
        ScenarioConfig(
            scenario=scenario,
            token=_TOKEN,
            symbol=_SYMBOL,
            count=count,
            seed=seed,
        )
    )
    source.start()
    return list(source.snapshots())


def process(snapshots: list[RawSnapshot]) -> tuple[list[EngineOutput], EngineStats]:
    """Process a pre-generated stream while advancing deterministic time."""
    clock = ManualClock()
    registry = EngineRegistry(audit_config(), clock=clock)
    outputs: list[EngineOutput] = []
    for snapshot in snapshots:
        clock.advance(200.0)
        output = registry.process(snapshot)
        if output is not None:
            outputs.append(output)
    return outputs, registry.statistics()[_TOKEN]


def decision_digest(outputs: list[EngineOutput]) -> str:
    """Hash every deterministic decision field, excluding measured latency."""
    records = [
        (
            output.snapshot_index,
            output.state.value,
            output.transition.previous.value,
            output.regime.regime.value,
            output.composite.score,
            output.composite.smoothed,
            output.composite.confidence,
            output.composite.valid,
            output.threshold.entry,
            output.threshold.watch,
            output.threshold.exit,
            output.quality.tradable,
            tuple(reason.value for reason in output.quality.reasons),
            tuple(
                (
                    name,
                    feature.raw,
                    feature.value,
                    feature.confidence,
                    feature.valid,
                )
                for name, feature in output.features.items()
            ),
        )
        for output in outputs
    ]
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty sample."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def invariant_violations(
    snapshots: list[RawSnapshot],
    outputs: list[EngineOutput],
    stats: EngineStats,
) -> list[str]:
    """Return concrete structural/numerical failures, capped for readability."""
    failures: list[str] = []

    def fail(message: str) -> None:
        if len(failures) < 20:
            failures.append(message)

    if stats.accepted + stats.rejected != len(snapshots):
        fail("accepted + rejected does not equal source count")
    if stats.accepted != len(outputs):
        fail("accepted counter does not equal output count")
    previous_timestamp = -1
    previous_volume = -1
    for index, raw in enumerate(snapshots, 1):
        if raw.sequence_number != index:
            fail(f"tick {index}: non-contiguous sequence")
        if raw.exchange_timestamp_ms <= previous_timestamp:
            fail(f"tick {index}: non-monotonic timestamp")
        if raw.volume_traded_today <= previous_volume:
            fail(f"tick {index}: non-increasing cumulative volume")
        previous_timestamp = raw.exchange_timestamp_ms
        previous_volume = raw.volume_traded_today
        if len(raw.bids) != 5 or len(raw.asks) != 5:
            fail(f"tick {index}: book does not contain five levels per side")
        if raw.bids[0].price_paise >= raw.asks[0].price_paise:
            fail(f"tick {index}: locked or crossed book")
        if any(level.price_paise % 5 for level in (*raw.bids, *raw.asks)):
            fail(f"tick {index}: off-tick depth price")

    previous_state = None
    for output in outputs:
        index = output.snapshot_index
        if previous_state is not None and output.transition.previous is not previous_state:
            fail(f"output {index}: broken state transition chain")
        previous_state = output.state
        if output.transition.current is not output.state:
            fail(f"output {index}: transition.current differs from state")
        if not (
            math.isfinite(output.composite.score)
            and math.isfinite(output.composite.smoothed)
            and math.isfinite(output.composite.confidence)
            and math.isfinite(output.compute_us)
        ):
            fail(f"output {index}: non-finite headline metric")
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
        for feature in output.features.values():
            low = -1.0 if feature.kind is FeatureKind.DIRECTIONAL else 0.0
            if not (
                math.isfinite(feature.raw)
                and math.isfinite(feature.value)
                and math.isfinite(feature.confidence)
            ):
                fail(f"output {index}: {feature.name} is non-finite")
            if not low <= feature.value <= 1.0:
                fail(f"output {index}: {feature.name} value out of bounds")
            if not 0.0 <= feature.confidence <= 1.0:
                fail(f"output {index}: {feature.name} confidence out of bounds")
        if output.pnl is not None:
            expected_gross = (
                (output.pnl.exit_price - output.pnl.entry_price)
                * output.pnl.direction.signum
                * output.pnl.quantity
            )
            if not math.isclose(output.pnl.gross_rupees, expected_gross, abs_tol=1e-9):
                fail(f"output {index}: PnL gross-rupee identity failed")
            if not math.isclose(
                output.pnl.net_rupees,
                output.pnl.gross_rupees - output.pnl.cost_rupees,
                abs_tol=1e-9,
            ):
                fail(f"output {index}: PnL net identity failed")
    return failures


def feature_summary(outputs: list[EngineOutput]) -> tuple[FeatureAudit, ...]:
    """Aggregate mature feature values/confidence/contributions."""
    values: dict[str, list[float]] = defaultdict(list)
    confidences: dict[str, list[float]] = defaultdict(list)
    contributions: dict[str, list[float]] = defaultdict(list)
    for output in outputs[_WARMUP_CUTOFF:]:
        for name, feature in output.features.items():
            if feature.valid and feature.kind is FeatureKind.DIRECTIONAL:
                values[name].append(feature.value)
                confidences[name].append(feature.confidence)
        for contribution in output.composite.contributions:
            contributions[contribution.name].append(contribution.contribution)

    summaries = []
    for name in sorted(values):
        feature_values = values[name]
        feature_confidences = confidences[name]
        feature_contributions = contributions[name]
        summaries.append(
            FeatureAudit(
                name=name,
                mean_value=statistics.fmean(feature_values),
                mean_confidence=statistics.fmean(feature_confidences),
                mean_contribution=(
                    statistics.fmean(feature_contributions)
                    if feature_contributions
                    else 0.0
                ),
            )
        )
    return tuple(
        sorted(summaries, key=lambda item: abs(item.mean_contribution), reverse=True)
    )


def audit_scenario(scenario: MarketScenario, count: int, seed: int) -> AuditResult:
    """Run and summarise one scenario, including an independent replay check."""
    snapshots = build_snapshots(scenario, count, seed)
    outputs, stats = process(snapshots)
    replay_outputs, _ = process(build_snapshots(scenario, count, seed))
    digest = decision_digest(outputs)
    replay_digest = decision_digest(replay_outputs)
    violations = invariant_violations(snapshots, outputs, stats)
    if digest != replay_digest:
        violations.append("decision sequence differs on exact replay")

    mature = outputs[min(_WARMUP_CUTOFF, len(outputs) - 1) :]
    scores = [output.composite.smoothed for output in mature]
    confidences = [output.composite.confidence for output in mature]
    thresholds = [output.threshold.entry for output in mature]
    efficiencies = [output.regime.efficiency_ratio for output in mature]
    volatilities = [output.regime.volatility_ticks for output in mature]
    latencies = [output.compute_us for output in outputs]
    state_counts = Counter(output.state.value for output in outputs)
    regime_counts = Counter(output.regime.regime.value for output in outputs)
    transitions = Counter(
        f"{output.transition.previous.value}->{output.transition.current.value}"
        for output in outputs
        if output.transition.changed
    )
    exits = [
        output
        for output in outputs
        if output.state.value in ("EXIT_LONG", "EXIT_SHORT") and output.pnl is not None
    ]
    entries = sum(
        output.transition.changed and output.state.value in ("LONG", "SHORT")
        for output in outputs
    )
    positive = sum(score > 1e-12 for score in scores)
    negative = sum(score < -1e-12 for score in scores)
    tick_delta = (
        snapshots[-1].bids[0].price - snapshots[0].bids[0].price
    ) / _TICK_SIZE

    return AuditResult(
        scenario=scenario,
        snapshots=len(snapshots),
        stats=stats,
        price_delta_ticks=tick_delta,
        tradable=sum(output.quality.tradable for output in outputs),
        regime_counts=dict(regime_counts),
        state_counts=dict(state_counts),
        transitions=dict(transitions),
        score_mean=statistics.fmean(scores),
        score_min=min(scores),
        score_max=max(scores),
        positive_ratio=positive / len(scores),
        negative_ratio=negative / len(scores),
        confidence_mean=statistics.fmean(confidences),
        threshold_mean=statistics.fmean(thresholds),
        efficiency_mean=statistics.fmean(efficiencies),
        volatility_mean=statistics.fmean(volatilities),
        feature_audits=feature_summary(outputs),
        entries=entries,
        exits=len(exits),
        realised_gross_ticks=sum(output.pnl.gross_ticks for output in exits if output.pnl),
        realised_net_rupees=sum(output.pnl.net_rupees for output in exits if output.pnl),
        latency_p50_us=percentile(latencies, 0.50),
        latency_p90_us=percentile(latencies, 0.90),
        latency_p99_us=percentile(latencies, 0.99),
        latency_max_us=max(latencies),
        decision_digest=digest[:16],
        deterministic=digest == replay_digest,
        violations=tuple(violations),
    )


def format_counts(counts: dict[str, int]) -> str:
    """Format a histogram in descending count order."""
    return ", ".join(
        f"{name}={count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ) or "none"


def print_result(result: AuditResult) -> None:
    """Print one dense, operator-readable scenario report."""
    stats = result.stats
    print(f"\n=== {result.scenario.value} ===")
    print(
        f"ticks={result.snapshots} accepted={stats.accepted} rejected={stats.rejected} "
        f"blocked={stats.blocked} tradable={result.tradable} "
        f"price_delta={result.price_delta_ticks:+.1f}t"
    )
    print(f"regimes: {format_counts(result.regime_counts)}")
    print(f"states:  {format_counts(result.state_counts)}")
    print(f"transitions: {format_counts(result.transitions)}")
    print(
        f"composite mature mean={result.score_mean:+.4f} "
        f"range=[{result.score_min:+.4f},{result.score_max:+.4f}] "
        f"positive={result.positive_ratio:.1%} negative={result.negative_ratio:.1%}"
    )
    print(
        f"confidence={result.confidence_mean:.4f} threshold={result.threshold_mean:.4f} "
        f"efficiency={result.efficiency_mean:.4f} volatility={result.volatility_mean:.3f}t"
    )
    top_features = result.feature_audits[:5]
    print(
        "top directional features: "
        + ", ".join(
            f"{item.name}(v={item.mean_value:+.3f},c={item.mean_confidence:.3f},"
            f"contrib={item.mean_contribution:+.4f})"
            for item in top_features
        )
    )
    print(
        f"execution: entries={result.entries} exits={result.exits} "
        f"realised_gross={result.realised_gross_ticks:+.1f}t "
        f"realised_net=Rs{result.realised_net_rupees:+.4f}"
    )
    print(
        f"latency: p50={result.latency_p50_us:.1f}us p90={result.latency_p90_us:.1f}us "
        f"p99={result.latency_p99_us:.1f}us max={result.latency_max_us:.1f}us"
    )
    print(
        f"determinism: {'PASS' if result.deterministic else 'FAIL'} "
        f"digest={result.decision_digest}  "
        f"invariants: {'PASS' if not result.violations else 'FAIL'} "
        f"violations={len(result.violations)}"
    )
    for violation in result.violations:
        print(f"  - {violation}")


def run(count: int, seed: int) -> list[AuditResult]:
    """Audit all four scenarios in declaration order."""
    return [audit_scenario(scenario, count, seed) for scenario in MarketScenario]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; non-zero means an invariant or determinism failure."""
    parser = argparse.ArgumentParser(
        prog="bench.scenario_audit",
        description="Audit four deterministic tick-by-tick market scenarios.",
    )
    parser.add_argument("--count", type=int, default=600, help="ticks per scenario")
    parser.add_argument("--seed", type=int, default=20_260_726, help="scenario seed")
    args = parser.parse_args(argv)
    if args.count <= _WARMUP_CUTOFF:
        print(f"count must be greater than {_WARMUP_CUTOFF}", file=sys.stderr)
        return 2

    print("Snapshot Quant V4 deterministic scenario audit")
    print("Synthetic software fixtures only; this is not a backtest or return estimate.")
    results = run(args.count, args.seed)
    for result in results:
        print_result(result)
    failures = sum(len(result.violations) for result in results)
    print(f"\nOVERALL: {'PASS' if failures == 0 else 'FAIL'} ({failures} violations)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
