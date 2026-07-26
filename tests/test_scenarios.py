"""Full-pipeline audit of deterministic scenario and pattern families.

These controlled fixtures are not claims about live market behaviour. Tests
verify that directional, multiple noise and multiple random paths retain valid
SnapQuote structure and produce coherent, bounded, deterministic engine output.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise

import pytest
from snapshot_quant_v4.adapter import (
    MarketScenario,
    NoisePattern,
    RandomPattern,
    ScenarioConfig,
    ScenarioSource,
    snapshot_to_json,
)
from snapshot_quant_v4.config import config_from_mapping
from snapshot_quant_v4.engine.quant_engine import EngineRegistry
from snapshot_quant_v4.utils.clock import ManualClock
from snapshot_quant_v4.utils.types import (
    BlockReason,
    Direction,
    EngineOutput,
    EngineStats,
    FeatureKind,
    MarketDataSource,
    Regime,
    TradeState,
)

TOKEN = "3045"
SYMBOL = "SBIN"
TICK_PAISE = 5
BASE_PAISE = 80_000
COUNT = 600


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    """One top-level scenario plus its active pattern selection."""

    scenario: MarketScenario
    noise_pattern: NoisePattern = NoisePattern.WHIPSAW
    random_pattern: RandomPattern = RandomPattern.UNBIASED_WALK

    @property
    def label(self) -> str:
        """Stable pytest/report identifier."""
        return ScenarioConfig(
            scenario=self.scenario,
            noise_pattern=self.noise_pattern,
            random_pattern=self.random_pattern,
        ).pattern_name


UPWARD_CASE = ScenarioCase(MarketScenario.UPWARD)
DOWNWARD_CASE = ScenarioCase(MarketScenario.DOWNWARD)
NOISE_CASES = tuple(
    ScenarioCase(MarketScenario.NOISE, noise_pattern=pattern)
    for pattern in NoisePattern
)
RANDOM_CASES = tuple(
    ScenarioCase(MarketScenario.RANDOM, random_pattern=pattern)
    for pattern in RandomPattern
)
ALL_CASES = (UPWARD_CASE, DOWNWARD_CASE, *NOISE_CASES, *RANDOM_CASES)


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """Captured outputs and counters for one complete deterministic run."""

    case: ScenarioCase
    outputs: tuple[EngineOutput, ...]
    stats: EngineStats


def audit_config():
    """Short-window profile used only to exercise the complete audit quickly."""
    return config_from_mapping(
        {
            "symbols": [
                {
                    "token": TOKEN,
                    "symbol": SYMBOL,
                    "exchange_type": 1,
                    "tick_size": 0.05,
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


def collect_source(
    case: ScenarioCase,
    *,
    count: int = COUNT,
    seed: int = 20_260_726,
):
    """Collect one exact scenario/pattern source sequence."""
    source = ScenarioSource(
        ScenarioConfig(
            scenario=case.scenario,
            noise_pattern=case.noise_pattern,
            random_pattern=case.random_pattern,
            token=TOKEN,
            symbol=SYMBOL,
            count=count,
            seed=seed,
        )
    )
    source.start()
    return source, list(source.snapshots())


def run_engine(
    case: ScenarioCase,
    *,
    count: int = COUNT,
    seed: int = 20_260_726,
) -> ScenarioRun:
    """Drive every source event through validation and all decision stages."""
    clock = ManualClock()
    registry = EngineRegistry(audit_config(), clock=clock)
    _, snapshots = collect_source(case, count=count, seed=seed)
    outputs: list[EngineOutput] = []
    for raw in snapshots:
        clock.advance(200.0)
        output = registry.process(raw)
        if output is not None:
            outputs.append(output)
    return ScenarioRun(case, tuple(outputs), registry.statistics()[TOKEN])


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile for latency sanity checks."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def deterministic_outputs(outputs: tuple[EngineOutput, ...]):
    """Retain every output field except nondeterministic measured compute time."""
    return tuple(replace(output, compute_us=0.0) for output in outputs)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.label)
def test_source_is_exactly_replayable_and_market_structurally_valid(
    case: ScenarioCase,
) -> None:
    source, first = collect_source(case, count=96)
    assert isinstance(source, MarketDataSource)
    assert source.pattern_name == case.label
    assert source.emitted == 96

    source.start()
    second = list(source.snapshots())
    assert first == second
    assert source.emitted == 96

    assert [raw.sequence_number for raw in first] == list(range(1, 97))
    assert all(
        later.exchange_timestamp_ms - earlier.exchange_timestamp_ms == 200
        for earlier, later in pairwise(first)
    )
    assert all(
        later.volume_traded_today > earlier.volume_traded_today
        for earlier, later in pairwise(first)
    )

    for raw in first:
        assert raw.token == TOKEN
        assert raw.symbol == SYMBOL
        assert len(raw.bids) == len(raw.asks) == 5
        assert raw.bids[0].price_paise < raw.asks[0].price_paise
        assert raw.last_traded_price in (raw.bids[0].price, raw.asks[0].price)
        assert raw.low_price <= raw.last_traded_price <= raw.high_price
        assert math.isfinite(raw.average_traded_price)
        assert raw.last_traded_quantity > 0
        assert raw.total_buy_quantity > 0.0
        assert raw.total_sell_quantity > 0.0
        assert all(
            left.price_paise > right.price_paise
            for left, right in pairwise(raw.bids)
        )
        assert all(
            left.price_paise < right.price_paise
            for left, right in pairwise(raw.asks)
        )
        for level in (*raw.bids, *raw.asks):
            assert level.price_paise % TICK_PAISE == 0
            assert level.price == level.price_paise / 100.0
            assert level.quantity > 0
            assert level.orders > 0


def test_source_lifecycle_stop_and_restart() -> None:
    source = ScenarioSource(ScenarioConfig(count=20))
    source.start()
    iterator = source.snapshots()
    assert next(iterator).sequence_number == 1
    source.stop()
    assert list(iterator) == []

    source.start()
    restarted = list(source.snapshots())
    assert len(restarted) == 20
    assert restarted[0].sequence_number == 1


def test_new_pattern_fields_preserve_the_existing_positional_config_api() -> None:
    config = ScenarioConfig(MarketScenario.NOISE, "TOKEN", "POSITIONAL")
    assert config.token == "TOKEN"
    assert config.symbol == "POSITIONAL"
    assert config.noise_pattern is NoisePattern.WHIPSAW
    assert config.random_pattern is RandomPattern.UNBIASED_WALK


def test_default_whipsaw_complete_stream_remains_backward_compatible() -> None:
    source = ScenarioSource(ScenarioConfig(scenario=MarketScenario.NOISE, count=16))
    source.start()
    payload = "\n".join(snapshot_to_json(raw) for raw in source.snapshots()).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "3a3a96774e5319f0dbdd2e2fe3444b511060a1e5ea3c666f9d919b7569ad67c0"
    )


def test_directional_price_paths_match_the_contract() -> None:
    _, upward = collect_source(UPWARD_CASE, count=32)
    _, downward = collect_source(DOWNWARD_CASE, count=32)
    upward_bids = [raw.bids[0].price_paise for raw in upward]
    downward_bids = [raw.bids[0].price_paise for raw in downward]
    assert {later - earlier for earlier, later in pairwise(upward_bids)} == {
        TICK_PAISE
    }
    assert {
        later - earlier for earlier, later in pairwise(downward_bids)
    } == {-TICK_PAISE}


@pytest.mark.parametrize("case", NOISE_CASES, ids=lambda case: case.label)
def test_each_noise_pattern_is_zero_sum_volatile_and_bidirectional(
    case: ScenarioCase,
) -> None:
    _, snapshots = collect_source(case, count=32)
    prices = [BASE_PAISE, *(raw.bids[0].price_paise for raw in snapshots)]
    changes = [
        (later - earlier) // TICK_PAISE for earlier, later in pairwise(prices)
    ]
    assert min(changes) < 0 < max(changes)
    assert max(abs(change) for change in changes) >= 6
    for cycle_start in range(0, 32, 8):
        assert sum(changes[cycle_start : cycle_start + 8]) == 0
        assert snapshots[cycle_start + 7].bids[0].price_paise == BASE_PAISE


def test_noise_patterns_are_economically_distinct() -> None:
    streams = [collect_source(case, count=32)[1] for case in NOISE_CASES]
    signatures = {
        tuple(raw.bids[0].price_paise for raw in stream) for stream in streams
    }
    assert len(signatures) == len(NoisePattern)


@pytest.mark.parametrize("case", RANDOM_CASES, ids=lambda case: case.label)
def test_each_random_pattern_is_seeded_but_not_hard_coded(case: ScenarioCase) -> None:
    _, first = collect_source(case, count=128, seed=11)
    _, replay = collect_source(case, count=128, seed=11)
    _, different = collect_source(case, count=128, seed=12)
    assert first == replay
    assert first != different
    assert len({raw.bids[0].price_paise for raw in first}) > 5
    assert len({raw.bids[0].quantity for raw in first}) > 10


def test_random_patterns_have_their_declared_path_characteristics() -> None:
    paths = {}
    for case in RANDOM_CASES:
        _, snapshots = collect_source(case)
        paths[case.random_pattern] = [
            (raw.bids[0].price_paise - BASE_PAISE) // TICK_PAISE
            for raw in snapshots
        ]

    assert len({tuple(path) for path in paths.values()}) == len(RandomPattern)
    mean_reverting = paths[RandomPattern.MEAN_REVERTING]
    assert max(abs(offset) for offset in mean_reverting) <= 10

    switching = paths[RandomPattern.TREND_SWITCHING]
    block_moves = [
        switching[start + 39] - switching[start]
        for start in range(0, 560, 40)
    ]
    assert min(block_moves) <= -30
    assert max(block_moves) >= 30
    assert all(left * right < 0 for left, right in pairwise(block_moves))


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.label)
def test_full_pipeline_accepts_every_tick_and_preserves_all_bounds(
    case: ScenarioCase,
) -> None:
    run = run_engine(case)
    outputs = run.outputs
    assert len(outputs) == COUNT
    assert run.stats.accepted == COUNT
    assert run.stats.rejected == 0
    assert run.stats.gaps == 0
    assert run.stats.resets == 0
    assert run.stats.rejects_by_reason == {}
    assert run.stats.blocked == sum(not output.quality.tradable for output in outputs)
    assert [output.snapshot_index for output in outputs] == list(range(1, COUNT + 1))

    previous_state = TradeState.WARMUP
    for output in outputs:
        assert output.symbol == SYMBOL
        assert output.token == TOKEN
        assert output.exchange_timestamp_ms == output.snapshot.exchange_timestamp_ms
        assert output.transition.previous is previous_state
        assert output.transition.current is output.state
        previous_state = output.state

        assert math.isfinite(output.compute_us) and output.compute_us > 0.0
        assert math.isfinite(output.composite.score)
        assert math.isfinite(output.composite.smoothed)
        assert math.isfinite(output.composite.confidence)
        assert -1.0 <= output.composite.score <= 1.0
        assert -1.0 <= output.composite.smoothed <= 1.0
        assert 0.0 <= output.composite.confidence <= 1.0
        assert output.composite.used_features >= 0
        if output.composite.valid:
            assert output.composite.used_features == len(output.composite.contributions)
            assert output.composite.weight_mass > 0.0
        else:
            assert output.composite.contributions == ()

        assert 0.0 <= output.threshold.exit <= output.threshold.watch
        assert output.threshold.watch <= output.threshold.entry <= 1.0
        assert 0.0 <= output.quality.book_quality <= 1.0
        assert 0.0 <= output.quality.liquidity_score <= 1.0
        assert math.isfinite(output.regime.efficiency_ratio)
        assert math.isfinite(output.regime.volatility_ticks)
        assert math.isfinite(output.regime.trend_ticks)
        assert 0.0 <= output.regime.efficiency_ratio <= 1.0
        assert output.regime.volatility_ticks >= 0.0

        assert len(output.features) == 10
        for feature in output.features.values():
            assert math.isfinite(feature.raw)
            assert math.isfinite(feature.value)
            assert math.isfinite(feature.local_confidence)
            assert math.isfinite(feature.confidence)
            assert 0.0 <= feature.local_confidence <= 1.0
            assert 0.0 <= feature.confidence <= 1.0
            if feature.kind is FeatureKind.DIRECTIONAL:
                assert -1.0 <= feature.value <= 1.0
            else:
                assert 0.0 <= feature.value <= 1.0

        contribution_names = [item.name for item in output.composite.contributions]
        assert len(contribution_names) == len(set(contribution_names))
        for item in output.composite.contributions:
            feature = output.features[item.name]
            assert feature.kind is FeatureKind.DIRECTIONAL
            assert feature.valid
            assert math.isfinite(item.contribution)
            assert item.weight >= 0.0
            assert 0.0 <= item.confidence <= 1.0

        if output.position is not None:
            assert output.position.is_open
            assert output.state in (TradeState.LONG, TradeState.SHORT)
        if output.entry_quote is not None:
            assert output.entry_quote.filled_quantity <= output.entry_quote.requested_quantity
            assert output.entry_quote.complete == (
                output.entry_quote.filled_quantity == output.entry_quote.requested_quantity
            )
        if output.exit_quote is not None:
            assert output.exit_quote.filled_quantity <= output.exit_quote.requested_quantity
        if output.pnl is not None:
            expected_gross = (
                (output.pnl.exit_price - output.pnl.entry_price)
                * output.pnl.direction.signum
                * output.pnl.quantity
            )
            assert output.pnl.gross_rupees == pytest.approx(expected_gross)
            assert output.pnl.gross_ticks == pytest.approx(
                expected_gross / (0.05 * output.pnl.quantity)
            )
            assert output.pnl.net_rupees == pytest.approx(
                output.pnl.gross_rupees - output.pnl.cost_rupees
            )
            assert output.pnl.cost_rupees >= 0.0
            assert output.pnl.holding_ms >= 0.0

    assert all(BlockReason.WARMUP in output.quality.reasons for output in outputs[:24])
    timings = [output.compute_us for output in outputs]
    p50 = percentile(timings, 0.50)
    p90 = percentile(timings, 0.90)
    p99 = percentile(timings, 0.99)
    assert 0.0 < p50 <= p90 <= p99 <= max(timings) < 50_000.0


@pytest.mark.parametrize(
    ("case", "watch", "position", "opposite", "direction", "signum"),
    [
        (
            UPWARD_CASE,
            TradeState.WATCH_LONG,
            TradeState.LONG,
            TradeState.SHORT,
            Direction.LONG,
            1.0,
        ),
        (
            DOWNWARD_CASE,
            TradeState.WATCH_SHORT,
            TradeState.SHORT,
            TradeState.LONG,
            Direction.SHORT,
            -1.0,
        ),
    ],
)
def test_directional_scenarios_reach_expected_features_states_and_execution(
    case: ScenarioCase,
    watch: TradeState,
    position: TradeState,
    opposite: TradeState,
    direction: Direction,
    signum: float,
) -> None:
    outputs = run_engine(case).outputs
    states = [output.state for output in outputs]
    assert TradeState.WARMUP in states
    assert TradeState.NEUTRAL in states
    assert watch in states
    assert position in states
    assert opposite not in states
    assert (TradeState.EXIT_LONG if signum > 0 else TradeState.EXIT_SHORT) in states

    mature = outputs[60:]
    assert statistics.fmean(output.composite.smoothed for output in mature) * signum > 0.30
    assert statistics.fmean(output.composite.confidence for output in mature) > 0.75
    assert statistics.fmean(
        output.features["weighted_obi"].value for output in mature
    ) * signum > 0.70
    assert statistics.fmean(
        output.features["microprice"].value for output in mature
    ) * signum > 0.70
    assert statistics.fmean(
        output.features["ltp_confirmation"].value for output in mature
    ) * signum > 0.90
    assert all(output.regime.regime is Regime.TREND for output in mature)

    entry = next(output for output in outputs if output.state is position)
    assert entry.position is not None
    assert entry.position.direction is direction
    assert entry.entry_quote is not None and entry.entry_quote.complete
    expected_entry_touch = (
        entry.snapshot.best_ask.price
        if direction is Direction.LONG
        else entry.snapshot.best_bid.price
    )
    assert entry.entry_quote.reference_price == expected_entry_touch
    price_in_ticks = entry.entry_quote.price / 0.05
    assert price_in_ticks == pytest.approx(round(price_in_ticks), abs=1e-9)

    exit_state = TradeState.EXIT_LONG if direction is Direction.LONG else TradeState.EXIT_SHORT
    exited = next(output for output in outputs if output.state is exit_state)
    assert exited.pnl is not None
    assert exited.exit_quote is not None and exited.exit_quote.complete
    assert exited.pnl.direction is direction
    expected_exit_touch = (
        exited.snapshot.best_bid.price
        if direction is Direction.LONG
        else exited.snapshot.best_ask.price
    )
    assert exited.exit_quote.reference_price == expected_exit_touch


@pytest.mark.parametrize("case", NOISE_CASES, ids=lambda case: case.label)
def test_each_noise_pattern_is_detected_without_directional_trades(
    case: ScenarioCase,
) -> None:
    outputs = run_engine(case).outputs
    mature = outputs[60:]
    directional_states = {
        TradeState.WATCH_LONG,
        TradeState.WATCH_SHORT,
        TradeState.LONG,
        TradeState.SHORT,
        TradeState.EXIT_LONG,
        TradeState.EXIT_SHORT,
    }
    assert sum(output.regime.regime is Regime.NOISE for output in outputs) > COUNT * 0.90
    assert statistics.fmean(output.regime.efficiency_ratio for output in mature) < 0.05
    assert statistics.fmean(output.regime.volatility_ticks for output in mature) > 3.0
    assert statistics.fmean(output.composite.confidence for output in mature) < 0.08
    assert statistics.fmean(output.threshold.entry for output in mature) > 0.70
    assert abs(statistics.fmean(output.composite.smoothed for output in mature)) < 0.03
    assert all(output.state not in directional_states for output in outputs)


@pytest.mark.parametrize("case", RANDOM_CASES, ids=lambda case: case.label)
def test_each_random_pipeline_is_reproducible_without_assuming_direction(
    case: ScenarioCase,
) -> None:
    first = run_engine(case, seed=99).outputs
    replay = run_engine(case, seed=99).outputs
    different = run_engine(case, seed=100).outputs

    assert deterministic_outputs(first) == deterministic_outputs(replay)
    assert deterministic_outputs(first) != deterministic_outputs(different)
    assert {output.regime.regime for output in first} <= set(Regime)
    transition_counts = Counter(
        output.transition.current for output in first if output.transition.changed
    )
    entries = transition_counts[TradeState.LONG] + transition_counts[TradeState.SHORT]
    assert entries <= 15
