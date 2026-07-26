"""End-to-end pipeline tests.

These tests answer questions that no unit test can:

*   does a full engine actually reach an entry, or does some interaction between
    confidence, threshold and the state machine make trading unreachable?
*   is a feed gap really followed by a reset and a fresh warmup?
*   is the composite score genuinely bounded, and the reported latency real?
*   does the whole runner, renderer and recorder wiring work together?

The market conditions are constructed deterministically rather than sampled, so a
failure points at the engine and not at a random seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from snapshot_quant_v4.adapter.replay import ReplaySource, SnapshotRecorder
from snapshot_quant_v4.adapter.synthetic import SyntheticConfig, SyntheticSource
from snapshot_quant_v4.engine.quant_engine import EngineRegistry, SymbolEngine, build_features
from snapshot_quant_v4.features.base import Feature, FeatureContext
from snapshot_quant_v4.output.console import NullRenderer, PlainRenderer, create_renderer
from snapshot_quant_v4.output.views import build_view, score_bar
from snapshot_quant_v4.runner import EngineRunner, LatencyTracker, drain_outputs
from snapshot_quant_v4.utils.clock import ManualClock
from snapshot_quant_v4.utils.types import FeatureKind, FeatureValue, TradeState

from conftest import BASE_PAISE, TICK_PAISE, TOKEN, SnapshotStream, build_config, make_snapshot

#: Configuration that shortens every warmup so a test can reach a decision in a
#: few hundred snapshots instead of several thousand.
FAST_CONFIG_SECTIONS = {
    "quality": {"warmup_snapshots": 25, "min_relative_depth": 0.2},
    "confidence": {
        "liquidity_window": 60,
        "volatility_window": 60,
        "stability_window": 20,
    },
    "regime": {"window": 40, "min_samples": 20, "min_dwell_snapshots": 2, "confirm_snapshots": 2},
    "threshold": {"reference_window": 60, "base": 0.22, "min_entry": 0.15},
    "composite": {"min_features": 3, "min_confidence": 0.05, "smoothing_half_life_ms": 300.0},
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
}


def fast_config(**overrides):
    """Build the short-warmup configuration, with optional section overrides."""
    sections = {key: dict(value) for key, value in FAST_CONFIG_SECTIONS.items()}
    for key, value in overrides.items():
        merged = sections.get(key, {})
        merged.update(value)
        sections[key] = merged
    return build_config(**sections)


def rising_bid_pressure(count: int, *, stream: SnapshotStream) -> list:
    """A book that drifts up one tick at a time with persistent bid dominance.

    Deliberately orderly: a one-tick drift, a one-tick spread, steady volume and a
    traded price that follows the mid. This is the condition under which the
    engine is *supposed* to take a long, so if it cannot, something is wrong.
    """
    snapshots = []
    for step in range(count):
        base = BASE_PAISE + (step // 3) * TICK_PAISE
        snapshots.append(
            stream.next(
                bid_paise=base,
                ask_paise=base + TICK_PAISE,
                bid_quantities=(2_400, 2_000, 1_600, 1_200, 900),
                ask_quantities=(400, 350, 300, 250, 200),
                last_traded_price=(base + TICK_PAISE) / 100.0,
                traded=120,
                total_buy_quantity=90_000.0,
                total_sell_quantity=40_000.0,
                interval_ms=200,
            )
        )
    return snapshots


class TestPipelineReachesDecisions:
    def test_engine_takes_a_long_in_a_bid_dominated_uptrend(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        outputs = drain_outputs(registry, rising_bid_pressure(180, stream=stream))

        states = [output.state for output in outputs]
        assert TradeState.WARMUP in states
        assert TradeState.NEUTRAL in states
        assert TradeState.WATCH_LONG in states, "engine never armed a long watch"
        assert TradeState.LONG in states, "engine never entered a long"

        entry = next(output for output in outputs if output.state is TradeState.LONG)
        assert entry.position is not None
        assert entry.composite.smoothed > 0.0
        assert entry.composite.confidence >= 0.10
        assert entry.quality.tradable

    def test_engine_takes_a_short_in_an_ask_dominated_downtrend(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        snapshots = []
        for step in range(180):
            base = BASE_PAISE - (step // 3) * TICK_PAISE
            snapshots.append(
                stream.next(
                    bid_paise=base,
                    ask_paise=base + TICK_PAISE,
                    bid_quantities=(400, 350, 300, 250, 200),
                    ask_quantities=(2_400, 2_000, 1_600, 1_200, 900),
                    last_traded_price=base / 100.0,
                    traded=120,
                    total_buy_quantity=40_000.0,
                    total_sell_quantity=90_000.0,
                    interval_ms=200,
                )
            )
        outputs = drain_outputs(registry, snapshots)
        assert TradeState.SHORT in [output.state for output in outputs]

    def test_composite_and_confidence_stay_in_range(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        source = SyntheticSource(SyntheticConfig(count=600, token=TOKEN, symbol="SBIN"))
        source.start()
        outputs = drain_outputs(registry, source.snapshots())
        assert outputs
        for output in outputs:
            assert -1.0 <= output.composite.score <= 1.0
            assert -1.0 <= output.composite.smoothed <= 1.0
            assert 0.0 <= output.composite.confidence <= 1.0
            assert 0.0 <= output.threshold.entry <= 1.0
            assert output.threshold.watch <= output.threshold.entry
            for feature in output.features.values():
                if feature.kind is FeatureKind.DIRECTIONAL:
                    assert -1.0 <= feature.value <= 1.0
                else:
                    assert 0.0 <= feature.value <= 1.0
                assert 0.0 <= feature.confidence <= 1.0

    def test_quality_features_never_enter_the_score(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        outputs = drain_outputs(registry, rising_bid_pressure(120, stream=stream))
        contributing = {
            contribution.name
            for output in outputs
            for contribution in output.composite.contributions
        }
        assert "spread" not in contributing
        assert "spread_compression" not in contributing

    def test_reported_latency_is_positive_and_plausible(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        outputs = drain_outputs(registry, rising_bid_pressure(100, stream=stream))
        assert all(output.compute_us > 0.0 for output in outputs)
        # A generous ceiling: this asserts the measurement is real, not that the
        # machine running the tests is fast.
        assert max(output.compute_us for output in outputs) < 50_000.0


class TestGapHandling:
    def test_gap_forces_flat_resets_estimators_and_restarts_warmup(self) -> None:
        config = fast_config(validation={"max_snapshot_gap_ms": 1_000.0})
        registry = EngineRegistry(config, clock=ManualClock())
        engine = registry.engines[TOKEN]
        stream = SnapshotStream()

        outputs = drain_outputs(registry, rising_bid_pressure(180, stream=stream))
        assert TradeState.LONG in [output.state for output in outputs]
        assert engine.shared_statistics.snapshots > 25

        # A three second silence: everything spanning it is invalid.
        gapped = stream.next(
            bid_paise=BASE_PAISE,
            ask_paise=BASE_PAISE + TICK_PAISE,
            interval_ms=3_000,
            traded=500,
        )
        output = registry.process(gapped)
        assert output is not None
        assert output.state is TradeState.WARMUP
        assert engine.statistics.gaps >= 1
        assert engine.statistics.resets >= 1
        assert engine.shared_statistics.snapshots == 1
        assert any("state reset" in reason for reason in output.reasons)

    def test_position_is_closed_when_a_gap_arrives(self) -> None:
        config = fast_config(validation={"max_snapshot_gap_ms": 1_000.0})
        registry = EngineRegistry(config, clock=ManualClock())
        engine = registry.engines[TOKEN]
        stream = SnapshotStream()
        drain_outputs(registry, rising_bid_pressure(180, stream=stream))
        assert engine.state in (TradeState.LONG, TradeState.WATCH_LONG, TradeState.NEUTRAL)

        gapped = stream.next(interval_ms=5_000, traded=100)
        registry.process(gapped)
        assert engine.state is TradeState.WARMUP

    def test_external_gap_notification_triggers_a_reset(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        engine = registry.engines[TOKEN]
        stream = SnapshotStream()
        drain_outputs(registry, rising_bid_pressure(60, stream=stream))
        before = engine.shared_statistics.snapshots
        assert before > 1

        registry.notify_gap("websocket reconnected")
        registry.process(stream.next())
        assert engine.shared_statistics.snapshots == 1
        assert engine.statistics.resets >= 1

    def test_rejected_snapshots_do_not_touch_the_estimators(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        engine = registry.engines[TOKEN]
        stream = SnapshotStream()
        drain_outputs(registry, rising_bid_pressure(40, stream=stream))
        before = engine.shared_statistics.snapshots

        crossed = stream.next(bid_paise=BASE_PAISE + 100, ask_paise=BASE_PAISE - 100)
        assert registry.process(crossed) is None
        assert engine.shared_statistics.snapshots == before
        assert engine.statistics.rejected == 1
        assert engine.statistics.rejects_by_reason["CROSSED_BOOK"] == 1


class TestRegistry:
    def test_unknown_tokens_are_counted_not_fatal(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        foreign = make_snapshot()
        object.__setattr__(foreign, "token", "9999")
        assert registry.process(foreign) is None
        assert registry.unknown_tokens["9999"] == 1

    def test_instruments_keep_independent_state(self) -> None:
        config = build_config(
            symbols=[
                {"token": "3045", "symbol": "SBIN"},
                {"token": "1594", "symbol": "INFY"},
            ],
            **FAST_CONFIG_SECTIONS,
        )
        registry = EngineRegistry(config, clock=ManualClock())
        stream = SnapshotStream()
        for raw in rising_bid_pressure(60, stream=stream):
            registry.process(raw)
        assert registry.engines["3045"].shared_statistics.snapshots == 60
        assert registry.engines["1594"].shared_statistics.snapshots == 0

    def test_empty_symbol_list_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            EngineRegistry(build_config(symbols=[]))


class TestFeatureInjection:
    def test_a_custom_feature_can_be_injected(self) -> None:
        class ConstantFeature(Feature):
            name = "constant"
            kind = FeatureKind.DIRECTIONAL

            def compute(self, context: FeatureContext) -> FeatureValue:
                return self._value(raw=1.0, value=1.0, detail="constant")

            def reset(self) -> None:
                return None

        config = fast_config(
            composite={"weights": {"UNKNOWN": {"constant": 1.0}}, "min_features": 1}
        )
        engine = SymbolEngine(
            config.symbols[0],
            config,
            clock=ManualClock(),
            features=[ConstantFeature()],
        )
        stream = SnapshotStream()
        output = None
        # Enough snapshots to clear warmup: before that the confidence model
        # correctly refuses to trust anything, including a constant.
        for raw in stream.many(40):
            output = engine.process(raw)
        assert output is not None
        assert output.features["constant"].value == 1.0
        assert output.composite.valid
        assert output.composite.score == pytest.approx(1.0)

    def test_dependency_order_is_validated_at_construction(self) -> None:
        config = fast_config()
        features = list(build_features(config.features))
        features.reverse()  # dependents now precede their dependencies
        with pytest.raises(ValueError) as info:
            SymbolEngine(config.symbols[0], config, features=features)
        assert "depends on" in str(info.value)

    def test_duplicate_feature_names_are_rejected(self) -> None:
        config = fast_config()
        features = list(build_features(config.features))
        with pytest.raises(ValueError):
            SymbolEngine(
                config.symbols[0], config, features=[features[0], features[0]]
            )


class TestRunnerAndOutput:
    def test_runner_processes_a_replayed_session_and_records_it(
        self, tmp_path: Path
    ) -> None:
        recording = tmp_path / "session.jsonl"
        stream = SnapshotStream()
        with SnapshotRecorder(recording) as recorder:
            for raw in rising_bid_pressure(120, stream=stream):
                recorder.write(raw)

        config = fast_config(runtime={"render_fps": 0.0, "log_file": None})
        registry = EngineRegistry(config, clock=ManualClock())
        source = ReplaySource(recording, strict=True)
        runner = EngineRunner(
            config,
            source,
            registry,
            NullRenderer(),
            clock=ManualClock(),
            install_signal_handlers=False,
        )
        stats = runner.run()
        assert stats.processed == 120
        assert stats.outputs == 120
        assert "processed 120" in runner.footer()
        assert any("accepted 120" in line for line in runner.summary_lines())

    def test_runner_honours_max_snapshots(self, tmp_path: Path) -> None:
        recording = tmp_path / "session.jsonl"
        stream = SnapshotStream()
        with SnapshotRecorder(recording) as recorder:
            for raw in stream.many(50):
                recorder.write(raw)

        config = fast_config(
            runtime={"render_fps": 0.0, "max_snapshots": 10, "log_file": None}
        )
        registry = EngineRegistry(config, clock=ManualClock())
        runner = EngineRunner(
            config,
            ReplaySource(recording),
            registry,
            NullRenderer(),
            clock=ManualClock(),
            install_signal_handlers=False,
        )
        stats = runner.run()
        assert stats.processed == 10

    def test_latency_tracker_percentiles(self) -> None:
        tracker = LatencyTracker(128)
        for value in range(1, 101):
            tracker.record(float(value))
        p50, p99, worst = tracker.percentiles()
        assert p50 == pytest.approx(51.0)
        assert p99 == pytest.approx(100.0)
        assert worst == pytest.approx(100.0)

    def test_empty_latency_tracker_is_zero(self) -> None:
        assert LatencyTracker(16).percentiles() == (0.0, 0.0, 0.0)

    def test_plain_renderer_writes_a_frame(self, tmp_path: Path) -> None:
        import io

        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        outputs = drain_outputs(registry, rising_bid_pressure(120, stream=stream))
        buffer = io.StringIO()
        renderer = PlainRenderer(stream=buffer, colour=False)
        renderer.start()
        renderer.render(outputs[-1:], "footer text")
        renderer.stop()
        text = buffer.getvalue()
        assert "SBIN" in text
        assert "composite" in text
        assert "threshold" in text
        assert "footer text" in text

    def test_view_model_formats_every_field(self) -> None:
        registry = EngineRegistry(fast_config(), clock=ManualClock())
        stream = SnapshotStream()
        outputs = drain_outputs(registry, rising_bid_pressure(180, stream=stream))
        entered = [output for output in outputs if output.state is TradeState.LONG]
        view = build_view(entered[0] if entered else outputs[-1])
        assert view.symbol == "SBIN"
        assert ":" in view.time
        assert view.composite
        assert view.threshold
        assert view.book
        assert view.latency.endswith("us")

    def test_score_bar_is_centred_and_bounded(self) -> None:
        neutral = score_bar(0.0, 0.3)
        positive = score_bar(0.9, 0.3)
        negative = score_bar(-0.9, 0.3)
        assert len(neutral) == len(positive) == len(negative)
        assert "|" in neutral
        assert positive.index("█") > len(positive) // 2
        assert negative.index("█") < len(negative) // 2

    def test_renderer_factory_falls_back_without_rich(self) -> None:
        renderer = create_renderer("auto")
        assert renderer is not None
        assert create_renderer("none").__class__ is NullRenderer
        with pytest.raises(ValueError):
            create_renderer("nonsense")
