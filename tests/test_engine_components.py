"""Tests for the confidence model, composite scorer, regime detector and threshold."""

from __future__ import annotations

import pytest
from snapshot_quant_v4.config import (
    CompositeConfig,
    ConfidenceConfig,
    ConfigError,
    FeatureSensitivity,
    QualityConfig,
    RegimeConfig,
    ThresholdConfig,
)
from snapshot_quant_v4.engine.composite import CompositeScorer
from snapshot_quant_v4.engine.confidence import ConfidenceModel
from snapshot_quant_v4.engine.quality import MarketQualityFilter
from snapshot_quant_v4.engine.regime import RegimeDetector
from snapshot_quant_v4.engine.stats import SharedStatistics
from snapshot_quant_v4.engine.threshold import AdaptiveThreshold
from snapshot_quant_v4.engine.validator import SnapshotValidator
from snapshot_quant_v4.utils.constants import (
    F_MICROPRICE,
    F_MOMENTUM,
    F_SPREAD,
    F_WEIGHTED_OBI,
)
from snapshot_quant_v4.utils.types import BlockReason, Regime

from conftest import (
    BASE_PAISE,
    TICK_PAISE,
    TICK_SIZE,
    SnapshotStream,
    build_config,
    feature_value,
    quality_report,
    validate_one,
)


def build_stats(app_config=None) -> SharedStatistics:
    """Construct shared statistics for the reference instrument."""
    config = app_config if app_config is not None else build_config()
    return SharedStatistics(
        tick_size=TICK_SIZE,
        confidence=config.confidence,
        regime=config.regime,
        threshold=config.threshold,
        quality=config.quality,
    )


class StatsFeeder:
    """Feeds snapshots through one validator into one statistics instance.

    The validator must be shared across the whole sequence: a fresh validator
    treats every snapshot as the first of a sequence, which suppresses interval
    and volume deltas and would make these tests assert on statistics that never
    received any input.
    """

    def __init__(self, app_config=None) -> None:
        config = app_config if app_config is not None else build_config()
        self.stats = build_stats(config)
        self.validator = SnapshotValidator("SBIN", TICK_SIZE, config.validation)

    def push(self, raw):
        """Validate and push one snapshot, returning the enriched snapshot."""
        result = self.validator.validate(raw)
        assert result.accepted, result.reason
        assert result.snapshot is not None
        self.stats.update_pre(result.snapshot)
        return result.snapshot

    def push_many(self, snapshots) -> None:
        """Push a sequence of snapshots."""
        for raw in snapshots:
            self.push(raw)


def feed(stats: SharedStatistics, snapshots) -> None:
    """Push snapshots into ``stats`` using a single shared validator.

    Kept for tests that only need the statistics populated and do not care about
    the validated snapshots themselves.
    """
    validator = SnapshotValidator("SBIN", TICK_SIZE, build_config().validation)
    for raw in snapshots:
        result = validator.validate(raw)
        assert result.accepted, result.reason
        assert result.snapshot is not None
        stats.update_pre(result.snapshot)


class TestSharedStatistics:
    def test_efficiency_ratio_is_high_for_a_straight_move(self) -> None:
        feeder = StatsFeeder()
        stream = SnapshotStream()
        for step in range(40):
            feeder.push(
                stream.next(
                    bid_paise=BASE_PAISE + step * TICK_PAISE - TICK_PAISE,
                    ask_paise=BASE_PAISE + step * TICK_PAISE + TICK_PAISE,
                )
            )
        assert feeder.stats.efficiency_ratio(40) > 0.9

    def test_efficiency_ratio_is_low_for_oscillation(self) -> None:
        feeder = StatsFeeder()
        stream = SnapshotStream()
        for step in range(40):
            offset = TICK_PAISE if step % 2 else 0
            feeder.push(
                stream.next(
                    bid_paise=BASE_PAISE + offset - TICK_PAISE,
                    ask_paise=BASE_PAISE + offset + TICK_PAISE,
                )
            )
        assert feeder.stats.efficiency_ratio(40) < 0.2

    def test_touch_stability_tolerates_one_tick_drift(self) -> None:
        # A trending book moves the touch every snapshot. Treating that as
        # instability would collapse confidence in exactly the regime the engine
        # is meant to trade, so a one-tick drift counts as orderly.
        feeder = StatsFeeder()
        stream = SnapshotStream()
        for step in range(20):
            feeder.push(
                stream.next(
                    bid_paise=BASE_PAISE + step * TICK_PAISE - TICK_PAISE,
                    ask_paise=BASE_PAISE + step * TICK_PAISE + TICK_PAISE,
                )
            )
        assert feeder.stats.touch_stability > 0.9

    def test_touch_stability_falls_for_multi_tick_jumps(self) -> None:
        feeder = StatsFeeder()
        stream = SnapshotStream()
        for step in range(20):
            feeder.push(
                stream.next(
                    bid_paise=BASE_PAISE + step * TICK_PAISE * 5 - TICK_PAISE,
                    ask_paise=BASE_PAISE + step * TICK_PAISE * 5 + TICK_PAISE,
                )
            )
        assert feeder.stats.touch_stability < 0.2

    def test_relative_depth_falls_when_the_book_thins(self) -> None:
        feeder = StatsFeeder()
        stream = SnapshotStream()
        feeder.push_many(
            stream.many(30, bid_quantities=(1_000, 900), ask_quantities=(1_000, 900))
        )
        assert feeder.stats.relative_depth == pytest.approx(1.0, rel=0.15)
        feeder.push(stream.next(bid_quantities=(50, 40), ask_quantities=(50, 40)))
        assert feeder.stats.relative_depth < 0.2

    def test_reset_clears_everything(self) -> None:
        feeder = StatsFeeder()
        stream = SnapshotStream()
        feeder.push_many(stream.many(20))
        feeder.stats.reset()
        assert feeder.stats.snapshots == 0
        assert feeder.stats.mid_volatility_ticks == 0.0
        assert not feeder.stats.warm


class TestMarketQualityFilter:
    def test_open_book_is_tradable_after_warmup(self) -> None:
        config = build_config(quality={"warmup_snapshots": 5})
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(stats, stream.many(10))
        snapshot = validate_one(stream.next(), config)
        report = MarketQualityFilter(config.quality).evaluate(
            snapshot, stats, ms_since_gap=None
        )
        assert report.tradable
        assert report.reasons == ()
        assert 0.0 < report.book_quality <= 1.0

    def test_warmup_blocks(self) -> None:
        config = build_config(quality={"warmup_snapshots": 50})
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(stats, stream.many(3))
        snapshot = validate_one(stream.next(), config)
        report = MarketQualityFilter(config.quality).evaluate(
            snapshot, stats, ms_since_gap=None
        )
        assert not report.tradable
        assert BlockReason.WARMUP in report.reasons

    def test_wide_spread_blocks(self) -> None:
        config = build_config(quality={"warmup_snapshots": 2, "max_signal_spread_ticks": 3.0})
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(stats, stream.many(10))
        wide = validate_one(
            stream.next(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + 8 * TICK_PAISE),
            config,
        )
        report = MarketQualityFilter(config.quality).evaluate(
            wide, stats, ms_since_gap=None
        )
        assert BlockReason.SPREAD_TOO_WIDE in report.reasons

    def test_thin_book_blocks_relative_to_its_own_history(self) -> None:
        config = build_config(
            quality={"warmup_snapshots": 2, "min_relative_depth": 0.5}
        )
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(
            stats,
            stream.many(40, bid_quantities=(2_000, 1_800), ask_quantities=(2_000, 1_800)),
        )
        thin_raw = stream.next(bid_quantities=(20, 10), ask_quantities=(20, 10))
        feed(stats, [thin_raw])
        thin = validate_one(thin_raw, config)
        report = MarketQualityFilter(config.quality).evaluate(
            thin, stats, ms_since_gap=None
        )
        assert BlockReason.LIQUIDITY_BELOW_THRESHOLD in report.reasons

    def test_recent_gap_blocks(self) -> None:
        config = build_config(quality={"warmup_snapshots": 2, "gap_block_ms": 5_000.0})
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(stats, stream.many(10))
        snapshot = validate_one(stream.next(), config)
        report = MarketQualityFilter(config.quality).evaluate(
            snapshot, stats, ms_since_gap=100.0
        )
        assert BlockReason.FEED_GAP in report.reasons

    def test_insufficient_levels_block(self) -> None:
        config = build_config(quality={"warmup_snapshots": 2, "min_depth_levels": 3})
        stats = build_stats(config)
        stream = SnapshotStream()
        feed(stats, stream.many(5, bid_quantities=(500, 400), ask_quantities=(500, 400)))
        snapshot = validate_one(
            stream.next(bid_quantities=(500, 400), ask_quantities=(500, 400)), config
        )
        report = MarketQualityFilter(config.quality).evaluate(
            snapshot, stats, ms_since_gap=None
        )
        assert BlockReason.INSUFFICIENT_DEPTH_LEVELS in report.reasons


class TestConfidenceModel:
    def _factors(self, model: ConfidenceModel, spread_ticks: float, stats, quality):
        return model.factors(
            spread_ticks=spread_ticks, stats=stats, quality=quality, features={}
        )

    def test_confidence_falls_as_the_spread_widens(self) -> None:
        config = ConfidenceConfig()
        model = ConfidenceModel(config)
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        quality = quality_report()
        feature = feature_value(F_MICROPRICE, 0.5)

        tight = model.confidence_for(
            feature, self._factors(model, 1.0, stats, quality), regime=Regime.RANGE
        )
        wide = model.confidence_for(
            feature, self._factors(model, 5.5, stats, quality), regime=Regime.RANGE
        )
        assert 0.0 < wide < tight <= 1.0

    def test_confidence_falls_as_liquidity_falls(self) -> None:
        model = ConfidenceModel(ConfidenceConfig())
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        feature = feature_value(F_WEIGHTED_OBI, 0.5)
        deep = model.confidence_for(
            feature,
            self._factors(model, 1.0, stats, quality_report(liquidity_score=1.0)),
            regime=Regime.RANGE,
        )
        thin = model.confidence_for(
            feature,
            self._factors(model, 1.0, stats, quality_report(liquidity_score=0.15)),
            regime=Regime.RANGE,
        )
        assert thin < deep

    def test_noise_regime_multiplier_reduces_confidence(self) -> None:
        model = ConfidenceModel(ConfidenceConfig())
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        factors = self._factors(model, 1.0, stats, quality_report())
        feature = feature_value(F_MOMENTUM, 0.5)
        calm = model.confidence_for(feature, factors, regime=Regime.TREND)
        noisy = model.confidence_for(feature, factors, regime=Regime.NOISE)
        assert noisy < calm

    def test_sensitivity_exponents_differentiate_features(self) -> None:
        # A feature with a high spread exponent must degrade faster than one with
        # a low exponent as the spread widens.
        config = ConfidenceConfig(
            sensitivities={
                "sensitive": FeatureSensitivity(
                    spread=3.0,
                    liquidity=0.0,
                    volatility=0.0,
                    book_quality=0.0,
                    stability=0.0,
                    warmup=0.0,
                ),
                "insensitive": FeatureSensitivity(
                    spread=0.2,
                    liquidity=0.0,
                    volatility=0.0,
                    book_quality=0.0,
                    stability=0.0,
                    warmup=0.0,
                ),
            }
        )
        model = ConfidenceModel(config)
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        factors = self._factors(model, 4.0, stats, quality_report())
        sensitive = model.confidence_for(
            feature_value("sensitive", 0.5), factors, regime=Regime.RANGE
        )
        insensitive = model.confidence_for(
            feature_value("insensitive", 0.5), factors, regime=Regime.RANGE
        )
        assert sensitive < insensitive

    def test_invalid_and_quality_features_get_zero(self) -> None:
        model = ConfidenceModel(ConfidenceConfig())
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        factors = self._factors(model, 1.0, stats, quality_report())
        assert (
            model.confidence_for(
                feature_value(F_MICROPRICE, 0.5, valid=False), factors, regime=Regime.RANGE
            )
            == 0.0
        )
        assert (
            model.confidence_for(
                feature_value(F_SPREAD, 0.5, directional=False),
                factors,
                regime=Regime.RANGE,
            )
            == 0.0
        )

    def test_apply_fills_confidence_without_mutating_values(self) -> None:
        model = ConfidenceModel(ConfidenceConfig())
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        features = {
            F_MICROPRICE: feature_value(F_MICROPRICE, 0.4, confidence=0.0),
            F_SPREAD: feature_value(F_SPREAD, 1.0, confidence=0.0, directional=False),
        }
        model.apply(
            features,
            spread_ticks=1.0,
            stats=stats,
            quality=quality_report(),
            regime=Regime.RANGE,
        )
        assert features[F_MICROPRICE].confidence > 0.0
        assert features[F_MICROPRICE].value == pytest.approx(0.4)
        assert features[F_SPREAD].confidence == 0.0

    def test_logarithm_form_matches_the_power_form(self) -> None:
        # The optimisation must be exactly equivalent to the documented product of
        # powers.
        model = ConfidenceModel(ConfidenceConfig())
        stats = build_stats()
        feed(stats, SnapshotStream().many(80))
        factors = model.factors(
            spread_ticks=2.0, stats=stats, quality=quality_report(), features={}
        )
        sensitivity = ConfidenceConfig().sensitivity_for(F_MICROPRICE)
        expected = (
            factors.spread**sensitivity.spread
            * factors.liquidity**sensitivity.liquidity
            * factors.volatility**sensitivity.volatility
            * factors.book_quality**sensitivity.book_quality
            * factors.stability**sensitivity.stability
            * factors.warmup**sensitivity.warmup
        )
        actual = model.confidence_for(
            feature_value(F_MICROPRICE, 0.5), factors, regime=Regime.TREND
        )
        assert actual == pytest.approx(expected, rel=1e-12)


class TestCompositeScorer:
    def test_score_is_the_confidence_weighted_mean(self) -> None:
        config = CompositeConfig(
            weights={"RANGE": {"a": 1.0, "b": 1.0, "c": 1.0}},
            collinear_groups={},
            min_features=1,
        )
        scorer = CompositeScorer(config)
        features = {
            "a": feature_value("a", 0.9, confidence=1.0),
            "b": feature_value("b", -0.3, confidence=1.0),
            "c": feature_value("c", 0.6, confidence=1.0),
        }
        result = scorer.score(features, regime=Regime.RANGE, dt_ms=0.0)
        assert result.valid
        assert result.score == pytest.approx((0.9 - 0.3 + 0.6) / 3.0)
        assert -1.0 <= result.score <= 1.0

    def test_uniform_weight_scaling_does_not_change_the_score(self) -> None:
        # The property that makes "lower every weight in a noisy regime" a no-op,
        # and the reason reduced trust is expressed through confidence instead.
        small = CompositeConfig(
            weights={"NOISE": {"a": 0.1, "b": 0.1}},
            collinear_groups={},
            min_features=1,
            min_confidence=0.0,
        )
        large = CompositeConfig(
            weights={"NOISE": {"a": 10.0, "b": 10.0}},
            collinear_groups={},
            min_features=1,
            min_confidence=0.0,
        )
        features = {
            "a": feature_value("a", 0.8, confidence=0.9),
            "b": feature_value("b", -0.2, confidence=0.4),
        }
        first = CompositeScorer(small).score(features, regime=Regime.NOISE, dt_ms=0.0)
        second = CompositeScorer(large).score(features, regime=Regime.NOISE, dt_ms=0.0)
        assert first.score == pytest.approx(second.score)

    def test_confidence_reweights_rather_than_deflating(self) -> None:
        config = CompositeConfig(
            weights={"RANGE": {"a": 1.0, "b": 1.0}}, collinear_groups={}, min_features=1
        )
        scorer = CompositeScorer(config)
        features = {
            "a": feature_value("a", 1.0, confidence=0.9),
            "b": feature_value("b", 1.0, confidence=0.1),
        }
        result = scorer.score(features, regime=Regime.RANGE, dt_ms=0.0)
        # Both features agree at full magnitude, so the normalised score is 1.0
        # regardless of how low one confidence is.
        assert result.score == pytest.approx(1.0)
        assert result.confidence == pytest.approx(0.5)

    def test_invalid_features_are_excluded_not_zeroed(self) -> None:
        config = CompositeConfig(
            weights={"RANGE": {"a": 1.0, "b": 1.0}}, collinear_groups={}, min_features=1
        )
        scorer = CompositeScorer(config)
        features = {
            "a": feature_value("a", 0.8, confidence=1.0),
            "b": feature_value("b", 0.0, confidence=0.0, valid=False),
        }
        result = scorer.score(features, regime=Regime.RANGE, dt_ms=0.0)
        assert result.used_features == 1
        assert result.score == pytest.approx(0.8)

    def test_collinear_group_weight_is_capped(self) -> None:
        config = CompositeConfig(
            weights={
                "RANGE": {F_MICROPRICE: 0.45, F_WEIGHTED_OBI: 0.45, "other": 0.10}
            },
            collinear_groups={f"{F_MICROPRICE}|{F_WEIGHTED_OBI}": 0.30},
            min_features=1,
        )
        scorer = CompositeScorer(config)
        features = {
            F_MICROPRICE: feature_value(F_MICROPRICE, 1.0, confidence=1.0),
            F_WEIGHTED_OBI: feature_value(F_WEIGHTED_OBI, 1.0, confidence=1.0),
            "other": feature_value("other", -1.0, confidence=1.0),
        }
        result = scorer.score(features, regime=Regime.RANGE, dt_ms=0.0)
        capped = {c.name: c.weight for c in result.contributions}
        group_weight = capped[F_MICROPRICE] + capped[F_WEIGHTED_OBI]
        assert group_weight == pytest.approx(0.30, rel=1e-9)
        # Without the cap the collinear pair would dominate; with it the score is
        # pulled markedly towards the independent feature.
        assert result.score < 0.6

    def test_below_min_features_is_invalid(self) -> None:
        config = CompositeConfig(
            weights={"RANGE": {"a": 1.0}}, collinear_groups={}, min_features=3
        )
        scorer = CompositeScorer(config)
        result = scorer.score(
            {"a": feature_value("a", 0.9)}, regime=Regime.RANGE, dt_ms=0.0
        )
        assert not result.valid

    def test_smoothing_lags_a_step_change(self) -> None:
        config = CompositeConfig(
            weights={"RANGE": {"a": 1.0}},
            collinear_groups={},
            min_features=1,
            smoothing_half_life_ms=1_000.0,
        )
        scorer = CompositeScorer(config)
        scorer.score({"a": feature_value("a", 0.0)}, regime=Regime.RANGE, dt_ms=0.0)
        result = scorer.score(
            {"a": feature_value("a", 1.0)}, regime=Regime.RANGE, dt_ms=1_000.0
        )
        assert result.score == pytest.approx(1.0)
        assert result.smoothed == pytest.approx(0.5, rel=1e-6)

    def test_unknown_regime_falls_back(self) -> None:
        config = CompositeConfig(
            weights={"UNKNOWN": {"a": 1.0}}, collinear_groups={}, min_features=1
        )
        scorer = CompositeScorer(config)
        result = scorer.score(
            {"a": feature_value("a", 0.5)}, regime=Regime.TREND, dt_ms=0.0
        )
        assert result.valid


class TestRegimeDetector:
    def _run(self, config: RegimeConfig, snapshots, momentum_sign: int = 0):
        stats = build_stats()
        detector = RegimeDetector(config)
        validator = SnapshotValidator("SBIN", TICK_SIZE, build_config().validation)
        result = None
        for raw in snapshots:
            outcome = validator.validate(raw)
            assert outcome.accepted, outcome.reason
            assert outcome.snapshot is not None
            stats.update_pre(outcome.snapshot)
            result = detector.update(stats, momentum_sign=momentum_sign)
        assert result is not None
        return result

    def test_starts_unknown(self) -> None:
        config = RegimeConfig(min_samples=30)
        result = self._run(config, SnapshotStream().many(5))
        assert result.regime is Regime.UNKNOWN

    def test_straight_move_is_a_trend(self) -> None:
        config = RegimeConfig(
            min_samples=20, min_dwell_snapshots=1, confirm_snapshots=1, window=60
        )
        stream = SnapshotStream()
        snapshots = [
            stream.next(
                bid_paise=BASE_PAISE + step * TICK_PAISE - TICK_PAISE,
                ask_paise=BASE_PAISE + step * TICK_PAISE + TICK_PAISE,
            )
            for step in range(60)
        ]
        result = self._run(config, snapshots, momentum_sign=1)
        assert result.regime is Regime.TREND
        assert result.efficiency_ratio > 0.9

    def test_counter_momentum_in_a_trend_is_a_pullback(self) -> None:
        config = RegimeConfig(
            min_samples=20, min_dwell_snapshots=1, confirm_snapshots=1, window=60
        )
        stream = SnapshotStream()
        snapshots = [
            stream.next(
                bid_paise=BASE_PAISE + step * TICK_PAISE - TICK_PAISE,
                ask_paise=BASE_PAISE + step * TICK_PAISE + TICK_PAISE,
            )
            for step in range(60)
        ]
        result = self._run(config, snapshots, momentum_sign=-1)
        assert result.regime is Regime.PULLBACK

    def test_flat_book_is_a_range(self) -> None:
        config = RegimeConfig(
            min_samples=20, min_dwell_snapshots=1, confirm_snapshots=1, window=60
        )
        result = self._run(config, SnapshotStream().many(60))
        assert result.regime is Regime.RANGE

    def test_violent_chop_is_noise(self) -> None:
        config = RegimeConfig(
            min_samples=20,
            min_dwell_snapshots=1,
            confirm_snapshots=1,
            window=60,
            vol_high_ticks=1.5,
        )
        stream = SnapshotStream()
        snapshots = []
        for step in range(60):
            offset = (step % 2) * 12 * TICK_PAISE
            snapshots.append(
                stream.next(
                    bid_paise=BASE_PAISE + offset - TICK_PAISE,
                    ask_paise=BASE_PAISE + offset + TICK_PAISE,
                )
            )
        result = self._run(config, snapshots)
        assert result.regime is Regime.NOISE

    def test_hysteresis_prevents_an_immediate_flip(self) -> None:
        config = RegimeConfig(
            min_samples=10,
            min_dwell_snapshots=25,
            confirm_snapshots=5,
            window=40,
        )
        stats = build_stats()
        detector = RegimeDetector(config)
        validator = SnapshotValidator("SBIN", TICK_SIZE, build_config().validation)
        stream = SnapshotStream()

        def push(raw, momentum_sign=0):
            outcome = validator.validate(raw)
            assert outcome.snapshot is not None
            stats.update_pre(outcome.snapshot)
            return detector.update(stats, momentum_sign=momentum_sign)

        for _ in range(30):
            push(stream.next())
        assert detector.regime is Regime.RANGE
        # One violent snapshot must not change the label.
        result = push(
            stream.next(
                bid_paise=BASE_PAISE + 40 * TICK_PAISE - TICK_PAISE,
                ask_paise=BASE_PAISE + 40 * TICK_PAISE + TICK_PAISE,
            )
        )
        assert result.regime is Regime.RANGE

    def test_reset_returns_to_unknown(self) -> None:
        config = RegimeConfig(min_samples=10, min_dwell_snapshots=1, confirm_snapshots=1)
        detector = RegimeDetector(config)
        stats = build_stats()
        feed(stats, SnapshotStream().many(30))
        detector.update(stats)
        detector.reset()
        assert detector.regime is Regime.UNKNOWN


class TestAdaptiveThreshold:
    def test_pinned_to_maximum_during_warmup(self) -> None:
        config = ThresholdConfig()
        threshold = AdaptiveThreshold(config)
        result = threshold.compute(build_stats())
        assert result.entry == pytest.approx(config.max_entry)
        assert "warming up" in result.detail

    def test_quiet_market_is_floored(self) -> None:
        config = ThresholdConfig(base=0.02, min_entry=0.2, max_entry=0.9)
        stats = build_stats()
        feed(stats, SnapshotStream().many(120))
        result = AdaptiveThreshold(config).compute(stats)
        assert result.entry == pytest.approx(0.2)
        assert result.floor_applied

    def test_volatile_market_raises_the_threshold(self) -> None:
        config = ThresholdConfig()
        calm_stats = build_stats()
        feed(calm_stats, SnapshotStream().many(150))
        calm = AdaptiveThreshold(config).compute(calm_stats)

        wild_stats = build_stats()
        stream = SnapshotStream()
        wild = []
        for step in range(150):
            offset = (step % 2) * 10 * TICK_PAISE
            wild.append(
                stream.next(
                    bid_paise=BASE_PAISE + offset - TICK_PAISE,
                    ask_paise=BASE_PAISE + offset + TICK_PAISE * (1 + step % 3),
                )
            )
        feed(wild_stats, wild)
        volatile = AdaptiveThreshold(config).compute(wild_stats)
        assert volatile.entry > calm.entry

    def test_clamped_to_maximum(self) -> None:
        config = ThresholdConfig(base=0.9, max_entry=0.5, min_entry=0.1)
        stats = build_stats()
        feed(stats, SnapshotStream().many(120))
        result = AdaptiveThreshold(config).compute(stats)
        assert result.entry == pytest.approx(0.5)

    def test_watch_and_exit_derive_from_entry(self) -> None:
        config = ThresholdConfig(watch_ratio=0.5, exit_ratio=0.25)
        stats = build_stats()
        feed(stats, SnapshotStream().many(120))
        result = AdaptiveThreshold(config).compute(stats)
        assert result.watch == pytest.approx(result.entry * 0.5)
        assert result.exit == pytest.approx(result.entry * 0.25)


class TestQualityConfigValidation:
    def test_rejects_out_of_range_relative_depth(self) -> None:
        with pytest.raises(ConfigError):
            QualityConfig(min_relative_depth=2.0)
