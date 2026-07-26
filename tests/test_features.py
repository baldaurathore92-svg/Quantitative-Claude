"""Feature tests.

These are the tests that matter most, because a feature can be wrong while the
whole system still runs and prints plausible numbers. Each test states a market
condition and asserts the sign, the bound or the invariance that the feature's
documented definition promises.
"""

from __future__ import annotations

import pytest
from snapshot_quant_v4.config import (
    DepthSlopeConfig,
    LtpConfirmationConfig,
    MicropriceConfig,
    MomentumConfig,
    QueuePersistenceConfig,
    RefillConfig,
    SpreadCompressionConfig,
    SpreadConfig,
    WeightedObiConfig,
)
from snapshot_quant_v4.features.acceleration import AccelerationFeature
from snapshot_quant_v4.features.depth_slope import DepthSlopeFeature
from snapshot_quant_v4.features.ltp_confirmation import LtpConfirmationFeature
from snapshot_quant_v4.features.microprice import MicropriceFeature
from snapshot_quant_v4.features.momentum import MomentumFeature
from snapshot_quant_v4.features.queue_persistence import QueuePersistenceFeature
from snapshot_quant_v4.features.refill import RefillFeature
from snapshot_quant_v4.features.spread import SpreadFeature
from snapshot_quant_v4.features.spread_compression import SpreadCompressionFeature
from snapshot_quant_v4.features.weighted_obi import WeightedObiFeature
from snapshot_quant_v4.utils.constants import F_ACCELERATION, F_MOMENTUM
from snapshot_quant_v4.utils.types import FeatureKind

from conftest import (
    BASE_PAISE,
    TICK_PAISE,
    FeatureHarness,
    SnapshotStream,
    build_config,
    ladder,
    level,
    make_snapshot,
)


class TestMicroprice:
    def test_heavy_bid_tilts_upwards(self) -> None:
        harness = FeatureHarness([MicropriceFeature(MicropriceConfig())])
        value = harness.single(
            make_snapshot(bid_quantities=(5_000, 400), ask_quantities=(200, 300))
        )
        assert value.valid
        assert value.value > 0.5
        assert value.raw > 0.0

    def test_heavy_ask_tilts_downwards(self) -> None:
        harness = FeatureHarness([MicropriceFeature(MicropriceConfig())])
        value = harness.single(
            make_snapshot(bid_quantities=(200, 300), ask_quantities=(5_000, 400))
        )
        assert value.value < -0.5

    def test_normalised_tilt_equals_the_level_one_imbalance(self) -> None:
        # The documented algebraic identity:
        #   (microprice - mid) / (spread / 2) == (bid_qty - ask_qty) / (bid + ask)
        # The first snapshot seeds the smoother, so the reported value is the
        # unsmoothed tilt and the identity is directly observable.
        harness = FeatureHarness([MicropriceFeature(MicropriceConfig())])
        bid_qty, ask_qty = 1_500, 500
        value = harness.single(
            make_snapshot(bid_quantities=(bid_qty,), ask_quantities=(ask_qty,))
        )
        expected = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        assert value.value == pytest.approx(expected, rel=1e-9)

    def test_balanced_book_is_neutral(self) -> None:
        harness = FeatureHarness([MicropriceFeature(MicropriceConfig())])
        value = harness.single(
            make_snapshot(bid_quantities=(800,), ask_quantities=(800,))
        )
        assert value.value == pytest.approx(0.0, abs=1e-12)

    def test_raw_is_reported_in_ticks(self) -> None:
        harness = FeatureHarness([MicropriceFeature(MicropriceConfig())])
        # Spread is two ticks, so a fully one-sided book tilts one tick.
        value = harness.single(
            make_snapshot(bid_quantities=(10_000,), ask_quantities=(1,))
        )
        assert value.raw == pytest.approx(1.0, rel=0.01)


class TestWeightedObi:
    def test_bid_heavy_book_is_positive(self) -> None:
        harness = FeatureHarness([WeightedObiFeature(WeightedObiConfig())])
        value = harness.single(
            make_snapshot(bid_quantities=(2_000, 1_800), ask_quantities=(300, 250))
        )
        assert value.value > 0.5
        assert value.kind is FeatureKind.DIRECTIONAL

    def test_is_scale_free_across_price_levels(self) -> None:
        # The same *relative* book at 150 rupees and at 3000 rupees must produce
        # the same imbalance. A rupee-denominated decay constant fails this.
        harness_cheap = FeatureHarness([WeightedObiFeature(WeightedObiConfig())])
        harness_rich = FeatureHarness([WeightedObiFeature(WeightedObiConfig())])
        cheap = harness_cheap.single(
            make_snapshot(
                bid_paise=15_000 - TICK_PAISE,
                ask_paise=15_000 + TICK_PAISE,
                bid_quantities=(1_200, 900),
                ask_quantities=(400, 300),
            )
        )
        rich = harness_rich.single(
            make_snapshot(
                bid_paise=300_000 - TICK_PAISE,
                ask_paise=300_000 + TICK_PAISE,
                bid_quantities=(1_200, 900),
                ask_quantities=(400, 300),
            )
        )
        assert cheap.value == pytest.approx(rich.value, rel=1e-9)

    def test_only_primary_levels_move_the_score(self) -> None:
        config = WeightedObiConfig(primary_levels=2)
        harness = FeatureHarness([WeightedObiFeature(config)])
        balanced_deep = harness.single(
            make_snapshot(
                bid_quantities=(1_000, 800, 100, 100, 100),
                ask_quantities=(1_000, 800, 100, 100, 100),
            )
        )
        harness_two = FeatureHarness([WeightedObiFeature(config)])
        lopsided_deep = harness_two.single(
            make_snapshot(
                bid_quantities=(1_000, 800, 9_000, 9_000, 9_000),
                ask_quantities=(1_000, 800, 100, 100, 100),
            )
        )
        # Deep levels are secondary: the score is unchanged, only confidence moves.
        assert lopsided_deep.value == pytest.approx(balanced_deep.value)
        assert lopsided_deep.local_confidence != balanced_deep.local_confidence

    def test_deep_disagreement_reduces_local_confidence(self) -> None:
        harness = FeatureHarness([WeightedObiFeature(WeightedObiConfig())])
        agreeing = harness.single(
            make_snapshot(
                bid_quantities=(2_000, 1_500, 1_400, 1_300, 1_200),
                ask_quantities=(300, 250, 200, 150, 100),
            )
        )
        harness_two = FeatureHarness([WeightedObiFeature(WeightedObiConfig())])
        disagreeing = harness_two.single(
            make_snapshot(
                bid_quantities=(2_000, 1_500, 50, 50, 50),
                ask_quantities=(300, 250, 5_000, 5_000, 5_000),
            )
        )
        assert disagreeing.local_confidence < agreeing.local_confidence
        assert 0.0 <= disagreeing.local_confidence <= 1.0

    def test_lopsided_totals_are_ignored_as_far_market_junk(self) -> None:
        config = WeightedObiConfig(aggregate_max_ratio=3.0)
        harness = FeatureHarness([WeightedObiFeature(config)])
        value = harness.single(
            make_snapshot(total_buy_quantity=1_000_000.0, total_sell_quantity=1_000.0)
        )
        assert "agg=n/a" in value.detail

    def test_empty_primary_levels_is_invalid(self) -> None:
        harness = FeatureHarness([WeightedObiFeature(WeightedObiConfig(primary_levels=1))])
        value = harness.single(
            make_snapshot(
                bids=(level(BASE_PAISE - TICK_PAISE, 500),),
                asks=(level(BASE_PAISE + TICK_PAISE, 500),),
            )
        )
        assert value.valid


class TestDepthSlope:
    def test_thin_ask_ladder_is_bullish(self) -> None:
        harness = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())])
        value = harness.single(
            make_snapshot(
                bid_quantities=(1_000, 1_000, 1_000, 1_000, 1_000),
                ask_quantities=(1_000, 100, 50, 25, 10),
            )
        )
        assert value.valid
        assert value.value > 0.0

    def test_thin_bid_ladder_is_bearish(self) -> None:
        harness = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())])
        value = harness.single(
            make_snapshot(
                bid_quantities=(1_000, 100, 50, 25, 10),
                ask_quantities=(1_000, 1_000, 1_000, 1_000, 1_000),
            )
        )
        assert value.value < 0.0

    def test_symmetric_ladder_is_neutral(self) -> None:
        harness = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())])
        value = harness.single(make_snapshot())
        assert value.value == pytest.approx(0.0, abs=1e-9)

    def test_missing_second_level_is_invalid(self) -> None:
        harness = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())])
        value = harness.single(
            make_snapshot(
                bids=(level(BASE_PAISE - TICK_PAISE, 500),),
                asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (500, 400)),
            )
        )
        assert not value.valid
        assert "level two" in value.detail

    def test_decay_is_normalised_by_tick_distance(self) -> None:
        # Identical quantity ratios but different level spacing must not produce
        # the same decay: the far ladder decays more slowly per tick.
        near = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())]).single(
            make_snapshot(
                bids=ladder(BASE_PAISE - TICK_PAISE, -TICK_PAISE, (1_000, 500)),
                asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (1_000, 1_000)),
            )
        )
        far = FeatureHarness([DepthSlopeFeature(DepthSlopeConfig())]).single(
            make_snapshot(
                bids=ladder(BASE_PAISE - TICK_PAISE, -4 * TICK_PAISE, (1_000, 500)),
                asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (1_000, 1_000)),
            )
        )
        assert abs(far.raw) < abs(near.raw)


class TestSpreadFeatures:
    def test_tightness_decreases_with_spread(self) -> None:
        config = SpreadConfig(tight_ticks=1.0, wide_ticks=6.0)
        tight = FeatureHarness([SpreadFeature(config)]).single(
            make_snapshot(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + TICK_PAISE)
        )
        wide = FeatureHarness([SpreadFeature(config)]).single(
            make_snapshot(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + 6 * TICK_PAISE)
        )
        assert tight.value == pytest.approx(1.0)
        assert wide.value == pytest.approx(0.0)
        assert tight.kind is FeatureKind.QUALITY

    def test_spread_raw_is_in_ticks(self) -> None:
        value = FeatureHarness([SpreadFeature(SpreadConfig())]).single(
            make_snapshot(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + 3 * TICK_PAISE)
        )
        assert value.raw == pytest.approx(3.0)

    def test_compression_rises_when_the_spread_narrows(self) -> None:
        config = SpreadCompressionConfig(
            fast_half_life_ms=200.0, slow_half_life_ms=4_000.0, scale_ticks=0.5
        )
        harness = FeatureHarness([SpreadCompressionFeature(config)])
        stream = SnapshotStream()
        for _ in range(20):
            harness.push(
                stream.next(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + 4 * TICK_PAISE)
            )
        latest = harness.push(
            stream.next(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + TICK_PAISE)
        )
        value = latest["spread_compression"]
        assert value.valid
        assert value.value > 0.5
        assert value.raw > 0.0

    def test_compression_falls_when_the_spread_widens(self) -> None:
        config = SpreadCompressionConfig(
            fast_half_life_ms=200.0, slow_half_life_ms=4_000.0, scale_ticks=0.5
        )
        harness = FeatureHarness([SpreadCompressionFeature(config)])
        stream = SnapshotStream()
        for _ in range(20):
            harness.push(
                stream.next(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + TICK_PAISE)
            )
        latest = harness.push(
            stream.next(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE + 5 * TICK_PAISE)
        )
        assert latest["spread_compression"].value < 0.5

    def test_compression_is_invalid_before_warmup(self) -> None:
        harness = FeatureHarness([SpreadCompressionFeature(SpreadCompressionConfig())])
        value = harness.single(make_snapshot())
        assert not value.valid


class TestQueuePersistence:
    CONFIG = QueuePersistenceConfig(window=10, tolerance=0.15, min_samples=3)

    def test_stable_book_reports_balanced_persistence(self) -> None:
        harness = FeatureHarness([QueuePersistenceFeature(self.CONFIG)])
        stream = SnapshotStream()
        latest = harness.push_many(stream.many(8))
        value = latest["queue_persistence"]
        assert value.valid
        assert value.value == pytest.approx(0.0, abs=1e-9)
        assert "bid=1.00" in value.detail
        assert "ask=1.00" in value.detail

    def test_price_keyed_comparison_survives_a_touch_improvement(self) -> None:
        # The regression test for the index-versus-price defect. The best bid
        # improves by one tick, so ``bids[0]`` refers to a different price than it
        # did a snapshot earlier. The previous level is still present one step
        # down the ladder, so persistence must remain high; an index-based
        # implementation would report a collapse.
        harness = FeatureHarness([QueuePersistenceFeature(self.CONFIG)])
        stream = SnapshotStream()
        harness.push_many(stream.many(4))
        latest = harness.push(
            stream.next(
                bid_paise=BASE_PAISE,
                bid_quantities=(120, 600, 450, 340, 250),
            )
        )
        value = latest["queue_persistence"]
        assert value.detail.startswith("bid=1.00")

    def test_vanishing_bid_levels_turn_the_feature_negative(self) -> None:
        harness = FeatureHarness([QueuePersistenceFeature(self.CONFIG)])
        stream = SnapshotStream()
        harness.push_many(stream.many(4))
        latest: dict = {}
        for step in range(1, 7):
            # The best bid keeps dropping away and the abandoned prices do not
            # reappear, so the bid side stops persisting while the ask side holds.
            latest = harness.push(
                stream.next(bid_paise=BASE_PAISE - TICK_PAISE * (1 + step))
            )
        value = latest["queue_persistence"]
        assert value.valid
        assert value.value < 0.0

    def test_invalid_without_a_previous_snapshot(self) -> None:
        harness = FeatureHarness([QueuePersistenceFeature(self.CONFIG)])
        assert not harness.single(make_snapshot()).valid

    def test_partial_quantity_loss_within_tolerance_still_persists(self) -> None:
        harness = FeatureHarness(
            [
                QueuePersistenceFeature(
                    QueuePersistenceConfig(window=6, tolerance=0.5, min_samples=2)
                )
            ]
        )
        stream = SnapshotStream()
        harness.push(stream.next(bid_quantities=(1_000, 500, 400, 300, 200)))
        harness.push(stream.next(bid_quantities=(600, 500, 400, 300, 200)))
        latest = harness.push(stream.next(bid_quantities=(600, 500, 400, 300, 200)))
        assert latest["queue_persistence"].detail.startswith("bid=1.00")


class TestMomentumAndAcceleration:
    def _harness(self) -> FeatureHarness:
        return FeatureHarness(
            [
                WeightedObiFeature(WeightedObiConfig()),
                MomentumFeature(
                    MomentumConfig(
                        fast_half_life_ms=300.0,
                        slow_half_life_ms=3_000.0,
                        warmup_updates=3,
                    )
                ),
                AccelerationFeature(
                    __import__(
                        "snapshot_quant_v4.config", fromlist=["AccelerationConfig"]
                    ).AccelerationConfig(smooth_half_life_ms=300.0)
                ),
            ]
        )

    def test_momentum_turns_positive_as_bid_pressure_builds(self) -> None:
        harness = self._harness()
        stream = SnapshotStream()
        for _ in range(10):
            harness.push(stream.next(bid_quantities=(600,), ask_quantities=(600,)))
        latest: dict = {}
        for step in range(1, 12):
            latest = harness.push(
                stream.next(
                    bid_quantities=(600 + step * 400,), ask_quantities=(600,)
                )
            )
        assert latest[F_MOMENTUM].valid
        assert latest[F_MOMENTUM].value > 0.3
        assert latest[F_MOMENTUM].raw > 0.0

    def test_momentum_turns_negative_as_ask_pressure_builds(self) -> None:
        harness = self._harness()
        stream = SnapshotStream()
        for _ in range(10):
            harness.push(stream.next(bid_quantities=(600,), ask_quantities=(600,)))
        latest: dict = {}
        for step in range(1, 12):
            latest = harness.push(
                stream.next(
                    bid_quantities=(600,), ask_quantities=(600 + step * 400,)
                )
            )
        assert latest[F_MOMENTUM].value < -0.3

    def test_momentum_is_invalid_until_its_emas_warm_up(self) -> None:
        harness = self._harness()
        first = harness.push(make_snapshot())
        assert not first[F_MOMENTUM].valid
        assert not first[F_ACCELERATION].valid

    def test_acceleration_is_a_per_second_rate(self) -> None:
        harness = self._harness()
        stream = SnapshotStream(interval_ms=100)
        for _ in range(12):
            harness.push(stream.next(bid_quantities=(600,), ask_quantities=(600,)))
        latest: dict = {}
        for step in range(1, 10):
            latest = harness.push(
                stream.next(bid_quantities=(600 + step * 600,), ask_quantities=(600,))
            )
        acceleration = latest[F_ACCELERATION]
        assert acceleration.valid
        assert acceleration.value > 0.0
        assert "/s" in acceleration.detail


class TestRefill:
    CONFIG = RefillConfig(
        consumption_fraction=0.4,
        recovery_fraction=0.7,
        window_ms=1_000.0,
        volume_confirmation_ratio=0.5,
        decay_half_life_ms=2_000.0,
        failure_weight=0.5,
    )

    def test_trade_confirmed_consumption_then_recovery_is_bullish(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        harness.push(stream.next(bid_quantities=(1_000, 500, 400, 300, 200)))
        # The touch loses 80 percent of its size while 900 shares print: the drop
        # is explained by trades, so it is consumption rather than cancellation.
        harness.push(
            stream.next(bid_quantities=(200, 500, 400, 300, 200), traded=900)
        )
        latest = harness.push(
            stream.next(bid_quantities=(950, 500, 400, 300, 200), traded=10)
        )
        value = latest["refill"]
        assert value.valid
        assert value.value > 0.0
        assert "refilled" in value.detail

    def test_cancellation_without_trades_creates_no_episode(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        harness.push(stream.next(bid_quantities=(1_000, 500, 400, 300, 200)))
        # Same quantity collapse, but no volume printed: this is a cancellation.
        harness.push(stream.next(bid_quantities=(200, 500, 400, 300, 200), traded=0))
        latest = harness.push(
            stream.next(bid_quantities=(950, 500, 400, 300, 200), traded=0)
        )
        # No consumption was recorded, so the reappearance is not a refill.
        assert latest["refill"].value == pytest.approx(0.0, abs=1e-12)

    def test_plain_growth_never_triggers(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        latest: dict = {}
        for step in range(6):
            latest = harness.push(
                stream.next(
                    bid_quantities=(500 + step * 500, 500, 400, 300, 200), traded=50
                )
            )
        assert latest["refill"].value == pytest.approx(0.0, abs=1e-12)

    def test_consumption_without_recovery_is_bearish(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=300)
        harness.push(stream.next(bid_quantities=(1_000, 500, 400, 300, 200)))
        harness.push(
            stream.next(bid_quantities=(150, 500, 400, 300, 200), traded=900)
        )
        latest: dict = {}
        for _ in range(5):
            latest = harness.push(
                stream.next(bid_quantities=(150, 500, 400, 300, 200), traded=5)
            )
        value = latest["refill"]
        assert value.value < 0.0
        assert "no-refill" in value.detail

    def test_ask_side_refill_is_bearish(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        harness.push(stream.next(ask_quantities=(1_000, 500, 400, 300, 200)))
        harness.push(
            stream.next(ask_quantities=(200, 500, 400, 300, 200), traded=900)
        )
        latest = harness.push(
            stream.next(ask_quantities=(980, 500, 400, 300, 200), traded=10)
        )
        assert latest["refill"].value < 0.0

    def test_invalid_without_a_previous_snapshot(self) -> None:
        harness = FeatureHarness([RefillFeature(self.CONFIG)])
        assert not harness.single(make_snapshot()).valid


class TestLtpConfirmation:
    CONFIG = LtpConfirmationConfig(
        half_life_ms=600.0, scale_ticks=1.0, stale_ms=1_000.0, position_weight=0.4
    )

    def test_rising_traded_price_is_bullish(self) -> None:
        harness = FeatureHarness([LtpConfirmationFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        latest: dict = {}
        for step in range(10):
            price = 800.0 + step * 0.05
            latest = harness.push(
                stream.next(
                    bid_paise=BASE_PAISE + step * TICK_PAISE - TICK_PAISE,
                    ask_paise=BASE_PAISE + step * TICK_PAISE + TICK_PAISE,
                    last_traded_price=price,
                    traded=100,
                )
            )
        value = latest["ltp_confirmation"]
        assert value.valid
        assert value.value > 0.0
        assert "tape=fresh" in value.detail

    def test_falling_traded_price_is_bearish(self) -> None:
        harness = FeatureHarness([LtpConfirmationFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=200)
        latest: dict = {}
        for step in range(10):
            price = 800.0 - step * 0.05
            latest = harness.push(
                stream.next(
                    bid_paise=BASE_PAISE - step * TICK_PAISE - TICK_PAISE,
                    ask_paise=BASE_PAISE - step * TICK_PAISE + TICK_PAISE,
                    last_traded_price=price,
                    traded=100,
                )
            )
        assert latest["ltp_confirmation"].value < 0.0

    def test_stale_tape_reduces_local_confidence(self) -> None:
        harness = FeatureHarness([LtpConfirmationFeature(self.CONFIG)])
        stream = SnapshotStream(interval_ms=400)
        harness.push(stream.next(last_traded_price=800.0, traded=100))
        harness.push(stream.next(last_traded_price=800.0, traded=100))
        latest: dict = {}
        for _ in range(6):
            latest = harness.push(stream.next(last_traded_price=800.0, traded=0))
        value = latest["ltp_confirmation"]
        assert value.local_confidence < 0.5
        assert "tape=stale" in value.detail

    def test_opposition_veto(self) -> None:
        feature = LtpConfirmationFeature(self.CONFIG)
        threshold = self.CONFIG.opposition_threshold
        assert feature.opposes(1, -threshold - 0.01)
        assert not feature.opposes(1, threshold)
        assert feature.opposes(-1, threshold + 0.01)
        assert not feature.opposes(0, -1.0)

    def test_first_snapshot_is_invalid(self) -> None:
        harness = FeatureHarness([LtpConfirmationFeature(self.CONFIG)])
        assert not harness.single(make_snapshot()).valid


class TestFeatureResetContract:
    def test_every_feature_resets_to_its_initial_state(self) -> None:
        # After a feed gap the engine resets every feature. A feature that keeps
        # state across a reset would silently carry pre-gap information into
        # post-gap output.
        config = build_config()
        from snapshot_quant_v4.engine.quant_engine import build_features

        features = build_features(config.features)
        harness = FeatureHarness(features)
        stream = SnapshotStream()
        harness.push_many(stream.many(30, bid_quantities=(2_000, 900, 800, 700, 600)))
        for feature in features:
            feature.reset()
        harness.previous = None
        fresh = harness.push(stream.next())
        # Every history-dependent feature must be invalid again immediately after
        # a reset.
        for name in ("queue_persistence", "refill", "momentum", "acceleration"):
            assert not fresh[name].valid, f"{name} survived reset"
