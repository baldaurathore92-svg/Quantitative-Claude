"""Tests for structural validation, derived fields and gap detection."""

from __future__ import annotations

import pytest
from snapshot_quant_v4.config import ValidationConfig
from snapshot_quant_v4.engine.validator import SnapshotValidator
from snapshot_quant_v4.utils.types import RejectReason, Side

from conftest import BASE_PAISE, SYMBOL, TICK_PAISE, TICK_SIZE, ladder, level, make_snapshot


@pytest.fixture
def validator() -> SnapshotValidator:
    """A validator with default thresholds for the reference instrument."""
    return SnapshotValidator(SYMBOL, TICK_SIZE, ValidationConfig())


class TestAcceptance:
    def test_accepts_a_normal_book_and_derives_fields(
        self, validator: SnapshotValidator
    ) -> None:
        result = validator.validate(make_snapshot())
        assert result.accepted
        snapshot = result.snapshot
        assert snapshot is not None
        assert snapshot.mid == pytest.approx(800.0)
        assert snapshot.spread == pytest.approx(0.10)
        assert snapshot.spread_ticks == pytest.approx(2.0)
        assert snapshot.is_first
        assert snapshot.dt_ms == 0.0
        assert snapshot.delta_volume == 0

    def test_second_snapshot_carries_interval_and_volume_delta(
        self, validator: SnapshotValidator
    ) -> None:
        validator.validate(make_snapshot(timestamp_ms=1_000, volume=500))
        result = validator.validate(make_snapshot(timestamp_ms=1_250, volume=800))
        assert result.accepted
        snapshot = result.snapshot
        assert snapshot is not None
        assert snapshot.dt_ms == pytest.approx(250.0)
        assert snapshot.delta_volume == 300
        assert not snapshot.is_first

    def test_price_keyed_lookup_finds_levels_by_price(
        self, validator: SnapshotValidator
    ) -> None:
        result = validator.validate(make_snapshot())
        snapshot = result.snapshot
        assert snapshot is not None
        found = snapshot.level_at(Side.BID, BASE_PAISE - 2 * TICK_PAISE)
        assert found is not None
        assert found.quantity == 450
        assert snapshot.level_at(Side.BID, 12_345) is None

    def test_depth_quantity_sums_requested_levels(
        self, validator: SnapshotValidator
    ) -> None:
        snapshot = validator.validate(make_snapshot()).snapshot
        assert snapshot is not None
        assert snapshot.depth_quantity(Side.BID, 2) == 1050


class TestStructuralRejections:
    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"bids": ()}, RejectReason.MISSING_L1),
            ({"asks": ()}, RejectReason.MISSING_L1),
            (
                {"bids": (level(BASE_PAISE - TICK_PAISE, 0),)},
                RejectReason.ZERO_BID,
            ),
            (
                {"asks": (level(BASE_PAISE + TICK_PAISE, 0),)},
                RejectReason.ZERO_ASK,
            ),
            (
                {"bids": (level(BASE_PAISE - TICK_PAISE, -5),)},
                RejectReason.NEGATIVE_QUANTITY,
            ),
            (
                {"bids": (level(BASE_PAISE - TICK_PAISE, 100, orders=-1),)},
                RejectReason.NEGATIVE_ORDERS,
            ),
        ],
    )
    def test_rejects_malformed_touch(
        self, validator: SnapshotValidator, kwargs: dict, reason: RejectReason
    ) -> None:
        result = validator.validate(make_snapshot(**kwargs))
        assert result.rejected
        assert result.reason is reason
        assert result.snapshot is None

    def test_rejects_crossed_book(self, validator: SnapshotValidator) -> None:
        result = validator.validate(
            make_snapshot(bid_paise=BASE_PAISE + 10, ask_paise=BASE_PAISE - 10)
        )
        assert result.reason is RejectReason.CROSSED_BOOK

    def test_rejects_locked_book_by_default(self, validator: SnapshotValidator) -> None:
        result = validator.validate(
            make_snapshot(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE)
        )
        assert result.reason is RejectReason.LOCKED_BOOK

    def test_locked_book_can_be_allowed(self) -> None:
        validator = SnapshotValidator(
            SYMBOL, TICK_SIZE, ValidationConfig(allow_locked_book=True)
        )
        result = validator.validate(
            make_snapshot(bid_paise=BASE_PAISE, ask_paise=BASE_PAISE)
        )
        assert result.accepted

    def test_rejects_spread_beyond_limit(self) -> None:
        validator = SnapshotValidator(
            SYMBOL, TICK_SIZE, ValidationConfig(max_spread_ticks=4.0)
        )
        result = validator.validate(
            make_snapshot(bid_paise=BASE_PAISE - 50, ask_paise=BASE_PAISE + 50)
        )
        assert result.reason is RejectReason.SPREAD_TOO_WIDE

    def test_rejects_unsorted_ladder(self, validator: SnapshotValidator) -> None:
        broken = (
            level(BASE_PAISE - TICK_PAISE, 500),
            level(BASE_PAISE, 400),  # higher than the touch: not descending
        )
        result = validator.validate(make_snapshot(bids=broken))
        assert result.reason is RejectReason.UNSORTED_DEPTH

    def test_rejects_non_positive_ltp_when_required(
        self, validator: SnapshotValidator
    ) -> None:
        result = validator.validate(make_snapshot(last_traded_price=0.0))
        assert result.reason is RejectReason.INVALID_LTP

    def test_rejects_tick_misalignment_when_configured(self) -> None:
        validator = SnapshotValidator(
            SYMBOL, TICK_SIZE, ValidationConfig(require_tick_alignment=True)
        )
        result = validator.validate(
            make_snapshot(bids=(level(BASE_PAISE - 3, 500),))
        )
        assert result.reason is RejectReason.TICK_MISALIGNED

    def test_rejects_more_than_five_levels(self, validator: SnapshotValidator) -> None:
        six = ladder(BASE_PAISE - TICK_PAISE, -TICK_PAISE, (1, 2, 3, 4, 5, 6))
        result = validator.validate(make_snapshot(bids=six))
        assert result.reason is RejectReason.MALFORMED_PAYLOAD


class TestTemporalRejections:
    def test_rejects_backwards_timestamp_and_flags_a_gap(
        self, validator: SnapshotValidator
    ) -> None:
        validator.validate(make_snapshot(timestamp_ms=2_000))
        result = validator.validate(make_snapshot(timestamp_ms=1_000))
        assert result.reason is RejectReason.NON_MONOTONIC_TIMESTAMP
        assert result.gap_detected

    def test_rejects_stale_snapshot_and_flags_a_gap(self) -> None:
        validator = SnapshotValidator(
            SYMBOL, TICK_SIZE, ValidationConfig(max_staleness_ms=1_000.0)
        )
        validator.validate(make_snapshot(timestamp_ms=1_000))
        result = validator.validate(make_snapshot(timestamp_ms=5_000))
        assert result.reason is RejectReason.STALE_TIMESTAMP
        assert result.gap_detected

    def test_rejects_volume_regression(self, validator: SnapshotValidator) -> None:
        validator.validate(make_snapshot(timestamp_ms=1_000, volume=5_000))
        result = validator.validate(make_snapshot(timestamp_ms=1_200, volume=4_000))
        assert result.reason is RejectReason.VOLUME_REGRESSION
        assert result.gap_detected

    def test_rejects_identical_repeat(self, validator: SnapshotValidator) -> None:
        first = make_snapshot(timestamp_ms=1_000, volume=5_000)
        validator.validate(first)
        result = validator.validate(first)
        assert result.reason is RejectReason.DUPLICATE_SNAPSHOT

    def test_same_timestamp_with_changed_book_is_accepted(
        self, validator: SnapshotValidator
    ) -> None:
        validator.validate(make_snapshot(timestamp_ms=1_000, volume=5_000))
        changed = make_snapshot(
            timestamp_ms=1_000, volume=5_000, bid_quantities=(999, 450, 340, 250, 190)
        )
        assert validator.validate(changed).accepted


class TestGapHandling:
    def test_long_silence_is_a_gap_and_resets_derived_fields(self) -> None:
        validator = SnapshotValidator(
            SYMBOL,
            TICK_SIZE,
            ValidationConfig(max_snapshot_gap_ms=500.0, max_staleness_ms=10_000.0),
        )
        validator.validate(make_snapshot(timestamp_ms=1_000, volume=1_000))
        result = validator.validate(make_snapshot(timestamp_ms=4_000, volume=9_000))
        assert result.accepted
        assert result.gap_detected
        snapshot = result.snapshot
        assert snapshot is not None
        # Interval and volume delta measured across a gap are meaningless, so the
        # snapshot is presented as the first of a new sequence.
        assert snapshot.is_first
        assert snapshot.dt_ms == 0.0
        assert snapshot.delta_volume == 0

    def test_large_mid_jump_is_a_gap(self) -> None:
        validator = SnapshotValidator(
            SYMBOL, TICK_SIZE, ValidationConfig(max_price_gap_ticks=5.0)
        )
        validator.validate(make_snapshot(timestamp_ms=1_000))
        jumped = make_snapshot(
            timestamp_ms=1_200,
            bid_paise=BASE_PAISE + 500,
            ask_paise=BASE_PAISE + 520,
        )
        result = validator.validate(jumped)
        assert result.accepted
        assert result.gap_detected

    def test_reset_forgets_history(self, validator: SnapshotValidator) -> None:
        validator.validate(make_snapshot(timestamp_ms=5_000))
        validator.reset()
        assert not validator.has_history
        # A timestamp older than the forgotten one is now acceptable again.
        assert validator.validate(make_snapshot(timestamp_ms=1_000)).accepted

    def test_rejects_non_positive_tick_size(self) -> None:
        with pytest.raises(ValueError):
            SnapshotValidator(SYMBOL, 0.0, ValidationConfig())
