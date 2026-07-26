"""Shared test fixtures and book builders.

The helpers here exist so that every test states *only* the market condition it
cares about. A test about crossed books should not contain thirty lines of
snapshot construction, and a test about refill should be able to say "the touch
lost 80 percent of its size while 500 shares traded" in one call.

All builders produce fully populated, internally consistent snapshots so that a
test failure means the code under test is wrong, not the fixture.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from snapshot_quant_v4.config import AppConfig, config_from_mapping
from snapshot_quant_v4.utils.types import DepthLevel, RawSnapshot

#: Reference instrument used throughout the tests.
TOKEN = "3045"
SYMBOL = "SBIN"
TICK_SIZE = 0.05
TICK_PAISE = 5
BASE_PAISE = 80_000
BASE_TIMESTAMP_MS = 1_753_500_000_000


def level(price_paise: int, quantity: int, orders: int = 3) -> DepthLevel:
    """Build one depth level from an integer paise price."""
    return DepthLevel(
        price_paise=price_paise,
        price=price_paise / 100.0,
        quantity=quantity,
        orders=orders,
    )


def ladder(
    touch_paise: int,
    step_paise: int,
    quantities: Sequence[int],
    orders: int = 3,
) -> tuple[DepthLevel, ...]:
    """Build one side of a ladder walking away from the touch."""
    return tuple(
        level(touch_paise + step_paise * index, quantity, orders)
        for index, quantity in enumerate(quantities)
    )


def make_snapshot(
    *,
    bid_paise: int = BASE_PAISE - TICK_PAISE,
    ask_paise: int = BASE_PAISE + TICK_PAISE,
    bid_quantities: Sequence[int] = (600, 450, 340, 250, 190),
    ask_quantities: Sequence[int] = (600, 450, 340, 250, 190),
    timestamp_ms: int = BASE_TIMESTAMP_MS,
    volume: int = 100_000,
    last_traded_price: float | None = None,
    sequence: int = 1,
    total_buy_quantity: float = 50_000.0,
    total_sell_quantity: float = 50_000.0,
    orders: int = 3,
    bids: tuple[DepthLevel, ...] | None = None,
    asks: tuple[DepthLevel, ...] | None = None,
) -> RawSnapshot:
    """Build a complete, valid :class:`RawSnapshot`.

    Every parameter has a sane default so a test overrides only what it is
    actually testing.
    """
    resolved_bids = (
        bids
        if bids is not None
        else ladder(bid_paise, -TICK_PAISE, bid_quantities, orders)
    )
    resolved_asks = (
        asks
        if asks is not None
        else ladder(ask_paise, TICK_PAISE, ask_quantities, orders)
    )
    mid = (bid_paise + ask_paise) / 200.0
    return RawSnapshot(
        token=TOKEN,
        exchange_type=1,
        exchange_timestamp_ms=timestamp_ms,
        received_monotonic_ms=float(timestamp_ms),
        sequence_number=sequence,
        last_traded_price=mid if last_traded_price is None else last_traded_price,
        last_traded_quantity=25,
        average_traded_price=mid,
        volume_traded_today=volume,
        total_buy_quantity=total_buy_quantity,
        total_sell_quantity=total_sell_quantity,
        open_price=BASE_PAISE / 100.0,
        high_price=(BASE_PAISE + 100) / 100.0,
        low_price=(BASE_PAISE - 100) / 100.0,
        close_price=BASE_PAISE / 100.0,
        bids=resolved_bids,
        asks=resolved_asks,
        symbol=SYMBOL,
    )


@dataclass(slots=True)
class SnapshotStream:
    """Generates a sequence of related snapshots with advancing time and volume.

    Keeps the bookkeeping (timestamps, cumulative volume, sequence numbers) out of
    the tests, since getting that bookkeeping wrong is what makes hand-written
    market-data fixtures unreliable.
    """

    timestamp_ms: int = BASE_TIMESTAMP_MS
    volume: int = 100_000
    sequence: int = 0
    interval_ms: int = 200

    def next(
        self,
        *,
        traded: int = 0,
        interval_ms: int | None = None,
        **kwargs: Any,
    ) -> RawSnapshot:
        """Produce the next snapshot, advancing time and cumulative volume."""
        self.timestamp_ms += interval_ms if interval_ms is not None else self.interval_ms
        self.volume += traded
        self.sequence += 1
        kwargs.setdefault("timestamp_ms", self.timestamp_ms)
        kwargs.setdefault("volume", self.volume)
        kwargs.setdefault("sequence", self.sequence)
        return make_snapshot(**kwargs)

    def many(self, count: int, **kwargs: Any) -> list[RawSnapshot]:
        """Produce ``count`` snapshots with identical parameters."""
        return [self.next(**kwargs) for _ in range(count)]


def build_config(**sections: Any) -> AppConfig:
    """Build an :class:`AppConfig` for the reference instrument.

    Sections are merged over the defaults, so a test states only the parameters it
    depends on and remains valid when unrelated defaults change.
    """
    data: dict[str, Any] = {
        "symbols": [
            {
                "token": TOKEN,
                "symbol": SYMBOL,
                "exchange_type": 1,
                "tick_size": TICK_SIZE,
                "quantity": 1,
            }
        ]
    }
    data.update(sections)
    return config_from_mapping(data)


@pytest.fixture
def config() -> AppConfig:
    """Default configuration for the reference instrument."""
    return build_config()


@pytest.fixture
def stream() -> SnapshotStream:
    """A fresh snapshot stream."""
    return SnapshotStream()


def quantities_scaled(base: Sequence[int], factor: float) -> tuple[int, ...]:
    """Scale a quantity ladder, keeping every level at least one lot."""
    return tuple(max(1, round(quantity * factor)) for quantity in base)


def collect(values: Iterable[float]) -> list[float]:
    """Materialise an iterable of floats, for readable assertions."""
    return list(values)



class FeatureHarness:
    """Drives one or more features over a snapshot sequence.

    Reproduces exactly the order the engine uses -- validate, update the shared
    statistics, then evaluate features with the previous snapshot available --
    without pulling in the regime detector, confidence model or state machine. A
    feature test therefore exercises the same code path as production while
    asserting on the feature in isolation.
    """

    def __init__(
        self,
        features: Sequence[Any],
        *,
        app_config: AppConfig | None = None,
    ) -> None:
        from snapshot_quant_v4.engine.stats import SharedStatistics
        from snapshot_quant_v4.engine.validator import SnapshotValidator

        resolved = app_config if app_config is not None else build_config()
        self.config = resolved
        self.features = list(features)
        self.validator = SnapshotValidator(SYMBOL, TICK_SIZE, resolved.validation)
        self.stats = SharedStatistics(
            tick_size=TICK_SIZE,
            confidence=resolved.confidence,
            regime=resolved.regime,
            threshold=resolved.threshold,
            quality=resolved.quality,
        )
        self.previous: Any = None
        self.snapshot: Any = None

    def push(self, raw: RawSnapshot) -> dict[str, Any]:
        """Validate and evaluate one snapshot, returning the feature map.

        Raises
        ------
        AssertionError
            If the snapshot is rejected. A feature test should never depend on a
            book the validator would have discarded.
        """
        from snapshot_quant_v4.features.base import FeatureContext

        result = self.validator.validate(raw)
        assert result.accepted, f"validator rejected the fixture: {result.reason}"
        snapshot = result.snapshot
        assert snapshot is not None
        self.stats.update_pre(snapshot)
        computed: dict[str, Any] = {}
        context = FeatureContext(
            snapshot=snapshot,
            previous=self.previous,
            stats=self.stats,
            computed=computed,
        )
        for feature in self.features:
            computed[feature.name] = feature.compute(context)
        obi = computed.get("weighted_obi")
        if obi is not None and obi.valid:
            self.stats.update_post(obi.value)
        self.previous = snapshot
        self.snapshot = snapshot
        return computed

    def push_many(self, snapshots: Iterable[RawSnapshot]) -> dict[str, Any]:
        """Push several snapshots and return the last feature map."""
        latest: dict[str, Any] = {}
        for raw in snapshots:
            latest = self.push(raw)
        return latest

    def single(self, raw: RawSnapshot) -> Any:
        """Push one snapshot and return the sole feature's value."""
        if len(self.features) != 1:
            raise ValueError("single() requires exactly one feature")
        return self.push(raw)[self.features[0].name]



def validate_one(raw: RawSnapshot, app_config: AppConfig | None = None) -> Any:
    """Validate a single snapshot and return the enriched :class:`Snapshot`.

    For tests of components downstream of the validator that need a real
    ``Snapshot`` but do not care about the validation itself.
    """
    from snapshot_quant_v4.engine.validator import SnapshotValidator

    resolved = app_config if app_config is not None else build_config()
    validator = SnapshotValidator(SYMBOL, TICK_SIZE, resolved.validation)
    result = validator.validate(raw)
    assert result.accepted, f"fixture rejected: {result.reason}"
    return result.snapshot


def feature_value(
    name: str,
    value: float,
    *,
    confidence: float = 1.0,
    valid: bool = True,
    raw: float | None = None,
    directional: bool = True,
    local_confidence: float = 1.0,
) -> Any:
    """Build a :class:`FeatureValue` for scoring and confidence tests."""
    from snapshot_quant_v4.utils.types import FeatureKind, FeatureValue

    return FeatureValue(
        name=name,
        kind=FeatureKind.DIRECTIONAL if directional else FeatureKind.QUALITY,
        raw=value if raw is None else raw,
        value=value,
        local_confidence=local_confidence,
        confidence=confidence,
        valid=valid,
        detail="",
    )


def quality_report(
    *,
    tradable: bool = True,
    reasons: tuple[Any, ...] = (),
    book_quality: float = 1.0,
    liquidity_score: float = 1.0,
) -> Any:
    """Build a :class:`QualityReport` for downstream component tests."""
    from snapshot_quant_v4.utils.types import QualityReport

    return QualityReport(
        tradable=tradable,
        reasons=reasons,
        book_quality=book_quality,
        liquidity_score=liquidity_score,
        detail="",
    )
