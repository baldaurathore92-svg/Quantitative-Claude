"""Deterministic adversarial market/feed fixtures for fail-safe audits.

The normal scenario source produces structurally healthy books. This module has
a different purpose: reproduce extreme but controlled failures that exercise
validator rejection, quality exits, gap resets, warmup recovery and depth-walk
execution. It is still synthetic software input, not a market simulator or
strategy evidence.

Every pattern starts from the same orderly bid-dominated uptrend. Except for the
high-confidence whipsaw and partial-fill patterns, the disturbance begins at
``stress_index`` so tests can first establish an open long position and then
observe whether the engine fails safely.

``DROPPED_OUT_OF_ORDER_FEED`` also records a deliberate protocol limitation:
a forward sequence skip is accepted because SmartAPI sequence values are not
assumed contiguous. The following backwards timestamp is rejected, and the
subsequent sequence regression is accepted as a gap that forces reset/re-warmup.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from ..utils.constants import SNAPQUOTE_DEPTH_LEVELS
from ..utils.types import DepthLevel, RawSnapshot
from .scenarios import MarketScenario, ScenarioConfig, ScenarioSource

_HEAVY_BID: tuple[int, ...] = (2_400, 2_000, 1_600, 1_200, 900)
_LIGHT_ASK: tuple[int, ...] = (400, 350, 300, 250, 200)
_LIGHT_BID: tuple[int, ...] = _LIGHT_ASK
_HEAVY_ASK: tuple[int, ...] = _HEAVY_BID
_VACUUM: tuple[int, ...] = (5, 4, 3, 2, 1)
_PARTIAL_ASK: tuple[int, ...] = (40, 35, 30, 25, 20)
_PARTIAL_BID: tuple[int, ...] = (300, 250, 200, 150, 100)


class ExtremeStressPattern(StrEnum):
    """Adversarial market, feed and execution conditions."""

    FLASH_CRASH = "FLASH_CRASH"
    LIQUIDITY_VACUUM = "LIQUIDITY_VACUUM"
    SPREAD_EXPLOSION = "SPREAD_EXPLOSION"
    HIGH_CONFIDENCE_WHIPSAW = "HIGH_CONFIDENCE_WHIPSAW"
    GAP_REVERSAL = "GAP_REVERSAL"
    DEPTH_EVAPORATION = "DEPTH_EVAPORATION"
    LIMIT_LOCK = "LIMIT_LOCK"
    HALT_AND_REOPEN = "HALT_AND_REOPEN"
    DROPPED_OUT_OF_ORDER_FEED = "DROPPED_OUT_OF_ORDER_FEED"
    PARTIAL_FILL_SLIPPAGE = "PARTIAL_FILL_SLIPPAGE"
    COMBINED_DISASTER = "COMBINED_DISASTER"


@dataclass(frozen=True, slots=True)
class ExtremeStressConfig:
    """Configuration for one deterministic extreme-stress stream."""

    pattern: ExtremeStressPattern = ExtremeStressPattern.FLASH_CRASH
    token: str = "3045"
    symbol: str = "EXTREME_STRESS"
    exchange_type: int = 1
    tick_paise: int = 5
    start_price_paise: int = 80_000
    base_quantity: int = 1_200
    initial_volume: int = 100_000
    trade_size: int = 120
    interval_ms: int = 200
    base_timestamp_ms: int = 1_753_500_000_000
    count: int = 260
    stress_index: int = 120
    seed: int = 20_260_726

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("token must be non-empty")
        if self.tick_paise <= 0:
            raise ValueError("tick_paise must be positive")
        if self.start_price_paise <= self.tick_paise * SNAPQUOTE_DEPTH_LEVELS:
            raise ValueError("start_price_paise is too low for five bid levels")
        if self.start_price_paise % self.tick_paise:
            raise ValueError("start_price_paise must be tick-aligned")
        if self.base_quantity <= 0 or self.trade_size <= 0:
            raise ValueError("base_quantity and trade_size must be positive")
        if self.initial_volume < 0:
            raise ValueError("initial_volume must be non-negative")
        if self.interval_ms <= 0 or self.base_timestamp_ms <= 0:
            raise ValueError("interval and base timestamp must be positive")
        if self.count < 100:
            raise ValueError("count must be at least 100 for warmup and recovery")
        if not 60 <= self.stress_index <= self.count - 12:
            raise ValueError("stress_index must leave warmup and at least 12 recovery ticks")


class ExtremeStressSource:
    """Restartable source of one named adversarial tick-by-tick stream."""

    __slots__ = ("_base", "_config", "_emitted", "_paced", "_stop")

    def __init__(
        self,
        config: ExtremeStressConfig | None = None,
        *,
        paced: bool = False,
    ) -> None:
        self._config = config if config is not None else ExtremeStressConfig()
        self._paced = paced
        self._stop = threading.Event()
        self._emitted = 0
        self._base = self._build_base()

    @property
    def emitted(self) -> int:
        """Number of records emitted since the latest start."""
        return self._emitted

    @property
    def pattern(self) -> ExtremeStressPattern:
        """Selected stress pattern."""
        return self._config.pattern

    def _build_base(self) -> ScenarioSource:
        config = self._config
        return ScenarioSource(
            ScenarioConfig(
                scenario=MarketScenario.UPWARD,
                token=config.token,
                symbol=config.symbol,
                exchange_type=config.exchange_type,
                tick_paise=config.tick_paise,
                start_price_paise=config.start_price_paise,
                base_quantity=config.base_quantity,
                initial_volume=config.initial_volume,
                trade_size=config.trade_size,
                interval_ms=config.interval_ms,
                base_timestamp_ms=config.base_timestamp_ms,
                count=config.count,
                seed=config.seed,
            )
        )

    def start(self) -> None:
        """Reset the base stream and lifecycle for exact replay."""
        self._stop.clear()
        self._emitted = 0
        self._base = self._build_base()
        self._base.start()

    def stop(self) -> None:
        """Request termination before the next snapshot."""
        self._stop.set()
        self._base.stop()

    def snapshots(self) -> Iterator[RawSnapshot]:
        """Yield the configured deterministic stress sequence."""
        for index, raw in enumerate(self._base.snapshots()):
            if self._stop.is_set():
                return
            transformed = self._transform(index, raw)
            self._emitted += 1
            yield transformed
            if self._paced and self._stop.wait(self._config.interval_ms / 1_000.0):
                return

    def _transform(self, index: int, raw: RawSnapshot) -> RawSnapshot:
        pattern = self._config.pattern
        event = self._config.stress_index

        if pattern is ExtremeStressPattern.FLASH_CRASH:
            if index < event:
                return raw
            crash_ticks = -20 * min(index - event + 1, 4)
            shifted = self._shift(raw, crash_ticks)
            return self._rebook(shifted, _LIGHT_BID, _HEAVY_ASK, at_ask=False)

        if pattern is ExtremeStressPattern.LIQUIDITY_VACUUM:
            if event <= index < event + 10:
                return self._rebook(raw, _VACUUM, _VACUUM, at_ask=True)
            return raw

        if pattern is ExtremeStressPattern.SPREAD_EXPLOSION:
            if event <= index < event + 10:
                return self._with_spread(raw, 12)
            return raw

        if pattern is ExtremeStressPattern.HIGH_CONFIDENCE_WHIPSAW:
            return self._orderly_whipsaw(index, raw)

        if pattern is ExtremeStressPattern.GAP_REVERSAL:
            if index == event or index == event + 1:
                return self._shift(raw, 60)
            if index >= event + 2:
                return self._shift(raw, -60)
            return raw

        if pattern is ExtremeStressPattern.DEPTH_EVAPORATION:
            if event <= index < event + 10:
                bid_quantities = (raw.bids[0].quantity, 0, 0, 0, 0)
                ask_quantities = (raw.asks[0].quantity, 0, 0, 0, 0)
                return self._rebook(raw, bid_quantities, ask_quantities, at_ask=True)
            return raw

        if pattern is ExtremeStressPattern.LIMIT_LOCK:
            if event <= index < event + 6:
                return self._with_spread(raw, 0)
            return raw

        if pattern is ExtremeStressPattern.HALT_AND_REOPEN:
            if index >= event:
                shifted = self._shift(raw, 30)
                return replace(
                    shifted,
                    exchange_timestamp_ms=shifted.exchange_timestamp_ms + 5_000,
                )
            return raw

        if pattern is ExtremeStressPattern.DROPPED_OUT_OF_ORDER_FEED:
            if index == event:
                return replace(raw, sequence_number=raw.sequence_number + 10)
            if index == event + 1:
                return replace(
                    raw,
                    exchange_timestamp_ms=raw.exchange_timestamp_ms - 400,
                )
            return raw

        if pattern is ExtremeStressPattern.PARTIAL_FILL_SLIPPAGE:
            limited = self._rebook(raw, _PARTIAL_BID, _PARTIAL_ASK, at_ask=True)
            if index >= event:
                return self._shift(limited, -20)
            return limited

        if pattern is ExtremeStressPattern.COMBINED_DISASTER:
            if event <= index < event + 4:
                collapsed = self._rebook(raw, _VACUUM, _VACUUM, at_ask=True)
                return self._with_spread(collapsed, 12)
            if event + 4 <= index < event + 7:
                return self._with_spread(raw, 0)
            if index == event + 7:
                return replace(
                    raw,
                    exchange_timestamp_ms=raw.exchange_timestamp_ms - 1_000,
                )
            if index >= event + 8:
                return self._shift(raw, -70)
            return raw

        raise AssertionError(f"unhandled stress pattern: {pattern}")

    def _orderly_whipsaw(self, index: int, raw: RawSnapshot) -> RawSnapshot:
        """Twenty one-tick rises followed by twenty one-tick falls."""
        phase = index % 40
        offset = phase if phase <= 20 else 40 - phase
        bid_paise = self._config.start_price_paise + offset * self._config.tick_paise
        positive = phase < 20
        return self._rebook_at(
            raw,
            bid_paise=bid_paise,
            ask_paise=bid_paise + self._config.tick_paise,
            bid_quantities=_HEAVY_BID if positive else _LIGHT_BID,
            ask_quantities=_LIGHT_ASK if positive else _HEAVY_ASK,
            at_ask=positive,
        )

    def _shift(self, raw: RawSnapshot, ticks: int) -> RawSnapshot:
        """Shift the complete visible book by an integer number of ticks."""
        delta = ticks * self._config.tick_paise
        at_ask = raw.last_traded_price == raw.asks[0].price
        return self._rebook_at(
            raw,
            bid_paise=raw.bids[0].price_paise + delta,
            ask_paise=raw.asks[0].price_paise + delta,
            bid_quantities=tuple(level.quantity for level in raw.bids),
            ask_quantities=tuple(level.quantity for level in raw.asks),
            at_ask=at_ask,
        )

    def _with_spread(self, raw: RawSnapshot, spread_ticks: int) -> RawSnapshot:
        """Replace only the visible spread while preserving depth quantities."""
        bid_paise = raw.bids[0].price_paise
        ask_paise = bid_paise + spread_ticks * self._config.tick_paise
        return self._rebook_at(
            raw,
            bid_paise=bid_paise,
            ask_paise=ask_paise,
            bid_quantities=tuple(level.quantity for level in raw.bids),
            ask_quantities=tuple(level.quantity for level in raw.asks),
            at_ask=spread_ticks > 0,
        )

    def _rebook(
        self,
        raw: RawSnapshot,
        bid_quantities: Sequence[int],
        ask_quantities: Sequence[int],
        *,
        at_ask: bool,
    ) -> RawSnapshot:
        return self._rebook_at(
            raw,
            bid_paise=raw.bids[0].price_paise,
            ask_paise=raw.asks[0].price_paise,
            bid_quantities=bid_quantities,
            ask_quantities=ask_quantities,
            at_ask=at_ask,
        )

    def _rebook_at(
        self,
        raw: RawSnapshot,
        *,
        bid_paise: int,
        ask_paise: int,
        bid_quantities: Sequence[int],
        ask_quantities: Sequence[int],
        at_ask: bool,
    ) -> RawSnapshot:
        tick = self._config.tick_paise
        bids = self._ladder(bid_paise, -tick, bid_quantities)
        asks = self._ladder(ask_paise, tick, ask_quantities)
        last_paise = ask_paise if at_ask else bid_paise
        mid = (bid_paise + ask_paise) / 200.0
        last_price = last_paise / 100.0
        return replace(
            raw,
            last_traded_price=last_price,
            average_traded_price=mid,
            total_buy_quantity=float(sum(level.quantity for level in bids) * 6),
            total_sell_quantity=float(sum(level.quantity for level in asks) * 6),
            high_price=max(raw.high_price, last_price),
            low_price=min(raw.low_price, last_price),
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _ladder(
        touch_paise: int,
        step_paise: int,
        quantities: Sequence[int],
    ) -> tuple[DepthLevel, ...]:
        levels = []
        for index, quantity in enumerate(quantities):
            price_paise = touch_paise + index * step_paise
            orders = 0 if quantity <= 0 else max(1, round(quantity / (120 + index * 20)))
            levels.append(
                DepthLevel(
                    price_paise=price_paise,
                    price=price_paise / 100.0,
                    quantity=quantity,
                    orders=orders,
                )
            )
        return tuple(levels)


__all__ = ["ExtremeStressConfig", "ExtremeStressPattern", "ExtremeStressSource"]
