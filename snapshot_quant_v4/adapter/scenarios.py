"""Deterministic tick-by-tick market scenarios for pipeline audits.

These streams are controlled research fixtures, not market simulators and not
strategy evidence.  Their purpose is to make the full engine react to four
known input shapes while preserving the structural rules of a SnapQuote book:
strictly increasing time/sequence/volume, tick-aligned uncrossed prices and five
sorted depth levels.

``UPWARD`` and ``DOWNWARD`` move exactly one price tick per snapshot and carry
matching one-sided book pressure. ``NOISE`` uses a zero-sum cycle of irregular
multi-tick changes and rapidly reversing pressure. ``RANDOM`` is a seeded random
walk whose outcome is reproducible but intentionally has no expected direction.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from ..utils.constants import SNAPQUOTE_DEPTH_LEVELS
from ..utils.types import DepthLevel, RawSnapshot

_NOISE_STEPS: tuple[int, ...] = (4, -7, 6, -5, 7, -6, 3, -2)
_NOISE_SPREADS: tuple[int, ...] = (1, 3, 2, 4, 1, 4, 2, 3)
_DEPTH_PROFILE: tuple[float, ...] = (1.00, 0.82, 0.67, 0.55, 0.45)
_MIN_SIDE_SCALE = 0.12


class MarketScenario(StrEnum):
    """Named deterministic input shapes supported by :class:`ScenarioSource`."""

    UPWARD = "UPWARD"
    DOWNWARD = "DOWNWARD"
    NOISE = "NOISE"
    RANDOM = "RANDOM"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Configuration shared by all deterministic scenarios.

    ``start_price_paise`` is the initial best bid, not a floating mid-price.
    Integer paise arithmetic keeps every generated price exactly on the tick
    grid.  A fixed timestamp origin makes complete ``RawSnapshot`` records equal
    across restarts, including their timing fields.
    """

    scenario: MarketScenario = MarketScenario.UPWARD
    token: str = "3045"
    symbol: str = "SYNTH_SCENARIO"
    exchange_type: int = 1
    tick_paise: int = 5
    start_price_paise: int = 80_000
    base_quantity: int = 1_200
    initial_volume: int = 100_000
    trade_size: int = 120
    interval_ms: int = 200
    base_timestamp_ms: int = 1_753_500_000_000
    count: int = 600
    seed: int = 20_260_726

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("token must be non-empty")
        if self.tick_paise <= 0:
            raise ValueError("tick_paise must be positive")
        if self.start_price_paise <= 0:
            raise ValueError("start_price_paise must be positive")
        if self.start_price_paise % self.tick_paise:
            raise ValueError("start_price_paise must be tick-aligned")
        if self.base_quantity <= 0:
            raise ValueError("base_quantity must be positive")
        if self.initial_volume < 0:
            raise ValueError("initial_volume must be non-negative")
        if self.trade_size <= 0:
            raise ValueError("trade_size must be positive")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if self.base_timestamp_ms <= 0:
            raise ValueError("base_timestamp_ms must be positive")
        if self.count <= 0:
            raise ValueError("count must be positive")
        minimum_bid = self.tick_paise * (SNAPQUOTE_DEPTH_LEVELS + 1)
        if (
            self.scenario is MarketScenario.DOWNWARD
            and self.start_price_paise - (self.count - 1) * self.tick_paise
            < minimum_bid
        ):
            raise ValueError("DOWNWARD scenario reaches a non-positive depth price")


class ScenarioSource:
    """Restartable source for one named deterministic market scenario.

    The generator owns no global state. Calling :meth:`start` resets both its
    random generator and emitted counter, so collecting the source twice yields
    byte-for-byte equal snapshots for every scenario.
    """

    __slots__ = ("_config", "_emitted", "_paced", "_rng", "_stop")

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        *,
        paced: bool = False,
    ) -> None:
        self._config = config if config is not None else ScenarioConfig()
        self._paced = paced
        self._stop = threading.Event()
        self._rng = random.Random(self._config.seed)
        self._emitted = 0

    @property
    def emitted(self) -> int:
        """Number of records emitted since the latest :meth:`start`."""
        return self._emitted

    @property
    def scenario(self) -> MarketScenario:
        """The selected named scenario."""
        return self._config.scenario

    def start(self) -> None:
        """Reset lifecycle and pseudo-random state for exact replay."""
        self._stop.clear()
        self._rng = random.Random(self._config.seed)
        self._emitted = 0

    def stop(self) -> None:
        """Request that iteration stops before the next snapshot."""
        self._stop.set()

    def snapshots(self) -> Iterator[RawSnapshot]:
        """Yield the configured valid, deterministic tick-by-tick stream."""
        config = self._config
        rng = self._rng
        best_bid_paise = config.start_price_paise
        volume = config.initial_volume
        session_high_paise = best_bid_paise + config.tick_paise
        session_low_paise = best_bid_paise

        for index in range(config.count):
            if self._stop.is_set():
                return

            step_ticks, pressure, spread_ticks, traded = self._market_step(index, rng)
            if index > 0 or config.scenario in (MarketScenario.NOISE, MarketScenario.RANDOM):
                best_bid_paise += step_ticks * config.tick_paise
            minimum_bid = config.tick_paise * (SNAPQUOTE_DEPTH_LEVELS + 1)
            best_bid_paise = max(minimum_bid, best_bid_paise)
            best_ask_paise = best_bid_paise + spread_ticks * config.tick_paise

            bids = self._ladder(best_bid_paise, -config.tick_paise, pressure, rng)
            asks = self._ladder(best_ask_paise, config.tick_paise, -pressure, rng)
            last_paise = best_ask_paise if pressure >= 0.0 else best_bid_paise
            volume += traded
            session_high_paise = max(session_high_paise, last_paise)
            session_low_paise = min(session_low_paise, last_paise)
            self._emitted += 1
            timestamp_ms = config.base_timestamp_ms + self._emitted * config.interval_ms

            yield RawSnapshot(
                token=config.token,
                exchange_type=config.exchange_type,
                exchange_timestamp_ms=timestamp_ms,
                received_monotonic_ms=float(timestamp_ms),
                sequence_number=self._emitted,
                last_traded_price=last_paise / 100.0,
                last_traded_quantity=traded,
                average_traded_price=(best_bid_paise + best_ask_paise) / 200.0,
                volume_traded_today=volume,
                total_buy_quantity=float(sum(level.quantity for level in bids) * 6),
                total_sell_quantity=float(sum(level.quantity for level in asks) * 6),
                open_price=config.start_price_paise / 100.0,
                high_price=session_high_paise / 100.0,
                low_price=session_low_paise / 100.0,
                close_price=config.start_price_paise / 100.0,
                bids=bids,
                asks=asks,
                symbol=config.symbol,
            )

            if self._paced and self._stop.wait(config.interval_ms / 1_000.0):
                return

    def _market_step(
        self,
        index: int,
        rng: random.Random,
    ) -> tuple[int, float, int, int]:
        """Return movement, pressure, spread and traded quantity for one event."""
        scenario = self._config.scenario
        if scenario is MarketScenario.UPWARD:
            return 1, 0.78, 1, self._config.trade_size
        if scenario is MarketScenario.DOWNWARD:
            return -1, -0.78, 1, self._config.trade_size
        if scenario is MarketScenario.NOISE:
            cycle = index % len(_NOISE_STEPS)
            step = _NOISE_STEPS[cycle]
            pressure = 0.88 if step > 0 else -0.88
            traded = self._config.trade_size + (index % 5) * 17
            return step, pressure, _NOISE_SPREADS[cycle], traded

        step = rng.choice((-2, -1, -1, 0, 0, 1, 1, 2))
        directional_component = 0.22 * step
        pressure = max(-0.90, min(0.90, directional_component + rng.uniform(-0.58, 0.58)))
        spread = 1 if rng.random() < 0.78 else rng.choice((2, 3, 4))
        traded = rng.randint(max(1, self._config.trade_size // 3), self._config.trade_size * 2)
        return step, pressure, spread, traded

    def _ladder(
        self,
        touch_paise: int,
        step_paise: int,
        pressure: float,
        rng: random.Random,
    ) -> tuple[DepthLevel, ...]:
        """Build a sorted five-level side with pressure encoded in quantity."""
        side_scale = max(_MIN_SIDE_SCALE, 1.0 + pressure)
        levels: list[DepthLevel] = []
        randomise = self._config.scenario in (MarketScenario.NOISE, MarketScenario.RANDOM)
        for index, depth_scale in enumerate(_DEPTH_PROFILE):
            variation = rng.uniform(0.92, 1.08) if randomise else 1.0
            quantity = max(
                1,
                round(self._config.base_quantity * depth_scale * side_scale * variation),
            )
            orders = max(1, round(quantity / (150.0 + 25.0 * index)))
            price_paise = touch_paise + step_paise * index
            levels.append(
                DepthLevel(
                    price_paise=price_paise,
                    price=price_paise / 100.0,
                    quantity=quantity,
                    orders=orders,
                )
            )
        return tuple(levels)


__all__ = ["MarketScenario", "ScenarioConfig", "ScenarioSource"]
