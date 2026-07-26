"""Deterministic synthetic book generator.

Purpose
-------
This is a *test and demonstration* source, and it is labelled as such so nobody
mistakes its output for market data. It exists for three concrete reasons:

*   the console renderer, the state machine and the runner can be exercised
    end-to-end without credentials and without a live session;
*   the benchmark needs a reproducible input of arbitrary length;
*   unit tests need books with specific, controllable properties (a one-sided
    imbalance, a consumption-then-refill episode, a widening spread).

The generator is driven by an explicitly seeded :class:`random.Random`, so a given
seed always produces the same sequence. It models a mid-price random walk on the
tick grid with mean reversion, a spread that occasionally widens, depth that
decays away from the touch, and traded volume that only increases. It does **not**
attempt to reproduce real microstructure, and no result derived from it should be
reported as evidence about a strategy.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

from ..utils.constants import SNAPQUOTE_DEPTH_LEVELS
from ..utils.types import DepthLevel, RawSnapshot


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Parameters of the synthetic book."""

    token: str = "3045"
    symbol: str = "SYNTH"
    exchange_type: int = 1
    tick_paise: int = 5
    start_price_paise: int = 80_000
    base_quantity: int = 600
    depth_decay: float = 0.75
    imbalance_persistence: float = 0.92
    spread_widen_probability: float = 0.05
    mean_reversion: float = 0.02
    interval_ms_min: int = 120
    interval_ms_max: int = 420
    trade_probability: float = 0.55
    trade_size_max: int = 400
    count: int = 2_000
    seed: int = 20260726

    def __post_init__(self) -> None:
        if self.tick_paise <= 0:
            raise ValueError("tick_paise must be positive")
        if self.start_price_paise <= 0:
            raise ValueError("start_price_paise must be positive")
        if self.base_quantity <= 0:
            raise ValueError("base_quantity must be positive")
        if not 0.0 < self.depth_decay < 1.0:
            raise ValueError("depth_decay must be in (0, 1)")
        if not 0.0 <= self.imbalance_persistence < 1.0:
            raise ValueError("imbalance_persistence must be in [0, 1)")
        if self.interval_ms_min <= 0 or self.interval_ms_max < self.interval_ms_min:
            raise ValueError("invalid interval bounds")
        if self.count <= 0:
            raise ValueError("count must be positive")


class SyntheticSource:
    """Generates a reproducible stream of synthetic snapshots.

    Implements :class:`~snapshot_quant_v4.utils.types.MarketDataSource`.

    Parameters
    ----------
    config:
        Generator parameters.
    paced:
        When ``True``, sleeps for the generated interval so the console output is
        watchable. Benchmarks leave it ``False``.
    """

    __slots__ = ("_config", "_emitted", "_paced", "_rng", "_stop")

    def __init__(self, config: SyntheticConfig | None = None, *, paced: bool = False) -> None:
        self._config = config if config is not None else SyntheticConfig()
        self._paced = paced
        self._stop = threading.Event()
        self._rng = random.Random(self._config.seed)
        self._emitted = 0

    @property
    def emitted(self) -> int:
        """Number of snapshots generated."""
        return self._emitted

    def start(self) -> None:
        """Reset the generator so a source can be restarted deterministically."""
        self._stop.clear()
        self._rng = random.Random(self._config.seed)
        self._emitted = 0

    def stop(self) -> None:
        """Request that generation end."""
        self._stop.set()

    def snapshots(self) -> Iterator[RawSnapshot]:
        """Yield the configured number of synthetic snapshots."""
        config = self._config
        rng = self._rng
        mid_paise = config.start_price_paise
        anchor_paise = config.start_price_paise
        imbalance = 0.0
        spread_ticks = 1
        volume = 0
        timestamp_ms = int(time.time() * 1000.0)
        last_price_paise = mid_paise

        for _ in range(config.count):
            if self._stop.is_set():
                return

            # Mid-price walk with weak mean reversion, on the tick grid.
            pull = (anchor_paise - mid_paise) * config.mean_reversion
            step = rng.choice((-1, 0, 0, 1)) * config.tick_paise
            mid_paise = max(config.tick_paise * 2, mid_paise + step + int(pull))

            # Persistent queue imbalance: an autoregressive process, so pressure
            # lasts several snapshots instead of being independent noise.
            imbalance = (
                imbalance * config.imbalance_persistence
                + rng.gauss(0.0, 0.35) * (1.0 - config.imbalance_persistence)
            )
            imbalance = max(-0.9, min(0.9, imbalance))

            spread_ticks = (
                rng.randint(2, 4)
                if rng.random() < config.spread_widen_probability
                else 1
            )

            half = spread_ticks * config.tick_paise / 2.0
            best_bid = round((mid_paise - half) / config.tick_paise) * config.tick_paise
            best_ask = best_bid + spread_ticks * config.tick_paise

            bids = self._ladder(best_bid, -config.tick_paise, imbalance, rng)
            asks = self._ladder(best_ask, config.tick_paise, -imbalance, rng)

            traded = 0
            if rng.random() < config.trade_probability:
                traded = rng.randint(1, config.trade_size_max)
                volume += traded
                last_price_paise = best_ask if imbalance > 0.0 else best_bid

            interval = rng.randint(config.interval_ms_min, config.interval_ms_max)
            timestamp_ms += interval
            self._emitted += 1

            yield RawSnapshot(
                token=config.token,
                exchange_type=config.exchange_type,
                exchange_timestamp_ms=timestamp_ms,
                received_monotonic_ms=time.monotonic_ns() / 1_000_000.0,
                sequence_number=self._emitted,
                last_traded_price=last_price_paise / 100.0,
                last_traded_quantity=traded,
                average_traded_price=mid_paise / 100.0,
                volume_traded_today=volume,
                total_buy_quantity=float(sum(level.quantity for level in bids) * 6),
                total_sell_quantity=float(sum(level.quantity for level in asks) * 6),
                open_price=config.start_price_paise / 100.0,
                high_price=(mid_paise + 40) / 100.0,
                low_price=(mid_paise - 40) / 100.0,
                close_price=config.start_price_paise / 100.0,
                bids=bids,
                asks=asks,
                symbol=config.symbol,
            )

            if self._paced and self._stop.wait(interval / 1000.0):
                return

    def _ladder(
        self,
        touch_paise: int,
        step_paise: int,
        tilt: float,
        rng: random.Random,
    ) -> tuple[DepthLevel, ...]:
        """Build one side of the ladder, decaying away from the touch."""
        config = self._config
        levels: list[DepthLevel] = []
        scale = 1.0 + tilt
        for index in range(SNAPQUOTE_DEPTH_LEVELS):
            price_paise = touch_paise + step_paise * index
            if price_paise <= 0:
                break
            decay = config.depth_decay**index
            quantity = max(
                1, round(config.base_quantity * scale * decay * rng.uniform(0.7, 1.3))
            )
            orders = max(1, round(quantity / rng.uniform(80.0, 260.0)))
            levels.append(
                DepthLevel(
                    price_paise=price_paise,
                    price=price_paise / 100.0,
                    quantity=quantity,
                    orders=orders,
                )
            )
        return tuple(levels)


__all__ = ["SyntheticConfig", "SyntheticSource"]
