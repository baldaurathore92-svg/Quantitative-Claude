"""Deterministic tick-by-tick market scenarios for pipeline audits.

These streams are controlled research fixtures, not market simulators and not
strategy evidence. Their purpose is to make the full engine react to known input
shapes while preserving SnapQuote structure: increasing time, sequence and
volume, tick-aligned uncrossed prices, and five sorted depth levels.

``UPWARD`` and ``DOWNWARD`` move exactly one tick per snapshot. ``NOISE`` offers
three zero-sum choppy patterns (whipsaw, burst reversal and volatility cluster).
``RANDOM`` offers three seeded stochastic processes (unbiased walk, mean
reversion and trend switching). Every variant is exactly reproducible after
:meth:`ScenarioSource.start` and owns no global mutable state.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from ..utils.constants import SNAPQUOTE_DEPTH_LEVELS
from ..utils.types import DepthLevel, RawSnapshot

_DEPTH_PROFILE: tuple[float, ...] = (1.00, 0.82, 0.67, 0.55, 0.45)
_MIN_SIDE_SCALE = 0.12
_TREND_BLOCK_LENGTH = 40


class MarketScenario(StrEnum):
    """Top-level deterministic input shapes supported by the source."""

    UPWARD = "UPWARD"
    DOWNWARD = "DOWNWARD"
    NOISE = "NOISE"
    RANDOM = "RANDOM"


class NoisePattern(StrEnum):
    """Directionless high-volatility paths available under ``NOISE``."""

    WHIPSAW = "WHIPSAW"
    BURST_REVERSAL = "BURST_REVERSAL"
    VOLATILITY_CLUSTER = "VOLATILITY_CLUSTER"


class RandomPattern(StrEnum):
    """Seeded stochastic paths available under ``RANDOM``."""

    UNBIASED_WALK = "UNBIASED_WALK"
    MEAN_REVERTING = "MEAN_REVERTING"
    TREND_SWITCHING = "TREND_SWITCHING"


_NOISE_STEPS: dict[NoisePattern, tuple[int, ...]] = {
    NoisePattern.WHIPSAW: (4, -7, 6, -5, 7, -6, 3, -2),
    NoisePattern.BURST_REVERSAL: (1, 1, 1, 6, -1, -1, -1, -6),
    NoisePattern.VOLATILITY_CLUSTER: (1, -1, 1, -1, 8, -8, 7, -7),
}
_NOISE_SPREADS: dict[NoisePattern, tuple[int, ...]] = {
    NoisePattern.WHIPSAW: (1, 3, 2, 4, 1, 4, 2, 3),
    NoisePattern.BURST_REVERSAL: (1, 1, 2, 4, 1, 1, 2, 4),
    NoisePattern.VOLATILITY_CLUSTER: (1, 1, 1, 2, 4, 4, 3, 3),
}


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Configuration shared by all deterministic scenarios and patterns.

    ``start_price_paise`` is the initial best bid. Integer paise arithmetic
    keeps generated prices on the tick grid, while a fixed timestamp origin
    makes complete :class:`RawSnapshot` records equal across restarts.

    Pattern fields are orthogonal: ``noise_pattern`` is consulted only for a
    ``NOISE`` scenario and ``random_pattern`` only for ``RANDOM``. Defaults
    preserve the original whipsaw and unbiased-walk behaviour.
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
    noise_pattern: NoisePattern = NoisePattern.WHIPSAW
    random_pattern: RandomPattern = RandomPattern.UNBIASED_WALK

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

    @property
    def pattern_name(self) -> str:
        """Stable report label for the active scenario/pattern combination."""
        if self.scenario is MarketScenario.NOISE:
            return f"{self.scenario.value}/{self.noise_pattern.value}"
        if self.scenario is MarketScenario.RANDOM:
            return f"{self.scenario.value}/{self.random_pattern.value}"
        return self.scenario.value


class ScenarioSource:
    """Restartable source for one deterministic scenario/pattern combination."""

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
        """Selected top-level scenario."""
        return self._config.scenario

    @property
    def pattern_name(self) -> str:
        """Selected scenario/pattern label used by tests and reports."""
        return self._config.pattern_name

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
        trend_bias = 0
        if (
            config.scenario is MarketScenario.RANDOM
            and config.random_pattern is RandomPattern.TREND_SWITCHING
        ):
            trend_bias = rng.choice((-1, 1))

        for index in range(config.count):
            if self._stop.is_set():
                return

            if (
                config.scenario is MarketScenario.RANDOM
                and config.random_pattern is RandomPattern.TREND_SWITCHING
                and index > 0
                and index % _TREND_BLOCK_LENGTH == 0
            ):
                # Alternating the seeded initial direction guarantees both trend
                # signs are exercised; movement inside each block remains random.
                trend_bias = -trend_bias

            offset_ticks = (best_bid_paise - config.start_price_paise) // config.tick_paise
            step_ticks, pressure, spread_ticks, traded = self._market_step(
                index,
                rng,
                offset_ticks=offset_ticks,
                trend_bias=trend_bias,
            )
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
        *,
        offset_ticks: int,
        trend_bias: int,
    ) -> tuple[int, float, int, int]:
        """Return movement, pressure, spread and traded quantity for one event."""
        config = self._config
        if config.scenario is MarketScenario.UPWARD:
            return 1, 0.78, 1, config.trade_size
        if config.scenario is MarketScenario.DOWNWARD:
            return -1, -0.78, 1, config.trade_size
        if config.scenario is MarketScenario.NOISE:
            return self._noise_step(index, rng)
        return self._random_step(rng, offset_ticks=offset_ticks, trend_bias=trend_bias)

    def _noise_step(
        self,
        index: int,
        rng: random.Random,
    ) -> tuple[int, float, int, int]:
        """Return one event from the selected zero-sum noise cycle."""
        pattern = self._config.noise_pattern
        steps = _NOISE_STEPS[pattern]
        spreads = _NOISE_SPREADS[pattern]
        cycle = index % len(steps)
        step = steps[cycle]
        pressure_magnitude = {
            NoisePattern.WHIPSAW: 0.88,
            NoisePattern.BURST_REVERSAL: 0.82,
            NoisePattern.VOLATILITY_CLUSTER: 0.90,
        }[pattern]
        pressure = pressure_magnitude if step > 0 else -pressure_magnitude
        # WHIPSAW is the original public default. It intentionally consumes no
        # extra RNG draw here, preserving its complete historical stream (trade
        # quantities, cumulative volume and subsequent randomised depth).
        traded = self._config.trade_size + (index % 5) * 17
        if pattern is not NoisePattern.WHIPSAW:
            traded += rng.randint(0, 9)
        return step, pressure, spreads[cycle], traded

    def _random_step(
        self,
        rng: random.Random,
        *,
        offset_ticks: int,
        trend_bias: int,
    ) -> tuple[int, float, int, int]:
        """Return one event from the selected seeded stochastic process."""
        pattern = self._config.random_pattern
        if pattern is RandomPattern.UNBIASED_WALK:
            step = rng.choice((-2, -1, -1, 0, 0, 1, 1, 2))
            structural_pressure = 0.22 * step
        elif pattern is RandomPattern.MEAN_REVERTING:
            shock = rng.choice((-2, -1, -1, 0, 0, 1, 1, 2))
            pull = -1 if offset_ticks >= 4 else 1 if offset_ticks <= -4 else 0
            step = max(-2, min(2, shock + pull))
            structural_pressure = 0.20 * step - 0.025 * offset_ticks
        else:
            step = trend_bias + rng.choice((-1, 0, 0, 0, 1))
            structural_pressure = 0.48 * trend_bias + 0.14 * step

        noise_width = 0.58 if pattern is not RandomPattern.TREND_SWITCHING else 0.24
        pressure = max(
            -0.90,
            min(0.90, structural_pressure + rng.uniform(-noise_width, noise_width)),
        )
        narrow_probability = 0.78 if pattern is not RandomPattern.TREND_SWITCHING else 0.88
        spread = 1 if rng.random() < narrow_probability else rng.choice((2, 3, 4))
        traded = rng.randint(
            max(1, self._config.trade_size // 3),
            self._config.trade_size * 2,
        )
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


__all__ = [
    "MarketScenario",
    "NoisePattern",
    "RandomPattern",
    "ScenarioConfig",
    "ScenarioSource",
]
