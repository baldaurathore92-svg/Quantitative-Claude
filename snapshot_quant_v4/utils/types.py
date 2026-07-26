"""Core data model of the Snapshot Quant Engine.

All records are immutable (``frozen=True``) and slotted (``slots=True``):

*   Immutability means a snapshot handed to ten features cannot be mutated by
    the third one, which removes an entire class of ordering bugs and keeps the
    engine deterministic.
*   ``slots=True`` removes the per-instance ``__dict__``, which cuts both the
    allocation cost and the memory footprint of the objects that are created
    once per snapshot.

The module contains no behaviour beyond cheap derived accessors, and imports
nothing from the engine, so it can be used freely by tests and by offline
research tooling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Protocol, runtime_checkable

from .constants import DEFAULT_TICK_SIZE
from .math_utils import safe_div

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Side(StrEnum):
    """Direction of a book side or a position."""

    BID = "BID"
    ASK = "ASK"


class Direction(Enum):
    """Signed trade direction."""

    FLAT = 0
    LONG = 1
    SHORT = -1

    @property
    def signum(self) -> int:
        """Return ``+1`` for long, ``-1`` for short, ``0`` for flat."""
        return self.value


class Regime(StrEnum):
    """Deterministic market regime label."""

    TREND = "TREND"
    PULLBACK = "PULLBACK"
    RANGE = "RANGE"
    NOISE = "NOISE"
    UNKNOWN = "UNKNOWN"


class TradeState(StrEnum):
    """State-machine states.

    ``WARMUP`` and ``COOLDOWN`` are additions to the originally specified set.
    They exist for correctness rather than convenience:

    ``WARMUP``
        Rolling estimators are not yet populated (or were just invalidated by a
        feed gap). Emitting signals from a half-filled variance estimator is
        the single easiest way to generate spurious entries.
    ``COOLDOWN``
        Prevents immediate re-entry churn after an exit, which a purely
        threshold-driven machine would otherwise do on every oscillation
        around the threshold.
    """

    WARMUP = "WARMUP"
    NEUTRAL = "NEUTRAL"
    WATCH_LONG = "WATCH_LONG"
    WATCH_SHORT = "WATCH_SHORT"
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    COOLDOWN = "COOLDOWN"


class FeatureKind(StrEnum):
    """How a feature participates in the composite score.

    ``DIRECTIONAL``
        Carries a sign that means "up" or "down". Only these features enter the
        composite score.
    ``QUALITY``
        Has magnitude but no direction (spread width, spread compression).
        Feeding an unsigned quantity into a signed sum would be a category
        error, so these features are routed into the confidence model instead.
    """

    DIRECTIONAL = "DIRECTIONAL"
    QUALITY = "QUALITY"


class RejectReason(StrEnum):
    """Why a raw snapshot was rejected by the validator."""

    NONE = "NONE"
    MISSING_L1 = "MISSING_L1"
    ZERO_BID = "ZERO_BID"
    ZERO_ASK = "ZERO_ASK"
    NEGATIVE_QUANTITY = "NEGATIVE_QUANTITY"
    NEGATIVE_ORDERS = "NEGATIVE_ORDERS"
    INVALID_PRICE = "INVALID_PRICE"
    CROSSED_BOOK = "CROSSED_BOOK"
    LOCKED_BOOK = "LOCKED_BOOK"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    UNSORTED_DEPTH = "UNSORTED_DEPTH"
    NON_MONOTONIC_TIMESTAMP = "NON_MONOTONIC_TIMESTAMP"
    STALE_TIMESTAMP = "STALE_TIMESTAMP"
    DUPLICATE_SNAPSHOT = "DUPLICATE_SNAPSHOT"
    INVALID_LTP = "INVALID_LTP"
    VOLUME_REGRESSION = "VOLUME_REGRESSION"
    TICK_MISALIGNED = "TICK_MISALIGNED"
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    MALFORMED_PAYLOAD = "MALFORMED_PAYLOAD"


class BlockReason(StrEnum):
    """Why an otherwise valid snapshot may not produce a fresh signal."""

    NONE = "NONE"
    WARMUP = "WARMUP"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    BOOK_TOO_THIN = "BOOK_TOO_THIN"
    LIQUIDITY_BELOW_THRESHOLD = "LIQUIDITY_BELOW_THRESHOLD"
    FEED_GAP = "FEED_GAP"
    PRICE_GAP = "PRICE_GAP"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SESSION_CLOSED = "SESSION_CLOSED"
    INSUFFICIENT_DEPTH_LEVELS = "INSUFFICIENT_DEPTH_LEVELS"


# --------------------------------------------------------------------------- #
# Market data records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DepthLevel:
    """A single price level of the limit order book.

    Attributes
    ----------
    price_paise:
        Integer price in the wire unit (paise for equity segments). Integer
        keys are what make *price-keyed* comparison between consecutive
        snapshots exact; comparing floats with ``==`` would be unsound.
    price:
        The same price in rupees, pre-computed once to keep the hot path free
        of divisions.
    quantity:
        Aggregate quantity resting at this price.
    orders:
        Number of orders aggregated into ``quantity``. SnapQuote publishes this
        per level; it is the only field that says anything about the
        *granularity* of the queue.
    """

    price_paise: int
    price: float
    quantity: int
    orders: int

    @property
    def average_order_size(self) -> float:
        """Mean quantity per order at this level (0.0 when unknown)."""
        return safe_div(float(self.quantity), float(self.orders))


#: An empty level, used as a typed sentinel instead of ``None``.
EMPTY_LEVEL: DepthLevel = DepthLevel(price_paise=0, price=0.0, quantity=0, orders=0)


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    """A parsed but *unvalidated* SnapQuote (mode 3) message.

    The adapter is responsible for producing this record: it converts wire
    integers into rupees exactly once, and normalises the several possible
    field spellings of the SmartAPI payload. Nothing downstream ever sees a raw
    dictionary.
    """

    token: str
    exchange_type: int
    exchange_timestamp_ms: int
    received_monotonic_ms: float
    sequence_number: int
    last_traded_price: float
    last_traded_quantity: int
    average_traded_price: float
    volume_traded_today: int
    total_buy_quantity: float
    total_sell_quantity: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    symbol: str = ""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A validated snapshot enriched with derived, reusable quantities.

    Derived fields are computed once, here, rather than inside each feature.
    That removes duplicated logic (the stated requirement) and also removes
    nine redundant divisions per snapshot.

    ``delta_volume`` is the increase in the cumulative day volume since the
    previous accepted snapshot. It is the *only* trade-level information a
    snapshot feed provides, and it is what allows the refill feature to tell a
    queue that was *traded away* from a queue that was *cancelled away*.
    """

    raw: RawSnapshot
    symbol: str
    token: str
    tick_size: float
    exchange_timestamp_ms: int
    received_monotonic_ms: float
    dt_ms: float
    is_first: bool
    delta_volume: int
    best_bid: DepthLevel
    best_ask: DepthLevel
    mid: float
    spread: float
    spread_ticks: float
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]

    @property
    def last_traded_price(self) -> float:
        """Last traded price in rupees."""
        return self.raw.last_traded_price

    @property
    def total_buy_quantity(self) -> float:
        """Exchange-published aggregate resting buy quantity (whole book)."""
        return self.raw.total_buy_quantity

    @property
    def total_sell_quantity(self) -> float:
        """Exchange-published aggregate resting sell quantity (whole book)."""
        return self.raw.total_sell_quantity

    def levels(self, side: Side) -> tuple[DepthLevel, ...]:
        """Return the depth ladder for ``side``, best price first."""
        return self.bids if side is Side.BID else self.asks

    def level_at(self, side: Side, price_paise: int) -> DepthLevel | None:
        """Return the level resting at an exact price, or ``None``.

        This is the primitive that makes price-keyed comparison possible. It
        scans at most :data:`constants.SNAPQUOTE_DEPTH_LEVELS` entries, so it
        is O(1) with a small constant and allocates nothing — deliberately
        cheaper than building a dictionary per snapshot.
        """
        for level in self.bids if side is Side.BID else self.asks:
            if level.price_paise == price_paise:
                return level
        return None

    def depth_quantity(self, side: Side, levels: int) -> int:
        """Sum the quantity of the first ``levels`` price levels of a side."""
        ladder = self.bids if side is Side.BID else self.asks
        total = 0
        for index, level in enumerate(ladder):
            if index >= levels:
                break
            total += level.quantity
        return total


# --------------------------------------------------------------------------- #
# Validation / gating results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of :class:`engine.validator.SnapshotValidator`."""

    accepted: bool
    reason: RejectReason
    snapshot: Snapshot | None
    detail: str = ""
    gap_detected: bool = False

    @property
    def rejected(self) -> bool:
        """Convenience inverse of :attr:`accepted`."""
        return not self.accepted


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Outcome of the market-quality gate."""

    tradable: bool
    reasons: tuple[BlockReason, ...]
    book_quality: float
    liquidity_score: float
    detail: str = ""


# --------------------------------------------------------------------------- #
# Feature records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """The output of a single feature for a single snapshot.

    Attributes
    ----------
    name:
        Canonical feature name (see :mod:`utils.constants`).
    kind:
        Whether the value is directional or a quality metric.
    raw:
        The value in its native, interpretable unit (ticks, ratio, per-second
        rate). Retained for logging, research and unit tests: a normalised
        number alone is impossible to sanity-check.
    value:
        The normalised value. Directional features are in ``[-1, +1]``;
        quality features are in ``[0, 1]``.
    local_confidence:
        Reliability that only the feature itself can judge, for example the
        agreement between the L1/L2 imbalance and the deeper levels. The engine
        multiplies this into the confidence produced by the shared model.
    confidence:
        Final confidence in ``[0, 1]``, filled in by the confidence model.
    valid:
        ``False`` while the feature's own estimators are not yet usable. An
        invalid feature is excluded from the composite instead of contributing
        a zero, which would silently bias the score towards neutral.
    detail:
        Short human-readable explanation used by the console renderer.
    """

    name: str
    kind: FeatureKind
    raw: float
    value: float
    local_confidence: float = 1.0
    confidence: float = 0.0
    valid: bool = True
    detail: str = ""

    @property
    def directional(self) -> bool:
        """``True`` when the feature carries a meaningful sign."""
        return self.kind is FeatureKind.DIRECTIONAL


#: Immutable view of all features computed for one snapshot.
FeatureMap = Mapping[str, FeatureValue]


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's share of the composite score, for display and audit."""

    name: str
    value: float
    weight: float
    confidence: float
    contribution: float


@dataclass(frozen=True, slots=True)
class CompositeResult:
    """Output of the composite scorer."""

    score: float
    smoothed: float
    confidence: float
    weight_mass: float
    used_features: int
    contributions: tuple[FeatureContribution, ...]
    valid: bool


@dataclass(frozen=True, slots=True)
class RegimeResult:
    """Output of the deterministic regime detector."""

    regime: Regime
    efficiency_ratio: float
    volatility_ticks: float
    trend_ticks: float
    dwell_snapshots: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Output of the adaptive threshold policy."""

    entry: float
    watch: float
    exit: float
    floor_applied: bool
    detail: str = ""


# --------------------------------------------------------------------------- #
# Execution records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FillQuote:
    """A modelled execution price.

    Attributes
    ----------
    price:
        Tick-aligned price at which the model assumes the order fills.
    reference_price:
        Touch price the quote was derived from (best ask for a buy, best bid
        for a sell).
    aggression_ticks:
        How far through the touch the quote reaches.
    filled_quantity / requested_quantity:
        When depth-walking is enabled these differ if the visible ladder cannot
        absorb the request. A partial fill is reported rather than silently
        assumed away.
    levels_consumed:
        Number of price levels the walk had to consume.
    complete:
        ``True`` when ``filled_quantity == requested_quantity``.
    """

    price: float
    reference_price: float
    aggression_ticks: float
    requested_quantity: int
    filled_quantity: int
    levels_consumed: int
    complete: bool


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Round-trip transaction cost, expressed both ways.

    Kept entirely outside the signal path: the engine can run with costs
    disabled, and research code can vary the cost model without touching a
    single feature.
    """

    brokerage_bps: float
    exchange_bps: float
    statutory_bps: float
    slippage_bps: float

    @property
    def total_bps(self) -> float:
        """Total round-trip cost in basis points of notional."""
        return (
            self.brokerage_bps
            + self.exchange_bps
            + self.statutory_bps
            + self.slippage_bps
        )


@dataclass(frozen=True, slots=True)
class Position:
    """An open modelled position.

    ``remaining_entry_cost_rupees`` carries the unallocated cost of the original
    entry order. Partial exits consume it pro rata, preventing each residual fill
    from being charged as though it opened through a fresh entry order.
    """

    direction: Direction
    quantity: int
    entry_price: float
    entry_monotonic_ms: float
    entry_exchange_ms: int
    entry_quote: FillQuote
    stop_price: float
    target_price: float
    remaining_entry_cost_rupees: float = 0.0

    @property
    def is_open(self) -> bool:
        """``True`` when the position has a direction and a size."""
        return self.direction is not Direction.FLAT and self.quantity > 0


@dataclass(frozen=True, slots=True)
class PnLReport:
    """Mark-to-market of an open or just-closed position."""

    direction: Direction
    quantity: int
    entry_price: float
    exit_price: float
    gross_ticks: float
    gross_rupees: float
    cost_rupees: float
    net_rupees: float
    net_bps: float
    holding_ms: float


# --------------------------------------------------------------------------- #
# Engine output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StateTransition:
    """A single state-machine transition."""

    previous: TradeState
    current: TradeState
    reasons: tuple[str, ...]

    @property
    def changed(self) -> bool:
        """``True`` when the state actually changed."""
        return self.previous is not self.current


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """Everything the engine produced for one accepted snapshot.

    This is the single object handed to the renderer, the logger and any
    downstream consumer, so that presentation code never reaches back into
    engine internals.
    """

    symbol: str
    token: str
    exchange_timestamp_ms: int
    wall_ms: float
    snapshot: Snapshot
    features: FeatureMap
    composite: CompositeResult
    regime: RegimeResult
    threshold: ThresholdResult
    quality: QualityReport
    state: TradeState
    transition: StateTransition
    position: Position | None
    pnl: PnLReport | None
    entry_quote: FillQuote | None
    exit_quote: FillQuote | None
    reasons: tuple[str, ...]
    compute_us: float
    snapshot_index: int


@dataclass(frozen=True, slots=True)
class EngineStats:
    """Counters exposed for monitoring. Never used for decisions."""

    accepted: int = 0
    rejected: int = 0
    blocked: int = 0
    gaps: int = 0
    resets: int = 0
    rejects_by_reason: Mapping[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Boundary protocols (dependency injection seams)
# --------------------------------------------------------------------------- #


@runtime_checkable
class MarketDataSource(Protocol):
    """A source of :class:`RawSnapshot` records.

    Implemented by the live Angel One adapter and by the offline replay
    adapter. The engine depends on this protocol only, which is what makes the
    whole pipeline testable without a websocket.
    """

    def start(self) -> None:
        """Begin producing snapshots."""

    def stop(self) -> None:
        """Stop producing snapshots and release resources."""

    def snapshots(self) -> Iterable[RawSnapshot]:
        """Yield snapshots until the source is exhausted or stopped."""


@runtime_checkable
class Renderer(Protocol):
    """A presentation surface for :class:`EngineOutput`."""

    def start(self) -> None:
        """Prepare the surface (clear screen, print header)."""

    def render(self, outputs: Iterable[EngineOutput], footer: str = "") -> None:
        """Render the latest output for each symbol, plus an optional footer."""

    def stop(self) -> None:
        """Restore the surface."""


__all__ = [
    "DEFAULT_TICK_SIZE",
    "EMPTY_LEVEL",
    "BlockReason",
    "CompositeResult",
    "CostBreakdown",
    "DepthLevel",
    "Direction",
    "EngineOutput",
    "EngineStats",
    "FeatureContribution",
    "FeatureKind",
    "FeatureMap",
    "FeatureValue",
    "FillQuote",
    "MarketDataSource",
    "PnLReport",
    "Position",
    "QualityReport",
    "RawSnapshot",
    "Regime",
    "RegimeResult",
    "RejectReason",
    "Renderer",
    "Side",
    "Snapshot",
    "StateTransition",
    "ThresholdResult",
    "TradeState",
    "ValidationResult",
]
