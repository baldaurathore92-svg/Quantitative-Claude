"""Structural validation of raw snapshots.

The validator is the only component allowed to turn a :class:`RawSnapshot` into
a :class:`Snapshot`. Everything downstream may therefore assume:

*   both sides have a level one with a positive price and quantity;
*   the book is not crossed (and, unless explicitly allowed, not locked);
*   the depth ladders are monotone in price;
*   the spread is inside a sane bound;
*   the exchange timestamp did not move backwards;
*   the cumulative traded volume did not decrease.

Derived quantities (mid, spread in ticks, ``dt_ms``, ``delta_volume``) are
computed exactly once, here.

**Gap handling.** A feed gap — a long silence, a backwards timestamp, a
sequence discontinuity or a large mid jump — invalidates every incremental
statistic that spans it. The validator does not merely report the gap: it is
the trigger for the engine to reset its estimators and return to ``WARMUP``.
Continuing to feed EMAs and sliding variances across a gap is silently wrong,
and is a defect that is almost impossible to spot in live output.

One deliberate choice deserves explanation: a snapshot whose *content* is
identical to the previous one is rejected as a duplicate. SmartAPI can repeat a
payload, and counting it twice would bias every time-aware estimator towards a
frozen market.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ValidationConfig
from ..utils.constants import SNAPQUOTE_DEPTH_LEVELS
from ..utils.logging_utils import get_logger
from ..utils.math_utils import is_tick_aligned, to_ticks
from ..utils.types import (
    DepthLevel,
    RawSnapshot,
    RejectReason,
    Snapshot,
    ValidationResult,
)

_LOGGER = get_logger(__name__)


@dataclass(slots=True)
class _PreviousState:
    """The minimum history required to validate the next snapshot."""

    exchange_timestamp_ms: int
    sequence_number: int
    volume_traded_today: int
    mid: float
    best_bid_paise: int
    best_ask_paise: int
    best_bid_qty: int
    best_ask_qty: int
    last_traded_price: float


def _reject(reason: RejectReason, detail: str, *, gap: bool = False) -> ValidationResult:
    """Build a rejection result."""
    return ValidationResult(
        accepted=False, reason=reason, snapshot=None, detail=detail, gap_detected=gap
    )


class SnapshotValidator:
    """Validates and enriches snapshots for a single instrument.

    One instance per token: the validator holds the previous snapshot's summary,
    which is per-instrument state by definition.

    Parameters
    ----------
    symbol:
        Display name, used in log messages only.
    tick_size:
        Instrument tick size in rupees.
    config:
        Validation thresholds.
    """

    __slots__ = ("_accepted", "_config", "_previous", "_symbol", "_tick_size")

    def __init__(self, symbol: str, tick_size: float, config: ValidationConfig) -> None:
        if tick_size <= 0.0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self._symbol = symbol
        self._tick_size = float(tick_size)
        self._config = config
        self._previous: _PreviousState | None = None
        self._accepted = 0

    # -- state ------------------------------------------------------------- #

    @property
    def accepted_count(self) -> int:
        """Number of snapshots accepted since the last reset."""
        return self._accepted

    @property
    def has_history(self) -> bool:
        """``True`` once at least one snapshot has been accepted."""
        return self._previous is not None

    def reset(self) -> None:
        """Forget history. Called after a gap or a websocket reconnect."""
        self._previous = None

    # -- validation -------------------------------------------------------- #

    def validate(self, raw: RawSnapshot) -> ValidationResult:
        """Validate ``raw`` and return an enriched snapshot on success.

        The checks are ordered cheapest-first and by severity, so that a
        malformed book short-circuits before any arithmetic is attempted.
        """
        config = self._config

        if not raw.bids or not raw.asks:
            return _reject(
                RejectReason.MISSING_L1,
                f"{self._symbol}: empty ladder (bids={len(raw.bids)} asks={len(raw.asks)})",
            )
        if len(raw.bids) > SNAPQUOTE_DEPTH_LEVELS or len(raw.asks) > SNAPQUOTE_DEPTH_LEVELS:
            return _reject(
                RejectReason.MALFORMED_PAYLOAD,
                f"{self._symbol}: more than {SNAPQUOTE_DEPTH_LEVELS} levels per side",
            )

        best_bid = raw.bids[0]
        best_ask = raw.asks[0]

        if best_bid.quantity == 0:
            return _reject(RejectReason.ZERO_BID, f"{self._symbol}: zero bid quantity")
        if best_ask.quantity == 0:
            return _reject(RejectReason.ZERO_ASK, f"{self._symbol}: zero ask quantity")

        ladder_error = self._validate_ladders(raw)
        if ladder_error is not None:
            return ladder_error

        if best_bid.price < config.min_price or best_ask.price < config.min_price:
            return _reject(
                RejectReason.INVALID_PRICE,
                f"{self._symbol}: price below min_price "
                f"(bid={best_bid.price} ask={best_ask.price})",
            )
        if config.require_tick_alignment and not (
            is_tick_aligned(best_bid.price, self._tick_size)
            and is_tick_aligned(best_ask.price, self._tick_size)
        ):
            return _reject(
                RejectReason.TICK_MISALIGNED,
                f"{self._symbol}: touch prices are not on the "
                f"{self._tick_size} grid (bid={best_bid.price} ask={best_ask.price})",
            )

        if best_ask.price_paise < best_bid.price_paise:
            return _reject(
                RejectReason.CROSSED_BOOK,
                f"{self._symbol}: crossed book bid={best_bid.price} ask={best_ask.price}",
            )
        if best_ask.price_paise == best_bid.price_paise and not config.allow_locked_book:
            return _reject(
                RejectReason.LOCKED_BOOK,
                f"{self._symbol}: locked book at {best_bid.price}",
            )

        if config.require_ltp and raw.last_traded_price <= 0.0:
            return _reject(
                RejectReason.INVALID_LTP,
                f"{self._symbol}: non-positive last traded price "
                f"({raw.last_traded_price})",
            )

        spread = best_ask.price - best_bid.price
        spread_ticks = to_ticks(spread, self._tick_size)
        if spread_ticks > config.max_spread_ticks:
            return _reject(
                RejectReason.SPREAD_TOO_WIDE,
                f"{self._symbol}: spread {spread_ticks:.2f} ticks exceeds "
                f"{config.max_spread_ticks}",
            )

        mid = (best_bid.price + best_ask.price) * 0.5
        previous = self._previous

        if previous is None:
            return self._accept(
                raw, best_bid, best_ask, mid, spread, spread_ticks, None, gap=False
            )

        temporal = self._validate_temporal(raw, previous)
        if temporal is not None:
            return temporal

        dt_ms = float(raw.exchange_timestamp_ms - previous.exchange_timestamp_ms)
        gap = self._detect_gap(raw, previous, mid, dt_ms)

        if config.reject_duplicates and self._is_duplicate(raw, previous, dt_ms):
            return _reject(
                RejectReason.DUPLICATE_SNAPSHOT,
                f"{self._symbol}: identical payload repeated at "
                f"{raw.exchange_timestamp_ms}",
            )

        return self._accept(
            raw, best_bid, best_ask, mid, spread, spread_ticks, previous, gap=gap
        )

    # -- helpers ----------------------------------------------------------- #

    def _validate_ladders(self, raw: RawSnapshot) -> ValidationResult | None:
        """Check per-level sanity and price monotonicity on both sides."""
        previous_paise = None
        for index, level in enumerate(raw.bids):
            error = self._validate_level(level, "bid", index)
            if error is not None:
                return error
            if previous_paise is not None and level.price_paise >= previous_paise:
                return _reject(
                    RejectReason.UNSORTED_DEPTH,
                    f"{self._symbol}: bid ladder not strictly descending at level "
                    f"{index + 1}",
                )
            previous_paise = level.price_paise

        previous_paise = None
        for index, level in enumerate(raw.asks):
            error = self._validate_level(level, "ask", index)
            if error is not None:
                return error
            if previous_paise is not None and level.price_paise <= previous_paise:
                return _reject(
                    RejectReason.UNSORTED_DEPTH,
                    f"{self._symbol}: ask ladder not strictly ascending at level "
                    f"{index + 1}",
                )
            previous_paise = level.price_paise
        return None

    def _validate_level(
        self, level: DepthLevel, side: str, index: int
    ) -> ValidationResult | None:
        """Validate a single depth level."""
        if level.quantity < 0:
            return _reject(
                RejectReason.NEGATIVE_QUANTITY,
                f"{self._symbol}: negative {side} quantity {level.quantity} at level "
                f"{index + 1}",
            )
        if level.orders < 0:
            return _reject(
                RejectReason.NEGATIVE_ORDERS,
                f"{self._symbol}: negative {side} order count {level.orders} at level "
                f"{index + 1}",
            )
        if level.price_paise <= 0 and level.quantity > 0:
            return _reject(
                RejectReason.INVALID_PRICE,
                f"{self._symbol}: non-positive {side} price with quantity at level "
                f"{index + 1}",
            )
        return None

    def _validate_temporal(
        self, raw: RawSnapshot, previous: _PreviousState
    ) -> ValidationResult | None:
        """Check timestamp monotonicity, staleness and volume monotonicity."""
        config = self._config
        delta_ms = raw.exchange_timestamp_ms - previous.exchange_timestamp_ms
        if delta_ms < 0:
            # Out-of-order delivery. The snapshot is discarded and the gap flag
            # forces a state reset, because the ordering assumption that every
            # incremental estimator relies on has been violated.
            return _reject(
                RejectReason.NON_MONOTONIC_TIMESTAMP,
                f"{self._symbol}: exchange timestamp moved backwards by "
                f"{-delta_ms} ms",
                gap=True,
            )
        if delta_ms > config.max_staleness_ms:
            return _reject(
                RejectReason.STALE_TIMESTAMP,
                f"{self._symbol}: {delta_ms} ms since previous snapshot exceeds "
                f"max_staleness_ms={config.max_staleness_ms}",
                gap=True,
            )
        if raw.volume_traded_today < previous.volume_traded_today:
            # Cumulative day volume can only increase. A decrease means either a
            # feed restart or a different instrument on the same token.
            return _reject(
                RejectReason.VOLUME_REGRESSION,
                f"{self._symbol}: cumulative volume fell from "
                f"{previous.volume_traded_today} to {raw.volume_traded_today}",
                gap=True,
            )
        return None

    def _detect_gap(
        self, raw: RawSnapshot, previous: _PreviousState, mid: float, dt_ms: float
    ) -> bool:
        """Classify an accepted snapshot as continuous or post-gap."""
        config = self._config
        if dt_ms > config.max_snapshot_gap_ms:
            _LOGGER.warning(
                "%s: feed gap of %.0f ms exceeds %.0f ms; statistics will be reset",
                self._symbol,
                dt_ms,
                config.max_snapshot_gap_ms,
            )
            return True
        jump_ticks = abs(to_ticks(mid - previous.mid, self._tick_size))
        if jump_ticks > config.max_price_gap_ticks:
            _LOGGER.warning(
                "%s: mid jumped %.1f ticks (limit %.1f); statistics will be reset",
                self._symbol,
                jump_ticks,
                config.max_price_gap_ticks,
            )
            return True
        if (
            raw.sequence_number
            and previous.sequence_number
            and raw.sequence_number < previous.sequence_number
        ):
            _LOGGER.warning(
                "%s: sequence number regressed %d -> %d; statistics will be reset",
                self._symbol,
                previous.sequence_number,
                raw.sequence_number,
            )
            return True
        return False

    def _is_duplicate(
        self, raw: RawSnapshot, previous: _PreviousState, dt_ms: float
    ) -> bool:
        """Return ``True`` for a byte-identical repeat of the previous snapshot.

        The test is intentionally conservative: only a snapshot that changed
        *nothing* observable — timestamp, touch prices, touch quantities, traded
        volume and last price — counts as a duplicate.
        """
        if dt_ms > 0.0:
            return False
        return (
            raw.bids[0].price_paise == previous.best_bid_paise
            and raw.asks[0].price_paise == previous.best_ask_paise
            and raw.bids[0].quantity == previous.best_bid_qty
            and raw.asks[0].quantity == previous.best_ask_qty
            and raw.volume_traded_today == previous.volume_traded_today
            and raw.last_traded_price == previous.last_traded_price
        )

    def _accept(
        self,
        raw: RawSnapshot,
        best_bid: DepthLevel,
        best_ask: DepthLevel,
        mid: float,
        spread: float,
        spread_ticks: float,
        previous: _PreviousState | None,
        *,
        gap: bool,
    ) -> ValidationResult:
        """Build the enriched snapshot and update stored history."""
        if previous is None or gap:
            # After a gap, ``dt_ms`` and ``delta_volume`` measured across the
            # discontinuity are meaningless. The snapshot is treated as the first
            # of a new sequence so that no estimator consumes them.
            dt_ms = 0.0
            delta_volume = 0
            is_first = True
        else:
            dt_ms = float(raw.exchange_timestamp_ms - previous.exchange_timestamp_ms)
            delta_volume = raw.volume_traded_today - previous.volume_traded_today
            is_first = False

        snapshot = Snapshot(
            raw=raw,
            symbol=raw.symbol or self._symbol,
            token=raw.token,
            tick_size=self._tick_size,
            exchange_timestamp_ms=raw.exchange_timestamp_ms,
            received_monotonic_ms=raw.received_monotonic_ms,
            dt_ms=dt_ms,
            is_first=is_first,
            delta_volume=delta_volume,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            spread=spread,
            spread_ticks=spread_ticks,
            bids=raw.bids,
            asks=raw.asks,
        )
        self._previous = _PreviousState(
            exchange_timestamp_ms=raw.exchange_timestamp_ms,
            sequence_number=raw.sequence_number,
            volume_traded_today=raw.volume_traded_today,
            mid=mid,
            best_bid_paise=best_bid.price_paise,
            best_ask_paise=best_ask.price_paise,
            best_bid_qty=best_bid.quantity,
            best_ask_qty=best_ask.quantity,
            last_traded_price=raw.last_traded_price,
        )
        self._accepted += 1
        return ValidationResult(
            accepted=True,
            reason=RejectReason.NONE,
            snapshot=snapshot,
            detail="",
            gap_detected=gap,
        )


__all__ = ["SnapshotValidator"]
