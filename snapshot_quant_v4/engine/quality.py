"""Market-quality gate.

The validator answers "is this snapshot structurally sound?". The quality gate
answers a different question: "is the market currently in a condition where a
*new* signal derived from this book would mean anything?".

The distinction matters operationally. A four-tick spread with one third of the
usual depth is a perfectly valid book; it is simply not a book in which a
microstructure edge of a few ticks survives. Such a snapshot must still update
every rolling statistic (otherwise the estimators develop holes) but must not
open a new position.

Two of the thresholds are deliberately *relative*:

*   Depth is compared with the instrument's own rolling geometric-mean depth. An
    absolute quantity floor cannot be shared by instruments whose typical top
    of book differs by three orders of magnitude.
*   The spread limit is in ticks, which is the unit in which the strategy's edge
    is also expressed, so the comparison is dimensionally meaningful.

The gate also produces two continuous scores, ``book_quality`` and
``liquidity_score``, which the confidence model consumes. A binary gate alone
would throw away the information that conditions are *deteriorating* rather than
already unacceptable.
"""

from __future__ import annotations

from ..config import QualityConfig
from ..utils.math_utils import linear_scale
from ..utils.types import BlockReason, QualityReport, Side, Snapshot
from .stats import SharedStatistics

#: Reported when nothing blocks trading, so the common path allocates no tuple.
_NO_REASONS: tuple[BlockReason, ...] = ()


class MarketQualityFilter:
    """Evaluates tradability of the current market state.

    Parameters
    ----------
    config:
        Quality thresholds.
    """

    __slots__ = ("_config",)

    def __init__(self, config: QualityConfig) -> None:
        self._config = config

    def evaluate(
        self,
        snapshot: Snapshot,
        stats: SharedStatistics,
        *,
        ms_since_gap: float | None,
    ) -> QualityReport:
        """Assess whether new signals may be generated from ``snapshot``.

        Parameters
        ----------
        snapshot:
            The validated snapshot.
        stats:
            The instrument's rolling statistics, already updated for this
            snapshot.
        ms_since_gap:
            Milliseconds since the last detected feed gap, or ``None`` if no gap
            has occurred in this session.

        Returns
        -------
        QualityReport
            ``tradable`` is ``True`` only when no blocking reason applies.
        """
        config = self._config
        reasons: list[BlockReason] = []

        if not stats.warm:
            reasons.append(BlockReason.WARMUP)

        spread_ticks = snapshot.spread_ticks
        spread_bps, spread_limit_ticks = self._spread_metrics(snapshot)
        if spread_ticks > spread_limit_ticks:
            reasons.append(BlockReason.SPREAD_TOO_WIDE)

        bid_levels = self._usable_levels(snapshot, Side.BID)
        ask_levels = self._usable_levels(snapshot, Side.ASK)
        if min(bid_levels, ask_levels) < config.min_depth_levels:
            reasons.append(BlockReason.INSUFFICIENT_DEPTH_LEVELS)

        top_quantity = min(snapshot.best_bid.quantity, snapshot.best_ask.quantity)
        if config.min_top_quantity > 0 and top_quantity < config.min_top_quantity:
            reasons.append(BlockReason.BOOK_TOO_THIN)

        relative_depth = stats.relative_depth
        if stats.depth_reference_ready and relative_depth < config.min_relative_depth:
            reasons.append(BlockReason.LIQUIDITY_BELOW_THRESHOLD)

        if ms_since_gap is not None and ms_since_gap < config.gap_block_ms:
            reasons.append(BlockReason.FEED_GAP)

        liquidity_score = self._liquidity_score(relative_depth)
        book_quality = self._book_quality(
            spread_ticks=spread_ticks,
            spread_limit_ticks=spread_limit_ticks,
            usable_levels=min(bid_levels, ask_levels),
            liquidity_score=liquidity_score,
            stats=stats,
        )
        return QualityReport(
            tradable=not reasons,
            reasons=tuple(reasons) if reasons else _NO_REASONS,
            book_quality=book_quality,
            liquidity_score=liquidity_score,
            detail=self._describe(
                reasons,
                spread_ticks,
                spread_bps,
                spread_limit_ticks,
                relative_depth,
            ),
            spread_bps=spread_bps,
            spread_limit_ticks=spread_limit_ticks,
        )

    # -- components -------------------------------------------------------- #

    def _spread_metrics(self, snapshot: Snapshot) -> tuple[float, float]:
        """Return the observed spread in bps and the configured tick limit."""
        spread_bps = snapshot.spread * 10_000.0 / snapshot.mid
        return spread_bps, self._config.max_signal_spread_ticks

    def _usable_levels(self, snapshot: Snapshot, side: Side) -> int:
        """Count levels carrying both quantity and the required order count.

        A level with quantity but a zero order count is treated as unusable
        because the field is then either missing or stale, and several features
        divide by it.
        """
        minimum_orders = self._config.min_orders_per_level
        usable = 0
        for level in snapshot.levels(side):
            if level.quantity <= 0 or level.price_paise <= 0:
                break
            if level.orders < minimum_orders:
                break
            usable += 1
        return usable

    def _liquidity_score(self, relative_depth: float) -> float:
        """Map relative depth onto ``[0, 1]``.

        The lower anchor is half the blocking threshold so that the score
        reaches zero measurably before the hard block engages; the upper anchor
        is the instrument's own typical depth.
        """
        floor = self._config.min_relative_depth * 0.5
        return linear_scale(relative_depth, floor, 1.0)

    def _book_quality(
        self,
        *,
        spread_ticks: float,
        spread_limit_ticks: float,
        usable_levels: int,
        liquidity_score: float,
        stats: SharedStatistics,
    ) -> float:
        """Continuous book-quality score in ``[0, 1]``.

        Combines spread tightness, ladder completeness, liquidity and warmup
        progress as a product, so that a single collapsed component is enough to
        drive the score down. A weighted sum would allow one severe defect to be
        masked by three healthy components.
        """
        config = self._config
        spread_component = linear_scale(
            -spread_ticks, -spread_limit_ticks, -1.0
        )
        level_component = linear_scale(
            float(usable_levels), float(config.min_depth_levels) - 1.0, 5.0
        )
        warmup_component = stats.warmup_ratio
        quality = spread_component * level_component * liquidity_score * warmup_component
        if quality < 0.0:
            return 0.0
        if quality > 1.0:
            return 1.0
        return quality

    def _describe(
        self,
        reasons: list[BlockReason],
        spread_ticks: float,
        spread_bps: float,
        spread_limit_ticks: float,
        relative_depth: float,
    ) -> str:
        """Build a short operator-facing explanation."""
        if not reasons:
            return ""
        return (
            f"blocked={'+'.join(reason.value for reason in reasons)} "
            f"spread={spread_ticks:.2f}t/{spread_bps:.2f}bps "
            f"limit={spread_limit_ticks:.2f}t depth={relative_depth:.2f}x"
        )


__all__ = ["MarketQualityFilter"]
