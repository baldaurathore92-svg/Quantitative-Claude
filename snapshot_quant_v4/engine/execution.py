"""Execution and cost model.

Separation of concerns
----------------------
This module is intentionally isolated from the signal path. The features, the
composite score, the regime detector and the threshold policy never import it and
never consult it. That separation exists so that research, backtesting and live
execution can evolve independently: changing the fill assumption or the broker
schedule cannot alter a single feature value or a single signal.

The only optional coupling is :meth:`ExecutionModel.clears_cost`, which the state
machine may consult before an entry when ``execution.enforce_cost_gate`` is
enabled. It defaults to disabled.

Prices are modelled, not decorated
----------------------------------
Three things that a naive model gets wrong are handled explicitly.

**The tick grid is real.** A basis-point offset is not a tradable price. Three
basis points is 3 ticks on a 500 rupee instrument and 18 ticks on a 3000 rupee
instrument, so the offset is converted to ticks against the live reference price
and then rounded onto the grid. The result is additionally constrained never to
be *better* than the touch: an aggressive order cannot fill at a price improvement
it did not ask for.

**Size matters.** With ``use_depth_walk`` enabled the model consumes the visible
ladder and returns a size-weighted average price, reporting a partial fill when
the five published levels cannot absorb the request. A model that fills any size
at the touch flatters every result derived from it.

**Costs are asymmetric and mostly statutory.** Securities transaction tax applies
to the sell leg, stamp duty to the buy leg, and GST applies to brokerage and
exchange charges but not to statutory taxes. All rates are configurable because
they differ per broker and change with regulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import CostConfig, ExecutionConfig, StateMachineConfig
from ..utils.math_utils import (
    bps_to_ticks,
    clamp,
    round_to_tick,
    safe_div,
    ticks_to_bps,
)
from ..utils.types import (
    CostBreakdown,
    DepthLevel,
    Direction,
    FillQuote,
    PnLReport,
    Position,
    Side,
    Snapshot,
)

#: Price comparison tolerance, in rupees. Orders of magnitude below one paise, and
#: orders of magnitude above double-precision representation error.
_PRICE_EPSILON: float = 1e-7


@dataclass(frozen=True, slots=True)
class CostQuote:
    """Round-trip cost for a specific notional, in both rupees and basis points."""

    breakdown: CostBreakdown
    rupees: float
    bps: float
    ticks: float


class CostModel:
    """Round-trip transaction cost for Indian intraday equity.

    Parameters
    ----------
    config:
        Rates and flat charges. When ``config.enabled`` is ``False`` every cost
        is reported as zero, which is the correct behaviour for pure signal
        research.
    """

    __slots__ = ("_config",)

    def __init__(self, config: CostConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        """Whether costs are being modelled at all."""
        return self._config.enabled

    def leg_cost(self, *, price: float, quantity: int, buying: bool) -> float:
        """Cost in rupees for one buy or sell order leg.

        This is used to allocate one original entry order across partial exits
        while charging each exit fill as its own order. Summing one buy and one
        sell leg is exactly equivalent to :meth:`round_trip`.
        """
        config = self._config
        if not config.enabled or quantity <= 0:
            return 0.0
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        exchange = config.exchange_txn_rate * turnover
        sebi = config.sebi_rate * turnover
        statutory = (
            config.stamp_duty_buy_rate * turnover
            if buying
            else config.stt_sell_rate * turnover
        )
        gst = config.gst_rate * (brokerage + exchange + sebi)
        return brokerage + exchange + sebi + statutory + gst

    def round_trip(
        self,
        *,
        buy_price: float,
        sell_price: float,
        quantity: int,
        tick_size: float,
    ) -> CostQuote:
        """Cost of one buy leg plus one sell leg.

        The basis-point figure is expressed against the mean of the two
        notionals, which is the convention that makes it directly comparable with
        an expected move quoted in basis points.
        """
        config = self._config
        if not config.enabled or quantity <= 0:
            zero = CostBreakdown(0.0, 0.0, 0.0, 0.0)
            return CostQuote(breakdown=zero, rupees=0.0, bps=0.0, ticks=0.0)

        buy_turnover = buy_price * quantity
        sell_turnover = sell_price * quantity
        turnover = buy_turnover + sell_turnover

        brokerage = self._brokerage(buy_turnover) + self._brokerage(sell_turnover)
        exchange = config.exchange_txn_rate * turnover
        sebi = config.sebi_rate * turnover
        statutory = (
            config.stt_sell_rate * sell_turnover
            + config.stamp_duty_buy_rate * buy_turnover
        )
        gst = config.gst_rate * (brokerage + exchange + sebi)

        reference = (buy_turnover + sell_turnover) * 0.5
        rupees = self.leg_cost(
            price=buy_price,
            quantity=quantity,
            buying=True,
        ) + self.leg_cost(
            price=sell_price,
            quantity=quantity,
            buying=False,
        )
        breakdown = CostBreakdown(
            brokerage_bps=self._as_bps(brokerage + gst, reference),
            exchange_bps=self._as_bps(exchange + sebi, reference),
            statutory_bps=self._as_bps(statutory, reference),
            slippage_bps=0.0,
        )
        # ``ticks`` answers the question the strategy actually asks: how many
        # ticks must price move in our favour before the round trip breaks even.
        return CostQuote(
            breakdown=breakdown,
            rupees=rupees,
            bps=self._as_bps(rupees, reference),
            ticks=safe_div(rupees, quantity * tick_size),
        )

    def _brokerage(self, turnover: float) -> float:
        """Per-order brokerage: the lower of the flat fee and the rate."""
        config = self._config
        rate_based = config.brokerage_rate * turnover
        candidates = [rate_based]
        if config.brokerage_flat_per_order > 0.0:
            candidates.append(config.brokerage_flat_per_order)
        if config.brokerage_cap_per_order > 0.0:
            candidates.append(config.brokerage_cap_per_order)
        return min(candidates)

    @staticmethod
    def _as_bps(rupees: float, reference_notional: float) -> float:
        """Convert a rupee charge into basis points of a reference notional."""
        return safe_div(rupees * 1e4, reference_notional)


class ExecutionModel:
    """Models entry and exit prices, positions and mark-to-market.

    Parameters
    ----------
    config:
        Execution parameters.
    state_machine:
        State-machine parameters, used only for the stop and target distances so
        that the position record carries its own exit levels.
    """

    __slots__ = ("_config", "_costs", "_state_machine")

    def __init__(
        self, config: ExecutionConfig, state_machine: StateMachineConfig
    ) -> None:
        self._config = config
        self._state_machine = state_machine
        self._costs = CostModel(config.cost)

    @property
    def costs(self) -> CostModel:
        """The cost model, exposed for reporting and for the optional gate."""
        return self._costs

    # -- quotes ------------------------------------------------------------ #

    def entry_quote(
        self, snapshot: Snapshot, direction: Direction, quantity: int
    ) -> FillQuote:
        """Price at which an entry in ``direction`` is assumed to fill.

        A long entry crosses the spread and pays the ask plus the configured
        aggression; a short entry sells the bid minus the aggression.
        """
        if direction is Direction.LONG:
            return self._aggressive_quote(snapshot, Side.ASK, quantity, buying=True)
        if direction is Direction.SHORT:
            return self._aggressive_quote(snapshot, Side.BID, quantity, buying=False)
        raise ValueError("entry_quote requires a directional position")

    def exit_quote(
        self, snapshot: Snapshot, direction: Direction, quantity: int
    ) -> FillQuote:
        """Price at which an open position in ``direction`` is assumed to close.

        Closing a long means selling into the bid; closing a short means buying
        the ask. Slippage is applied in the adverse direction in both cases.
        """
        if direction is Direction.LONG:
            return self._passive_exit_quote(snapshot, Side.BID, quantity, buying=False)
        if direction is Direction.SHORT:
            return self._passive_exit_quote(snapshot, Side.ASK, quantity, buying=True)
        raise ValueError("exit_quote requires a directional position")

    def _aggressive_quote(
        self, snapshot: Snapshot, side: Side, quantity: int, *, buying: bool
    ) -> FillQuote:
        """Build an entry quote against ``side`` of the book."""
        aggression_ticks = self._entry_aggression_ticks(snapshot)
        return self._quote(
            snapshot, side, quantity, offset_ticks=aggression_ticks, buying=buying
        )

    def _passive_exit_quote(
        self, snapshot: Snapshot, side: Side, quantity: int, *, buying: bool
    ) -> FillQuote:
        """Build an exit quote against ``side`` of the book."""
        slippage_ticks = self._exit_slippage_ticks(snapshot)
        return self._quote(
            snapshot, side, quantity, offset_ticks=slippage_ticks, buying=buying
        )

    def _quote(
        self,
        snapshot: Snapshot,
        side: Side,
        quantity: int,
        *,
        offset_ticks: float,
        buying: bool,
    ) -> FillQuote:
        """Common quote construction for entries and exits."""
        levels = snapshot.levels(side)
        if not levels or quantity <= 0:
            raise ValueError("cannot quote against an empty ladder or zero quantity")
        touch = levels[0]

        if self._config.use_depth_walk:
            base_price, filled, consumed = self._walk(levels, quantity)
        else:
            base_price, filled, consumed = touch.price, quantity, 1

        tick = snapshot.tick_size
        offset = offset_ticks * tick
        raw_price = base_price + offset if buying else base_price - offset
        price = round_to_tick(raw_price, tick)
        # An aggressive order can never fill better than the touch it crossed.
        price = max(price, touch.price) if buying else min(price, touch.price)

        return FillQuote(
            price=price,
            reference_price=touch.price,
            aggression_ticks=offset_ticks,
            requested_quantity=quantity,
            filled_quantity=filled,
            levels_consumed=consumed,
            complete=filled >= quantity,
        )

    def _walk(
        self, levels: tuple[DepthLevel, ...], quantity: int
    ) -> tuple[float, int, int]:
        """Size-weighted average price across the visible ladder.

        Returns the VWAP, the quantity actually absorbed and the number of levels
        consumed. Bounded by the five published levels, so this is O(1).
        """
        remaining = quantity
        notional = 0.0
        consumed = 0
        for level in levels:
            if remaining <= 0:
                break
            if level.quantity <= 0:
                break
            take = level.quantity if level.quantity < remaining else remaining
            notional += take * level.price
            remaining -= take
            consumed += 1
        filled = quantity - remaining
        if filled <= 0:
            return levels[0].price, 0, 0
        return notional / filled, filled, consumed

    def _entry_aggression_ticks(self, snapshot: Snapshot) -> float:
        """Entry aggression in ticks, from either the tick or the bps setting."""
        config = self._config
        if config.entry_aggression_ticks is not None:
            return config.entry_aggression_ticks
        return bps_to_ticks(
            config.entry_aggression_bps, snapshot.mid, snapshot.tick_size
        )

    def _exit_slippage_ticks(self, snapshot: Snapshot) -> float:
        """Exit slippage in ticks, from either the tick or the bps setting."""
        config = self._config
        if config.exit_slippage_ticks is not None:
            return config.exit_slippage_ticks
        return bps_to_ticks(config.exit_slippage_bps, snapshot.mid, snapshot.tick_size)

    # -- positions --------------------------------------------------------- #

    def open_position(
        self,
        snapshot: Snapshot,
        direction: Direction,
        quantity: int,
        *,
        monotonic_ms: float,
    ) -> Position:
        """Create a position record with its stop and target already resolved.

        Stop and target distances are configured in ticks and converted here, so
        the levels are on the price grid and are stable for the life of the
        position rather than being recomputed from a moving reference.
        """
        quote = self.entry_quote(snapshot, direction, quantity)
        if quote.filled_quantity <= 0:
            raise ValueError("entry order has no executable visible depth")
        if not quote.complete and not self._config.allow_partial_fill:
            raise ValueError(
                "partial entry fill is disabled: "
                f"requested {quote.requested_quantity}, visible {quote.filled_quantity}"
            )
        tick = snapshot.tick_size
        stop_distance = self._state_machine.stop_ticks * tick
        target_distance = self._state_machine.target_ticks * tick
        signum = direction.signum
        stop_price = round_to_tick(quote.price - signum * stop_distance, tick)
        target_price = round_to_tick(quote.price + signum * target_distance, tick)
        entry_cost = self._costs.leg_cost(
            price=quote.price,
            quantity=quote.filled_quantity,
            buying=direction is Direction.LONG,
        )
        return Position(
            direction=direction,
            quantity=quote.filled_quantity,
            entry_price=quote.price,
            entry_monotonic_ms=monotonic_ms,
            entry_exchange_ms=snapshot.exchange_timestamp_ms,
            entry_quote=quote,
            stop_price=stop_price,
            target_price=target_price,
            remaining_entry_cost_rupees=entry_cost,
        )

    def mark_to_market(
        self, position: Position, snapshot: Snapshot, *, monotonic_ms: float
    ) -> tuple[FillQuote, PnLReport]:
        """Value an open position at the current book.

        Returns both the exit quote and the profit and loss report, so callers do
        not have to recompute the quote to display the exit price. When visible
        depth is insufficient, the report is explicitly limited to the quote's
        executable quantity; it never applies a partial quote to the full
        position.
        """
        quote = self.exit_quote(snapshot, position.direction, position.quantity)
        quantity = quote.filled_quantity
        signum = position.direction.signum
        gross_rupees = (quote.price - position.entry_price) * signum * quantity
        gross_ticks = (
            safe_div(
                (quote.price - position.entry_price) * signum,
                snapshot.tick_size,
            )
            if quantity > 0
            else 0.0
        )

        entry_cost = self._allocated_entry_cost(position, quantity)
        exit_cost = self._costs.leg_cost(
            price=quote.price,
            quantity=quantity,
            buying=position.direction is Direction.SHORT,
        )
        cost_rupees = entry_cost + exit_cost
        net_rupees = gross_rupees - cost_rupees
        reference_notional = position.entry_price * quantity
        report = PnLReport(
            direction=position.direction,
            quantity=quantity,
            entry_price=position.entry_price,
            exit_price=quote.price,
            gross_ticks=gross_ticks,
            gross_rupees=gross_rupees,
            cost_rupees=cost_rupees,
            net_rupees=net_rupees,
            net_bps=safe_div(net_rupees * 1e4, reference_notional),
            holding_ms=monotonic_ms - position.entry_monotonic_ms,
        )
        return quote, report

    def residual_position(
        self,
        position: Position,
        filled_quantity: int,
    ) -> Position:
        """Return the unfilled residual with entry cost allocated pro rata."""
        if not 0 <= filled_quantity < position.quantity:
            raise ValueError(
                "filled_quantity must be non-negative and smaller than position size"
            )
        allocated = self._allocated_entry_cost(position, filled_quantity)
        return replace(
            position,
            quantity=position.quantity - filled_quantity,
            remaining_entry_cost_rupees=max(
                0.0,
                position.remaining_entry_cost_rupees - allocated,
            ),
        )

    @staticmethod
    def _allocated_entry_cost(position: Position, filled_quantity: int) -> float:
        """Allocate the original entry order cost to one exit fill."""
        if filled_quantity <= 0 or position.quantity <= 0:
            return 0.0
        if filled_quantity >= position.quantity:
            return position.remaining_entry_cost_rupees
        return (
            position.remaining_entry_cost_rupees
            * filled_quantity
            / position.quantity
        )

    # -- optional cost gate ------------------------------------------------ #

    def entry_is_fillable(
        self,
        snapshot: Snapshot,
        direction: Direction,
        quantity: int,
    ) -> tuple[bool, str]:
        """Whether visible directional depth satisfies the entry-fill policy.

        Touch-fill models always report complete execution. With depth walking,
        an incomplete quote is allowed only when ``allow_partial_fill`` is true.
        Keeping this separate from :meth:`clears_cost` preserves that method's
        established two-argument override contract.
        """
        config = self._config
        if not config.use_depth_walk or config.allow_partial_fill:
            return True, ""
        quote = self.entry_quote(snapshot, direction, quantity)
        if quote.complete:
            return True, ""
        return False, (
            "insufficient visible entry depth: "
            f"requested {quote.requested_quantity}, available {quote.filled_quantity}"
        )

    def clears_cost(self, snapshot: Snapshot, quantity: int) -> tuple[bool, str]:
        """Whether the configured target clears the round-trip cost.

        The comparison is between the configured target distance and the modelled
        round-trip cost, both converted to basis points of the entry notional, with
        a required safety multiple. A strategy whose target does not clear its own
        costs by a margin is blocked at the point of entry.

        Disabled by default; when disabled the method reports success without
        consulting the cost model at all.
        """
        config = self._config
        if not config.enforce_cost_gate or not self._costs.enabled:
            return True, ""
        target_bps = ticks_to_bps(
            self._state_machine.target_ticks, snapshot.mid, snapshot.tick_size
        )
        spread_bps = ticks_to_bps(
            snapshot.spread_ticks, snapshot.mid, snapshot.tick_size
        )
        cost = self._costs.round_trip(
            buy_price=snapshot.best_ask.price,
            sell_price=snapshot.best_bid.price,
            quantity=quantity,
            tick_size=snapshot.tick_size,
        )
        required = (cost.bps + spread_bps) * config.min_edge_multiple
        if target_bps >= required:
            return True, ""
        return False, (
            f"target {target_bps:.1f}bps below required {required:.1f}bps "
            f"(cost {cost.bps:.1f} + spread {spread_bps:.1f}, "
            f"x{config.min_edge_multiple:g})"
        )

    def stop_hit(self, position: Position, snapshot: Snapshot) -> bool:
        """Whether the stop level has been reached on the exit side of the book.

        The test uses the price the position would actually exit at — the bid for
        a long, the ask for a short — rather than the mid. Using the mid reports
        stops that a real order would not have triggered, and misses stops that a
        real order would have hit.

        Comparisons carry :data:`_PRICE_EPSILON` so that a level sitting exactly
        on a tick is treated as reached. Two doubles representing the same
        exchange price can differ in the last bits depending on how they were
        derived, and without the tolerance a stop precisely at the touch would
        occasionally be skipped.
        """
        if position.direction is Direction.LONG:
            return snapshot.best_bid.price <= position.stop_price + _PRICE_EPSILON
        if position.direction is Direction.SHORT:
            return snapshot.best_ask.price >= position.stop_price - _PRICE_EPSILON
        return False

    def target_hit(self, position: Position, snapshot: Snapshot) -> bool:
        """Whether the target level has been reached on the exit side."""
        if position.direction is Direction.LONG:
            return snapshot.best_bid.price >= position.target_price - _PRICE_EPSILON
        if position.direction is Direction.SHORT:
            return snapshot.best_ask.price <= position.target_price + _PRICE_EPSILON
        return False

    def unrealised_ticks(self, position: Position, snapshot: Snapshot) -> float:
        """Signed unrealised move in ticks, using the exit side of the book."""
        if position.direction is Direction.LONG:
            current = snapshot.best_bid.price
        elif position.direction is Direction.SHORT:
            current = snapshot.best_ask.price
        else:
            return 0.0
        return safe_div(
            (current - position.entry_price) * position.direction.signum,
            snapshot.tick_size,
        )

    @staticmethod
    def clamp_confidence(value: float) -> float:
        """Clamp a confidence-like value into ``[0, 1]``.

        Exposed here so that reporting code does not have to import the maths
        helpers directly.
        """
        return clamp(value, 0.0, 1.0)


__all__ = ["CostModel", "CostQuote", "ExecutionModel"]
