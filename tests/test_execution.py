"""Execution and cost model tests."""

from __future__ import annotations

import pytest
from snapshot_quant_v4.config import CostConfig, ExecutionConfig, StateMachineConfig
from snapshot_quant_v4.engine.execution import CostModel, ExecutionModel
from snapshot_quant_v4.utils.types import Direction

from conftest import BASE_PAISE, TICK_PAISE, ladder, make_snapshot, validate_one

STATE = StateMachineConfig(stop_ticks=5.0, target_ticks=10.0)


def model(**kwargs) -> ExecutionModel:
    """Build an execution model with cost disabled unless requested."""
    kwargs.setdefault("cost", CostConfig(enabled=False))
    return ExecutionModel(ExecutionConfig(**kwargs), STATE)


class TestQuotes:
    def test_long_entry_pays_the_ask_plus_aggression(self) -> None:
        execution = model(entry_aggression_ticks=2.0)
        snapshot = validate_one(make_snapshot())
        quote = execution.entry_quote(snapshot, Direction.LONG, 1)
        assert quote.reference_price == pytest.approx(800.05)
        assert quote.price == pytest.approx(800.05 + 2 * 0.05)
        assert quote.complete

    def test_short_entry_sells_the_bid_minus_aggression(self) -> None:
        execution = model(entry_aggression_ticks=2.0)
        snapshot = validate_one(make_snapshot())
        quote = execution.entry_quote(snapshot, Direction.SHORT, 1)
        assert quote.price == pytest.approx(799.95 - 2 * 0.05)

    def test_basis_points_are_converted_against_the_live_price(self) -> None:
        # Three basis points is a different number of ticks at different prices,
        # which is exactly why the conversion is done here and not hardcoded.
        execution = model(entry_aggression_bps=3.0)
        cheap = validate_one(
            make_snapshot(bid_paise=15_000 - TICK_PAISE, ask_paise=15_000 + TICK_PAISE)
        )
        rich = validate_one(
            make_snapshot(bid_paise=300_000 - TICK_PAISE, ask_paise=300_000 + TICK_PAISE)
        )
        cheap_quote = execution.entry_quote(cheap, Direction.LONG, 1)
        rich_quote = execution.entry_quote(rich, Direction.LONG, 1)
        cheap_ticks = (cheap_quote.price - cheap_quote.reference_price) / 0.05
        rich_ticks = (rich_quote.price - rich_quote.reference_price) / 0.05
        assert rich_ticks > cheap_ticks * 5

    def test_quotes_are_on_the_tick_grid(self) -> None:
        execution = model(entry_aggression_bps=3.0)
        snapshot = validate_one(make_snapshot())
        quote = execution.entry_quote(snapshot, Direction.LONG, 1)
        assert round(quote.price / 0.05) == pytest.approx(quote.price / 0.05, abs=1e-9)

    def test_entry_is_never_better_than_the_touch(self) -> None:
        # A zero or negative offset must not produce a price improvement.
        execution = model(entry_aggression_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        buy = execution.entry_quote(snapshot, Direction.LONG, 1)
        sell = execution.entry_quote(snapshot, Direction.SHORT, 1)
        assert buy.price >= snapshot.best_ask.price
        assert sell.price <= snapshot.best_bid.price

    def test_exit_quote_uses_the_opposite_side(self) -> None:
        execution = model(exit_slippage_ticks=1.0)
        snapshot = validate_one(make_snapshot())
        long_exit = execution.exit_quote(snapshot, Direction.LONG, 1)
        short_exit = execution.exit_quote(snapshot, Direction.SHORT, 1)
        assert long_exit.price == pytest.approx(799.95 - 0.05)
        assert short_exit.price == pytest.approx(800.05 + 0.05)

    def test_flat_direction_is_rejected(self) -> None:
        execution = model()
        snapshot = validate_one(make_snapshot())
        with pytest.raises(ValueError):
            execution.entry_quote(snapshot, Direction.FLAT, 1)
        with pytest.raises(ValueError):
            execution.exit_quote(snapshot, Direction.FLAT, 1)

    def test_zero_quantity_is_rejected(self) -> None:
        execution = model()
        snapshot = validate_one(make_snapshot())
        with pytest.raises(ValueError):
            execution.entry_quote(snapshot, Direction.LONG, 0)


class TestDepthWalk:
    def test_large_order_walks_the_ladder(self) -> None:
        execution = model(use_depth_walk=True, entry_aggression_ticks=0.0)
        snapshot = validate_one(
            make_snapshot(
                asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (100, 100, 100))
            )
        )
        quote = execution.entry_quote(snapshot, Direction.LONG, 250)
        assert quote.levels_consumed == 3
        assert quote.filled_quantity == 250
        # Volume-weighted: 100 at 800.05, 100 at 800.10, 50 at 800.15.
        expected = (100 * 800.05 + 100 * 800.10 + 50 * 800.15) / 250
        assert quote.price == pytest.approx(round(expected / 0.05) * 0.05)
        assert quote.price > snapshot.best_ask.price

    def test_insufficient_depth_reports_a_partial_fill(self) -> None:
        execution = model(use_depth_walk=True, entry_aggression_ticks=0.0)
        snapshot = validate_one(
            make_snapshot(asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (10, 10)))
        )
        quote = execution.entry_quote(snapshot, Direction.LONG, 500)
        assert not quote.complete
        assert quote.filled_quantity == 20

    def test_partial_fill_policy_blocks_entry_before_and_during_open(self) -> None:
        execution = model(
            use_depth_walk=True,
            allow_partial_fill=False,
            entry_aggression_ticks=0.0,
        )
        snapshot = validate_one(
            make_snapshot(asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (10, 10)))
        )
        allowed, detail = execution.entry_is_fillable(
            snapshot,
            Direction.LONG,
            500,
        )
        assert not allowed
        assert "available 20" in detail
        with pytest.raises(ValueError, match="partial entry fill is disabled"):
            execution.open_position(
                snapshot,
                Direction.LONG,
                500,
                monotonic_ms=0.0,
            )

    def test_allowed_partial_fill_opens_only_the_executed_quantity(self) -> None:
        execution = model(
            use_depth_walk=True,
            allow_partial_fill=True,
            entry_aggression_ticks=0.0,
        )
        snapshot = validate_one(
            make_snapshot(asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (10, 10)))
        )
        position = execution.open_position(
            snapshot,
            Direction.LONG,
            500,
            monotonic_ms=0.0,
        )
        assert position.quantity == 20
        assert not position.entry_quote.complete
        assert position.entry_quote.requested_quantity == 500

    def test_touch_fill_ignores_size(self) -> None:
        execution = model(use_depth_walk=False, entry_aggression_ticks=0.0)
        snapshot = validate_one(
            make_snapshot(asks=ladder(BASE_PAISE + TICK_PAISE, TICK_PAISE, (10, 10)))
        )
        quote = execution.entry_quote(snapshot, Direction.LONG, 500)
        assert quote.complete
        assert quote.price == pytest.approx(800.05)


class TestPositionsAndPnL:
    def test_position_levels_are_derived_from_tick_distances(self) -> None:
        execution = model(entry_aggression_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            snapshot, Direction.LONG, 10, monotonic_ms=1_000.0
        )
        assert position.entry_price == pytest.approx(800.05)
        assert position.stop_price == pytest.approx(800.05 - 5 * 0.05)
        assert position.target_price == pytest.approx(800.05 + 10 * 0.05)
        assert position.quantity == 10
        assert position.is_open

    def test_profitable_long_mark_to_market(self) -> None:
        execution = model(entry_aggression_ticks=0.0, exit_slippage_ticks=0.0)
        entry_snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            entry_snapshot, Direction.LONG, 100, monotonic_ms=0.0
        )
        rallied = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE + 20 * TICK_PAISE,
                ask_paise=BASE_PAISE + 22 * TICK_PAISE,
            )
        )
        quote, report = execution.mark_to_market(
            position, rallied, monotonic_ms=2_500.0
        )
        assert quote.price == pytest.approx(801.00)
        assert report.gross_ticks == pytest.approx((801.00 - 800.05) / 0.05)
        assert report.gross_rupees == pytest.approx((801.00 - 800.05) * 100)
        assert report.cost_rupees == 0.0
        assert report.net_rupees == pytest.approx(report.gross_rupees)
        assert report.holding_ms == pytest.approx(2_500.0)

    def test_partial_exit_pnl_is_limited_to_the_executable_quantity(self) -> None:
        execution = model(
            use_depth_walk=True,
            entry_aggression_ticks=0.0,
            exit_slippage_ticks=0.0,
        )
        entry_snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            entry_snapshot,
            Direction.LONG,
            500,
            monotonic_ms=0.0,
        )
        thin_exit = validate_one(
            make_snapshot(
                bids=ladder(
                    BASE_PAISE - TICK_PAISE,
                    -TICK_PAISE,
                    (10, 10),
                )
            )
        )
        quote, report = execution.mark_to_market(
            position,
            thin_exit,
            monotonic_ms=100.0,
        )
        assert not quote.complete
        assert quote.filled_quantity == 20
        assert report.quantity == 20
        assert report.gross_rupees == pytest.approx(
            (quote.price - position.entry_price) * 20
        )

    def test_repeated_partial_exits_allocate_one_entry_order_cost(self) -> None:
        execution = model(
            use_depth_walk=True,
            entry_aggression_ticks=0.0,
            exit_slippage_ticks=0.0,
            cost=CostConfig(enabled=True),
        )
        position = execution.open_position(
            validate_one(make_snapshot()),
            Direction.LONG,
            500,
            monotonic_ms=0.0,
        )
        original_entry_cost = position.remaining_entry_cost_rupees
        thin_exit = validate_one(
            make_snapshot(
                bids=ladder(
                    BASE_PAISE - TICK_PAISE,
                    -TICK_PAISE,
                    (10, 10),
                )
            )
        )
        total_reported_cost = 0.0
        exit_price = 0.0
        for index in range(25):
            quote, report = execution.mark_to_market(
                position,
                thin_exit,
                monotonic_ms=float(index),
            )
            assert quote.filled_quantity == 20
            assert report.quantity == 20
            total_reported_cost += report.cost_rupees
            exit_price = quote.price
            if quote.complete:
                break
            position = execution.residual_position(
                position,
                quote.filled_quantity,
            )

        expected_exit_cost = 25 * execution.costs.leg_cost(
            price=exit_price,
            quantity=20,
            buying=False,
        )
        assert total_reported_cost == pytest.approx(
            original_entry_cost + expected_exit_cost
        )

    def test_short_pnl_has_the_opposite_sign(self) -> None:
        execution = model(entry_aggression_ticks=0.0, exit_slippage_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            snapshot, Direction.SHORT, 50, monotonic_ms=0.0
        )
        fell = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE - 20 * TICK_PAISE,
                ask_paise=BASE_PAISE - 18 * TICK_PAISE,
            )
        )
        _, report = execution.mark_to_market(position, fell, monotonic_ms=1_000.0)
        assert report.gross_ticks > 0.0
        assert report.gross_rupees > 0.0

    def test_stop_is_evaluated_on_the_exit_side_not_the_mid(self) -> None:
        execution = model(entry_aggression_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            snapshot, Direction.LONG, 1, monotonic_ms=0.0
        )
        assert position.stop_price == pytest.approx(799.80)

        # The bid has reached the stop while the mid is still above it. A
        # mid-based test would report no stop; a real sell order would have been
        # filled, so the exit-side test is the correct one.
        touching = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE - 4 * TICK_PAISE,
                ask_paise=BASE_PAISE - 2 * TICK_PAISE,
            )
        )
        assert touching.best_bid.price == pytest.approx(799.80)
        assert touching.mid > position.stop_price
        assert execution.stop_hit(position, touching)

        safe = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE - 3 * TICK_PAISE,
                ask_paise=BASE_PAISE - TICK_PAISE,
            )
        )
        assert not execution.stop_hit(position, safe)
        assert not execution.target_hit(position, safe)

    def test_target_is_evaluated_on_the_exit_side(self) -> None:
        execution = model(entry_aggression_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            snapshot, Direction.LONG, 1, monotonic_ms=0.0
        )
        assert position.target_price == pytest.approx(800.55)
        reached = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE + 11 * TICK_PAISE,
                ask_paise=BASE_PAISE + 13 * TICK_PAISE,
            )
        )
        assert reached.best_bid.price == pytest.approx(800.55)
        assert execution.target_hit(position, reached)

    def test_unrealised_ticks_tracks_the_exit_side(self) -> None:
        execution = model(entry_aggression_ticks=0.0)
        snapshot = validate_one(make_snapshot())
        position = execution.open_position(
            snapshot, Direction.LONG, 1, monotonic_ms=0.0
        )
        moved = validate_one(
            make_snapshot(
                bid_paise=BASE_PAISE + 6 * TICK_PAISE,
                ask_paise=BASE_PAISE + 8 * TICK_PAISE,
            )
        )
        assert execution.unrealised_ticks(position, moved) == pytest.approx(5.0)


class TestCostModel:
    def test_disabled_costs_are_zero(self) -> None:
        costs = CostModel(CostConfig(enabled=False))
        quote = costs.round_trip(
            buy_price=800.0, sell_price=801.0, quantity=100, tick_size=0.05
        )
        assert quote.rupees == 0.0
        assert quote.bps == 0.0
        assert not costs.enabled

    def test_round_trip_components_are_asymmetric(self) -> None:
        # Securities transaction tax applies to the sell leg and stamp duty to the
        # buy leg, so swapping the legs changes the total.
        costs = CostModel(
            CostConfig(
                enabled=True,
                brokerage_flat_per_order=20.0,
                brokerage_rate=0.0003,
                stt_sell_rate=0.00025,
                stamp_duty_buy_rate=0.00003,
            )
        )
        low_buy = costs.round_trip(
            buy_price=100.0, sell_price=1_000.0, quantity=100, tick_size=0.05
        )
        high_buy = costs.round_trip(
            buy_price=1_000.0, sell_price=100.0, quantity=100, tick_size=0.05
        )
        assert low_buy.rupees != high_buy.rupees

    def test_brokerage_is_the_lower_of_flat_and_rate(self) -> None:
        costs = CostModel(
            CostConfig(
                enabled=True,
                brokerage_flat_per_order=20.0,
                brokerage_rate=0.0003,
                brokerage_cap_per_order=20.0,
                exchange_txn_rate=0.0,
                sebi_rate=0.0,
                stt_sell_rate=0.0,
                stamp_duty_buy_rate=0.0,
                gst_rate=0.0,
            )
        )
        # Small notional: the rate is cheaper than the flat fee.
        small = costs.round_trip(
            buy_price=100.0, sell_price=100.0, quantity=10, tick_size=0.05
        )
        assert small.rupees == pytest.approx(2 * 0.0003 * 1_000.0)
        # Large notional: the flat fee caps it.
        large = costs.round_trip(
            buy_price=1_000.0, sell_price=1_000.0, quantity=1_000, tick_size=0.05
        )
        assert large.rupees == pytest.approx(40.0)

    def test_cost_in_ticks_is_the_breakeven_move(self) -> None:
        costs = CostModel(CostConfig(enabled=True))
        quote = costs.round_trip(
            buy_price=800.0, sell_price=800.0, quantity=100, tick_size=0.05
        )
        # Moving ``ticks`` in our favour must recover exactly the rupee cost.
        assert quote.ticks * 0.05 * 100 == pytest.approx(quote.rupees)

    def test_zero_quantity_is_costless(self) -> None:
        costs = CostModel(CostConfig(enabled=True))
        quote = costs.round_trip(
            buy_price=800.0, sell_price=800.0, quantity=0, tick_size=0.05
        )
        assert quote.rupees == 0.0


class TestCostGate:
    def test_gate_is_transparent_when_disabled(self) -> None:
        execution = model(enforce_cost_gate=False)
        snapshot = validate_one(make_snapshot())
        allowed, detail = execution.clears_cost(snapshot, 1)
        assert allowed
        assert detail == ""

    def test_gate_blocks_a_target_that_cannot_cover_costs(self) -> None:
        execution = ExecutionModel(
            ExecutionConfig(
                enforce_cost_gate=True,
                min_edge_multiple=2.0,
                cost=CostConfig(enabled=True),
            ),
            StateMachineConfig(target_ticks=1.0),
        )
        snapshot = validate_one(make_snapshot())
        allowed, detail = execution.clears_cost(snapshot, 1)
        assert not allowed
        assert "below required" in detail

    def test_gate_allows_a_target_that_clears_costs(self) -> None:
        execution = ExecutionModel(
            ExecutionConfig(
                enforce_cost_gate=True,
                min_edge_multiple=1.0,
                cost=CostConfig(
                    enabled=True,
                    brokerage_flat_per_order=0.0,
                    brokerage_rate=0.0,
                    brokerage_cap_per_order=0.0,
                    exchange_txn_rate=0.0,
                    sebi_rate=0.0,
                    stt_sell_rate=0.0,
                    stamp_duty_buy_rate=0.0,
                    gst_rate=0.0,
                ),
            ),
            StateMachineConfig(target_ticks=40.0),
        )
        snapshot = validate_one(make_snapshot())
        allowed, _ = execution.clears_cost(snapshot, 1)
        assert allowed
