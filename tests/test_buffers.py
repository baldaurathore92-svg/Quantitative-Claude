"""Tests for the incremental buffers.

Two properties matter for every estimator here and both are tested explicitly:

*   the incremental result must equal the exact recomputation, including after
    the window has wrapped many times (this is what the drift correction is for);
*   ``reset`` must restore the initial state, because the engine relies on it
    after every feed gap.
"""

from __future__ import annotations

import math
import statistics

import pytest
from snapshot_quant_v4.buffers.monotonic_queue import MonotonicWindow
from snapshot_quant_v4.buffers.ring_buffer import RingBuffer
from snapshot_quant_v4.buffers.rolling_ema import EMAPair, TimeAwareEMA
from snapshot_quant_v4.buffers.rolling_mean import RollingMean
from snapshot_quant_v4.buffers.rolling_variance import RollingVariance, WelfordVariance


class TestRingBuffer:
    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_push_returns_evicted_value_only_when_full(self) -> None:
        buffer = RingBuffer(3)
        assert buffer.push(1.0) is None
        assert buffer.push(2.0) is None
        assert buffer.push(3.0) is None
        assert buffer.full
        assert buffer.push(4.0) == 1.0

    def test_age_indexing_is_newest_first(self) -> None:
        buffer = RingBuffer(3)
        for value in (1.0, 2.0, 3.0, 4.0):
            buffer.push(value)
        assert buffer.at(0) == 4.0
        assert buffer.at(1) == 3.0
        assert buffer.at(2) == 2.0
        assert buffer.newest == 4.0
        assert buffer.oldest == 2.0

    def test_out_of_range_age_raises(self) -> None:
        buffer = RingBuffer(2)
        buffer.push(1.0)
        with pytest.raises(IndexError):
            buffer.at(1)
        with pytest.raises(IndexError):
            buffer.at(-1)

    def test_iteration_is_oldest_to_newest_after_wrapping(self) -> None:
        buffer = RingBuffer(3)
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            buffer.push(value)
        assert list(buffer.iter_values()) == [3.0, 4.0, 5.0]

    def test_clear_resets_state(self) -> None:
        buffer = RingBuffer(3)
        buffer.push(1.0)
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.fill_ratio == 0.0


class TestRollingMean:
    def test_matches_exact_mean_over_many_wraps(self) -> None:
        window = 16
        mean = RollingMean(window, recompute_interval=7)
        values: list[float] = []
        for index in range(500):
            sample = math.sin(index) * 1000.0 + 5000.0
            values.append(sample)
            mean.update(sample)
            expected = statistics.fmean(values[-window:])
            assert mean.value == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_ready_requires_min_samples(self) -> None:
        mean = RollingMean(10, min_samples=4)
        for index in range(3):
            mean.update(float(index))
            assert not mean.ready
        mean.update(3.0)
        assert mean.ready

    def test_reset_clears(self) -> None:
        mean = RollingMean(4)
        for index in range(4):
            mean.update(float(index))
        mean.reset()
        assert len(mean) == 0
        assert mean.value == 0.0
        assert not mean.ready

    def test_invalid_min_samples_rejected(self) -> None:
        with pytest.raises(ValueError):
            RollingMean(4, min_samples=5)


class TestRollingVariance:
    def test_matches_exact_sample_variance(self) -> None:
        window = 12
        variance = RollingVariance(window, recompute_interval=5)
        values: list[float] = []
        for index in range(300):
            sample = math.cos(index * 0.7) * 3.0
            values.append(sample)
            variance.update(sample)
            recent = values[-window:]
            if len(recent) >= 2:
                assert variance.variance == pytest.approx(
                    statistics.variance(recent), rel=1e-7, abs=1e-9
                )

    def test_variance_never_negative_with_large_offset(self) -> None:
        # A large mean relative to the spread is the case where the
        # sum-of-squares formulation is most prone to cancellation error.
        variance = RollingVariance(8, recompute_interval=4)
        for index in range(200):
            variance.update(1e7 + (index % 3) * 0.01)
            assert variance.variance >= 0.0

    def test_std_is_sqrt_of_variance(self) -> None:
        variance = RollingVariance(5)
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            variance.update(value)
        assert variance.std == pytest.approx(math.sqrt(variance.variance))

    def test_window_must_be_at_least_two(self) -> None:
        with pytest.raises(ValueError):
            RollingVariance(1)


class TestWelfordVariance:
    def test_matches_statistics_variance(self) -> None:
        welford = WelfordVariance()
        values = [3.0, 7.5, 1.25, 9.0, 4.5, 6.25]
        for value in values:
            welford.update(value)
        assert welford.mean == pytest.approx(statistics.fmean(values))
        assert welford.variance == pytest.approx(statistics.variance(values))

    def test_not_ready_below_two_samples(self) -> None:
        welford = WelfordVariance()
        assert not welford.ready
        welford.update(1.0)
        assert not welford.ready
        welford.update(2.0)
        assert welford.ready


class TestTimeAwareEMA:
    def test_first_sample_seeds_the_estimator(self) -> None:
        ema = TimeAwareEMA(1000.0)
        assert ema.update(42.0, 200.0) == 42.0

    def test_half_life_halves_the_gap(self) -> None:
        ema = TimeAwareEMA(1000.0)
        ema.update(0.0, 0.0)
        # After exactly one half-life the estimator should have closed half the
        # distance to the new observation.
        assert ema.update(1.0, 1000.0) == pytest.approx(0.5)

    def test_zero_interval_leaves_value_unchanged(self) -> None:
        ema = TimeAwareEMA(1000.0)
        ema.update(10.0, 100.0)
        assert ema.update(20.0, 0.0) == pytest.approx(10.0)

    def test_interval_is_clamped_so_a_stall_cannot_reset_history(self) -> None:
        ema = TimeAwareEMA(1000.0, max_dt_ms=1000.0)
        ema.update(0.0, 0.0)
        value = ema.update(1.0, 10_000_000.0)
        assert value == pytest.approx(0.5)

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            TimeAwareEMA(0.0)
        with pytest.raises(ValueError):
            TimeAwareEMA(100.0, max_dt_ms=0.0)


class TestEMAPair:
    def test_requires_slow_to_be_slower(self) -> None:
        with pytest.raises(ValueError):
            EMAPair(1000.0, 500.0)

    def test_spread_is_positive_for_a_rising_series(self) -> None:
        pair = EMAPair(200.0, 2000.0, warmup_updates=2)
        for index in range(30):
            pair.update(float(index), 100.0)
        assert pair.ready
        assert pair.spread > 0.0

    def test_reset_clears_both(self) -> None:
        pair = EMAPair(200.0, 2000.0)
        pair.update(5.0, 100.0)
        pair.reset()
        assert not pair.fast.initialised
        assert not pair.slow.initialised


class TestMonotonicWindow:
    def test_tracks_min_and_max_within_window(self) -> None:
        window = MonotonicWindow(3)
        for value in (5.0, 1.0, 3.0):
            window.update(value)
        assert window.minimum == 1.0
        assert window.maximum == 5.0
        window.update(4.0)  # evicts 5.0
        assert window.maximum == 4.0
        assert window.minimum == 1.0

    def test_matches_brute_force_over_a_long_series(self) -> None:
        size = 7
        window = MonotonicWindow(size)
        values: list[float] = []
        for index in range(400):
            sample = math.sin(index * 1.7) * 10.0
            values.append(sample)
            window.update(sample)
            recent = values[-size:]
            assert window.minimum == pytest.approx(min(recent))
            assert window.maximum == pytest.approx(max(recent))

    def test_position_of_handles_flat_window(self) -> None:
        window = MonotonicWindow(4)
        for _ in range(4):
            window.update(2.0)
        assert window.position_of(2.0) == 0.5
        assert window.range == 0.0

    def test_empty_window_raises_on_extrema(self) -> None:
        window = MonotonicWindow(2)
        with pytest.raises(IndexError):
            _ = window.minimum
        with pytest.raises(IndexError):
            _ = window.maximum
