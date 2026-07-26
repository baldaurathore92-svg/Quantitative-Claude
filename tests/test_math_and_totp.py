"""Tests for the numerical helpers and the TOTP implementation."""

from __future__ import annotations

import math

import pytest
from snapshot_quant_v4.utils.math_utils import (
    bps_to_ticks,
    clamp,
    clamp_unit,
    ema_alpha,
    is_tick_aligned,
    linear_scale,
    log_ratio,
    robust_scale,
    round_to_tick,
    safe_div,
    sign,
    tanh_scale,
    ticks_to_bps,
    to_ticks,
)
from snapshot_quant_v4.utils.totp import TotpError, generate_totp, seconds_until_next_step


class TestScalingHelpers:
    def test_clamp_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError):
            clamp(1.0, 2.0, 1.0)

    def test_clamp_unit(self) -> None:
        assert clamp_unit(2.0) == 1.0
        assert clamp_unit(-2.0) == -1.0
        assert clamp_unit(0.25) == 0.25

    def test_safe_div_returns_default_for_zero_denominator(self) -> None:
        assert safe_div(1.0, 0.0) == 0.0
        assert safe_div(1.0, 0.0, default=-1.0) == -1.0
        assert safe_div(3.0, 1.5) == 2.0

    def test_sign_with_deadband(self) -> None:
        assert sign(0.001) == 1
        assert sign(0.001, deadband=0.01) == 0
        assert sign(-0.5) == -1

    def test_linear_scale_clamps_outside_range(self) -> None:
        assert linear_scale(-5.0, 0.0, 10.0) == 0.0
        assert linear_scale(15.0, 0.0, 10.0) == 1.0
        assert linear_scale(5.0, 0.0, 10.0) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            linear_scale(1.0, 5.0, 5.0)

    def test_tanh_scale_is_bounded_and_odd(self) -> None:
        # Mathematically the range is open; in binary floating point tanh
        # saturates to exactly 1.0 for arguments beyond roughly 19, which is why
        # the engine treats the bound as inclusive everywhere.
        assert tanh_scale(100.0, 1.0) <= 1.0
        assert tanh_scale(2.0, 1.0) < 1.0
        assert tanh_scale(-3.0, 2.0) == pytest.approx(-tanh_scale(3.0, 2.0))
        with pytest.raises(ValueError):
            tanh_scale(1.0, 0.0)

    def test_robust_scale_uses_the_floor_when_dispersion_collapses(self) -> None:
        # With a near-zero dispersion the floor must prevent saturation.
        quiet = robust_scale(0.01, 0.0, floor=0.5, k=2.0)
        assert abs(quiet) < 0.05

    def test_robust_scale_saturates_for_large_moves(self) -> None:
        assert robust_scale(10.0, 0.1, floor=0.01, k=2.0) > 0.99

    def test_log_ratio_is_finite_when_a_level_empties(self) -> None:
        assert math.isfinite(log_ratio(0.0, 5000.0))
        assert log_ratio(100.0, 100.0) == pytest.approx(0.0)


class TestTickConversions:
    def test_to_ticks(self) -> None:
        assert to_ticks(0.15, 0.05) == pytest.approx(3.0)

    def test_bps_and_ticks_round_trip(self) -> None:
        price = 800.0
        tick = 0.05
        ticks = bps_to_ticks(3.0, price, tick)
        assert ticks == pytest.approx(4.8)
        assert ticks_to_bps(ticks, price, tick) == pytest.approx(3.0)

    def test_same_bps_is_a_different_number_of_ticks_at_different_prices(self) -> None:
        # This is the whole reason the execution model converts basis points to
        # ticks against a live price instead of hardcoding an offset.
        cheap = bps_to_ticks(3.0, 150.0, 0.05)
        rich = bps_to_ticks(3.0, 3000.0, 0.05)
        assert rich > cheap * 15

    def test_round_to_tick_uses_the_grid(self) -> None:
        assert round_to_tick(800.03, 0.05) == pytest.approx(800.05)
        assert round_to_tick(800.02, 0.05) == pytest.approx(800.0)

    def test_is_tick_aligned(self) -> None:
        assert is_tick_aligned(800.05, 0.05)
        assert not is_tick_aligned(800.03, 0.05)

    def test_rejects_non_positive_tick_size(self) -> None:
        with pytest.raises(ValueError):
            round_to_tick(1.0, 0.0)
        with pytest.raises(ValueError):
            is_tick_aligned(1.0, -0.05)


class TestEmaAlpha:
    def test_alpha_is_one_half_at_the_half_life(self) -> None:
        assert ema_alpha(1000.0, 1000.0, 5000.0) == pytest.approx(0.5)

    def test_alpha_is_zero_for_zero_interval(self) -> None:
        assert ema_alpha(0.0, 1000.0, 5000.0) == 0.0

    def test_alpha_is_capped_below_one(self) -> None:
        assert ema_alpha(1e9, 1000.0, 5000.0) < 1.0

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            ema_alpha(100.0, 0.0, 1000.0)
        with pytest.raises(ValueError):
            ema_alpha(100.0, 1000.0, 0.0)


class TestTotp:
    #: RFC 6238 appendix B seed: the ASCII string "12345678901234567890".
    SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            (59, "94287082"),
            (1_111_111_109, "07081804"),
            (1_111_111_111, "14050471"),
            (1_234_567_890, "89005924"),
            (2_000_000_000, "69279037"),
        ],
    )
    def test_matches_rfc6238_sha1_vectors(self, timestamp: int, expected: str) -> None:
        assert generate_totp(self.SECRET, timestamp=timestamp, digits=8) == expected

    def test_six_digit_code_is_the_tail_of_the_eight_digit_code(self) -> None:
        eight = generate_totp(self.SECRET, timestamp=59, digits=8)
        six = generate_totp(self.SECRET, timestamp=59, digits=6)
        assert six == eight[-6:]

    def test_accepts_lowercase_and_spaced_secrets(self) -> None:
        spaced = "gezd gnbv gy3t qojq gezd gnbv gy3t qojq"
        assert generate_totp(spaced, timestamp=59, digits=8) == "94287082"

    def test_rejects_invalid_secret(self) -> None:
        with pytest.raises(TotpError):
            generate_totp("not base32 !!", timestamp=0)
        with pytest.raises(TotpError):
            generate_totp("", timestamp=0)

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(TotpError):
            generate_totp(self.SECRET, timestamp=0, digits=4)
        with pytest.raises(TotpError):
            generate_totp(self.SECRET, timestamp=0, interval=0)

    def test_seconds_until_next_step(self) -> None:
        assert seconds_until_next_step(timestamp=25.0, interval=30) == pytest.approx(5.0)
        with pytest.raises(TotpError):
            seconds_until_next_step(interval=0)
