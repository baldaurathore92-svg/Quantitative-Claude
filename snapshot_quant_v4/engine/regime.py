"""Deterministic regime detection.

No machine learning, no fitted parameters, no hidden state: the label is a pure
function of three observable statistics plus an explicit hysteresis counter, so
the same snapshot sequence always yields the same labels.

Inputs
------
``efficiency_ratio``
    Kaufman's ratio of net displacement to path length over the regime window,
    in ``[0, 1]``. Near 1 the mid walked in a straight line; near 0 it
    oscillated. This single statistic separates "moved" from "moved *somewhere*"
    far more reliably than a return threshold, because it is invariant to the
    size of the move.
``mid volatility``
    Standard deviation of per-snapshot mid changes, in ticks.
``trend displacement``
    Fast minus slow mid EMA, in ticks: a signed measure of where price is
    relative to its own recent baseline.

A fourth input, the sign of pressure momentum, is supplied by the caller and is
used only to distinguish a pullback (trend intact, pressure temporarily against
it) from a trend.

Hysteresis
----------
Without hysteresis the label flips on almost every snapshot near a boundary, and
because the composite weights are regime dependent, the score would flip with
it. Two guards are applied: a new label must be observed ``confirm_snapshots``
times consecutively, and the incumbent label must have held for
``min_dwell_snapshots``. The first transition out of ``UNKNOWN`` waives the
dwell requirement, since there is no incumbent to protect.
"""

from __future__ import annotations

from ..config import RegimeConfig
from ..utils.math_utils import sign
from ..utils.types import Regime, RegimeResult
from .stats import SharedStatistics


class RegimeDetector:
    """Classifies market state into TREND, PULLBACK, RANGE or NOISE.

    One instance per instrument: the hysteresis counters are per-instrument
    state.

    Parameters
    ----------
    config:
        Regime thresholds and hysteresis parameters.
    """

    __slots__ = ("_candidate", "_candidate_count", "_config", "_dwell", "_regime")

    def __init__(self, config: RegimeConfig) -> None:
        self._config = config
        self._regime = Regime.UNKNOWN
        self._dwell = 0
        self._candidate = Regime.UNKNOWN
        self._candidate_count = 0

    # -- state ------------------------------------------------------------- #

    @property
    def regime(self) -> Regime:
        """The currently held regime label."""
        return self._regime

    def reset(self) -> None:
        """Return to ``UNKNOWN``. Called on feed gaps and reconnects."""
        self._regime = Regime.UNKNOWN
        self._dwell = 0
        self._candidate = Regime.UNKNOWN
        self._candidate_count = 0

    # -- classification ---------------------------------------------------- #

    def update(
        self, stats: SharedStatistics, *, momentum_sign: int = 0
    ) -> RegimeResult:
        """Classify the current state and apply hysteresis.

        Parameters
        ----------
        stats:
            Shared statistics, already updated for this snapshot.
        momentum_sign:
            Sign of the pressure momentum feature, or ``0`` when unavailable.

        Returns
        -------
        RegimeResult
            The held label plus the statistics behind the decision.
        """
        config = self._config
        efficiency = stats.efficiency_ratio(config.window)
        volatility = stats.mid_volatility_ticks
        trend = stats.trend_ticks

        proposed = self._classify(
            stats=stats,
            efficiency=efficiency,
            volatility=volatility,
            trend=trend,
            momentum_sign=momentum_sign,
        )
        self._apply_hysteresis(proposed)

        return RegimeResult(
            regime=self._regime,
            efficiency_ratio=efficiency,
            volatility_ticks=volatility,
            trend_ticks=trend,
            dwell_snapshots=self._dwell,
            detail=(
                f"er={efficiency:.2f} vol={volatility:.2f}t trend={trend:+.2f}t "
                f"proposed={proposed.value}"
            ),
        )

    def _classify(
        self,
        *,
        stats: SharedStatistics,
        efficiency: float,
        volatility: float,
        trend: float,
        momentum_sign: int,
    ) -> Regime:
        """Return the instantaneous label, before hysteresis."""
        config = self._config
        if stats.snapshots < config.min_samples or not stats.trend_ready:
            return Regime.UNKNOWN

        trend_magnitude = abs(trend)
        trend_sign = sign(trend)
        directional = trend_magnitude >= config.trend_min_ticks
        opposed = (
            directional and momentum_sign != 0 and momentum_sign == -trend_sign
        )

        # Violent but directionless: classify as noise before anything else, so
        # that a high-volatility chop is never mistaken for a trend on the
        # strength of a large displacement alone.
        if volatility >= config.vol_high_ticks and efficiency < config.er_trend:
            return Regime.NOISE

        if efficiency >= config.er_trend and directional:
            return Regime.PULLBACK if opposed else Regime.TREND

        if efficiency >= config.pullback_min_er and opposed:
            return Regime.PULLBACK

        if efficiency <= config.er_range and volatility < config.vol_high_ticks:
            return Regime.RANGE

        return Regime.NOISE

    def _apply_hysteresis(self, proposed: Regime) -> None:
        """Commit ``proposed`` only when both hysteresis conditions are met."""
        config = self._config
        self._dwell += 1

        if proposed is self._regime:
            self._candidate = proposed
            self._candidate_count = 0
            return

        if proposed is self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = proposed
            self._candidate_count = 1

        confirmed = self._candidate_count >= config.confirm_snapshots
        dwelt = (
            self._regime is Regime.UNKNOWN
            or self._dwell >= config.min_dwell_snapshots
        )
        if confirmed and dwelt:
            self._regime = proposed
            self._dwell = 0
            self._candidate_count = 0


__all__ = ["RegimeDetector"]
