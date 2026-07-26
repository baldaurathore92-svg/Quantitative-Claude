"""Per-feature confidence model.

Model
-----
Confidence is a product of market-condition factors, each raised to a
per-feature exponent::

    confidence(f) = local_confidence(f) * regime_multiplier
                    * prod_i factor_i ** sensitivity_i(f)

Every ``factor_i`` lies in ``[0, 1]``, so the product is automatically in
``[0, 1]`` with no clipping, and it is monotone in every factor: conditions can
only ever reduce confidence, never manufacture it. Exponents express *how much*
a given feature cares about a given condition, which is where the asymmetry
between feature families lives:

*   a level feature (microprice, imbalance, depth slope) degrades primarily with
    a widening spread and vanishing depth;
*   a history feature (momentum, acceleration, persistence, refill) degrades
    primarily with an unstable, gap-interrupted sample history, and is far less
    sensitive to the instantaneous spread.

The five factors
----------------
``spread``
    Tightness of the current spread in ticks.
``liquidity``
    Current depth relative to the instrument's own rolling reference depth.
``volatility``
    Mid volatility relative to a tolerated multiple of its own baseline. High
    realised volatility widens the confidence interval around every estimate.
``book_quality``
    The continuous score produced by the quality gate (ladder completeness,
    warmup progress, spread, liquidity combined).
``stability``
    How often the touch prices stayed put, blended with spread compression.
    Compression enters here rather than in the score because a narrowing spread
    has magnitude but no direction.

Why confidence is *not* used to shrink the score
------------------------------------------------
The composite divides by its own weight mass, so confidence reweights features
*relative to each other* without deflating the overall magnitude. Aggregate
confidence is then applied separately, as a gate on entry. The alternative —
multiplying the score by confidence and comparing it with a volatility-adaptive
threshold — counts deteriorating market conditions twice and makes the score
incomparable between regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ConfidenceConfig, FeatureSensitivity
from ..utils.constants import F_SPREAD_COMPRESSION
from ..utils.math_utils import clamp, linear_scale, safe_div
from ..utils.types import FeatureKind, FeatureMap, FeatureValue, QualityReport, Regime
from .stats import SharedStatistics

#: Smallest factor value fed into the power function. A hard zero would make the
#: whole product collapse to zero for any feature with a non-zero exponent, which
#: destroys the ordering information the model is meant to preserve. The value is
#: small enough to express "this condition has failed" and large enough that a
#: single weak factor with a high exponent does not annihilate an otherwise sound
#: reading.
_MIN_FACTOR = 0.02

#: Weakest multiplier a fully widening spread may apply to the stability factor.
_COMPRESSION_FLOOR = 0.75


@dataclass(frozen=True, slots=True)
class ConfidenceFactors:
    """The market-condition factors for one snapshot, each in ``[0, 1]``.

    Exposed as a record so the console and the logs can show *why* confidence
    fell, rather than only that it fell.
    """

    spread: float
    liquidity: float
    volatility: float
    book_quality: float
    stability: float
    warmup: float

    def describe(self) -> str:
        """Return a compact operator-facing summary."""
        return (
            f"spr={self.spread:.2f} liq={self.liquidity:.2f} "
            f"vol={self.volatility:.2f} book={self.book_quality:.2f} "
            f"stab={self.stability:.2f} warm={self.warmup:.2f}"
        )

    def logarithms(self) -> tuple[float, float, float, float, float, float]:
        """Return the natural logarithms of the six factors.

        The confidence model is a product of powers, which is evaluated as
        ``exp(sum(exponent * log(factor)))``: one exponential and six
        multiplications per feature instead of six ``pow`` calls. The result is
        mathematically identical, and the substitution is safe because every
        factor is clamped strictly above zero before it reaches this method.
        """
        return (
            math.log(self.spread),
            math.log(self.liquidity),
            math.log(self.volatility),
            math.log(self.book_quality),
            math.log(self.stability),
            math.log(self.warmup),
        )


class ConfidenceModel:
    """Computes per-feature confidence from shared market conditions.

    The model is stateless: it derives everything from the snapshot's statistics
    and the quality report. That makes it trivially testable and means it can be
    shared between instruments without any per-symbol state.

    Parameters
    ----------
    config:
        Confidence parameters and per-feature sensitivities.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ConfidenceConfig) -> None:
        self._config = config

    # -- factors ----------------------------------------------------------- #

    def factors(
        self,
        *,
        spread_ticks: float,
        stats: SharedStatistics,
        quality: QualityReport,
        features: FeatureMap,
    ) -> ConfidenceFactors:
        """Compute the five market-condition factors plus warmup progress."""
        config = self._config
        spread_factor = linear_scale(
            -spread_ticks, -config.spread_wide_ticks, -config.spread_tight_ticks
        )
        liquidity_factor = quality.liquidity_score
        volatility_factor = self._volatility_factor(stats)
        stability_factor = self._stability_factor(stats, features)
        return ConfidenceFactors(
            spread=self._guard(spread_factor),
            liquidity=self._guard(liquidity_factor),
            volatility=self._guard(volatility_factor),
            book_quality=self._guard(quality.book_quality),
            stability=self._guard(stability_factor),
            warmup=self._guard(stats.warmup_ratio),
        )

    def _volatility_factor(self, stats: SharedStatistics) -> float:
        """Map realised mid volatility onto ``[0, 1]``.

        The comparison is against the instrument's own mean spread, in ticks:
        volatility that is small relative to the spread cannot dislodge a
        few-tick edge, whereas volatility of several spreads per snapshot makes
        every level estimate transient. Returns ``1.0`` until the estimator is
        ready so that a warming engine is not penalised twice (``warmup`` already
        covers that).
        """
        if not stats.mid_volatility_ready:
            return 1.0
        reference = max(stats.spread_mean_ticks, 1.0) * self._config.volatility_tolerance
        excess = safe_div(stats.mid_volatility_ticks, reference, default=0.0)
        return clamp(1.0 - excess, 0.0, 1.0)

    def _stability_factor(
        self, stats: SharedStatistics, features: FeatureMap
    ) -> float:
        """Combine touch stability with spread compression.

        Touch stability is the dominant term. Compression enters only as a mild
        multiplier in ``[COMPRESSION_FLOOR, 1]``: its neutral value is 0.5 by
        construction, so using it symmetrically would permanently halve the
        stability factor in a perfectly normal book. Widening reduces confidence
        somewhat, compression restores it, and neither dominates the measure.
        """
        touch = clamp(stats.touch_stability, 0.0, 1.0)
        compression = features.get(F_SPREAD_COMPRESSION)
        if compression is None or not compression.valid:
            return touch
        modifier = _COMPRESSION_FLOOR + (1.0 - _COMPRESSION_FLOOR) * clamp(
            compression.value, 0.0, 1.0
        )
        return touch * modifier

    @staticmethod
    def _guard(factor: float) -> float:
        """Clamp a factor into ``[_MIN_FACTOR, 1]``."""
        return clamp(factor, _MIN_FACTOR, 1.0)

    # -- application ------------------------------------------------------- #

    def confidence_for(
        self,
        feature: FeatureValue,
        factors: ConfidenceFactors,
        *,
        regime: Regime,
        log_factors: tuple[float, float, float, float, float, float] | None = None,
    ) -> float:
        """Return the final confidence for one feature.

        Quality features receive a confidence of zero: they never enter the
        composite, and giving them a non-zero value would invite a future change
        that accidentally included them.

        ``log_factors`` is an optional pre-computed
        :meth:`ConfidenceFactors.logarithms`, supplied by :meth:`apply` so the
        logarithms are taken once per snapshot rather than once per feature. The
        arithmetic is identical either way.
        """
        if not feature.valid:
            return 0.0
        if feature.kind is FeatureKind.QUALITY:
            return 0.0

        config = self._config
        sensitivity: FeatureSensitivity = config.sensitivity_for(feature.name)
        logs = factors.logarithms() if log_factors is None else log_factors
        exponent = (
            sensitivity.spread * logs[0]
            + sensitivity.liquidity * logs[1]
            + sensitivity.volatility * logs[2]
            + sensitivity.book_quality * logs[3]
            + sensitivity.stability * logs[4]
            + sensitivity.warmup * logs[5]
        )
        confidence = math.exp(exponent)
        confidence *= clamp(feature.local_confidence, 0.0, 1.0)
        confidence *= config.regime_multipliers.get(regime.value, 1.0)
        if confidence < config.min_feature_confidence:
            return 0.0
        return clamp(confidence, 0.0, 1.0)

    def apply(
        self,
        features: dict[str, FeatureValue],
        *,
        spread_ticks: float,
        stats: SharedStatistics,
        quality: QualityReport,
        regime: Regime,
    ) -> ConfidenceFactors:
        """Fill in ``confidence`` on every feature, in place.

        The mapping is mutated rather than rebuilt because it is a short-lived,
        engine-owned dictionary; the :class:`FeatureValue` records themselves stay
        immutable and are rebuilt, not modified.

        The rebuild is written out explicitly instead of using
        ``dataclasses.replace``, which reflects over every field on every call and
        measured as the single most expensive operation in the pipeline.

        Returns
        -------
        ConfidenceFactors
            The factors used, for logging and display.
        """
        factors = self.factors(
            spread_ticks=spread_ticks, stats=stats, quality=quality, features=features
        )
        logs = factors.logarithms()
        for name, feature in features.items():
            confidence = self.confidence_for(
                feature, factors, regime=regime, log_factors=logs
            )
            features[name] = FeatureValue(
                name=feature.name,
                kind=feature.kind,
                raw=feature.raw,
                value=feature.value,
                local_confidence=feature.local_confidence,
                confidence=confidence,
                valid=feature.valid,
                detail=feature.detail,
            )
        return factors


__all__ = ["ConfidenceFactors", "ConfidenceModel"]
