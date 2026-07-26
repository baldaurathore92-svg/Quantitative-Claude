"""Composite score.

Formula
-------
::

    composite = sum(value * weight * confidence) / sum(weight * confidence)

over the *valid, directional* features only.

The division is the important part. Three properties follow from it and none of
them hold for a bare weighted sum:

1.  **Bounded.** Every feature value is in ``[-1, +1]``, so the ratio is too. The
    score can be compared with a threshold expressed in the same units, with no
    magic scale constant anywhere.
2.  **Comparable across regimes.** Uniformly scaling the weight table cannot
    change the score, so a regime table only expresses *relative* emphasis. A
    corollary that is easy to get wrong: "in a noisy regime, reduce every
    weight" has literally no effect here. Reduced trust in a noisy regime is
    therefore expressed through the confidence multiplier and the threshold,
    where it does what is intended.
3.  **Confidence reweights instead of deflating.** A feature whose confidence
    collapses loses influence relative to its peers, but the surviving features
    still produce a full-magnitude score. Deflating the score with confidence and
    then comparing it against a condition-adaptive threshold would penalise poor
    conditions twice.

Invalid features are *excluded*, not zeroed. A zero would assert "this feature
says neutral", which is a claim, whereas the truth is "this feature has nothing
to say yet".

The usability test is deliberately also scale-free: a score is rejected when
fewer than ``min_features`` features contributed, or when the *weighted average*
confidence falls below ``min_confidence``. An earlier formulation compared the
raw sum of ``weight * confidence`` against a threshold, which quietly reintroduced
a dependence on the absolute size of the numbers in the weight table and
contradicted property 2 above.

Collinearity capping
--------------------
The normalised microprice tilt is algebraically identical to the level-one
imbalance (see :mod:`features.microprice`), so microprice and the L1/L2 weighted
imbalance are near-duplicates of one observation. ``collinear_groups`` caps their
*joint* share of the weight mass, which prevents one piece of evidence from
voting twice. The capping is a single pass and only ever reduces weights, so it
cannot introduce oscillation.
"""

from __future__ import annotations

from ..buffers.rolling_ema import TimeAwareEMA
from ..config import CompositeConfig
from ..utils.math_utils import clamp_unit, safe_div
from ..utils.types import (
    CompositeResult,
    FeatureContribution,
    FeatureKind,
    FeatureMap,
    Regime,
)

#: Returned when nothing usable was available, so callers never see a stale score.
_EMPTY_RESULT = CompositeResult(
    score=0.0,
    smoothed=0.0,
    confidence=0.0,
    weight_mass=0.0,
    used_features=0,
    contributions=(),
    valid=False,
)


class CompositeScorer:
    """Blends directional features into one normalised score.

    One instance per instrument: it owns the smoothing EMA.

    Parameters
    ----------
    config:
        Weight tables and smoothing parameters.
    """

    __slots__ = ("_config", "_smoother")

    def __init__(self, config: CompositeConfig) -> None:
        self._config = config
        self._smoother = TimeAwareEMA(
            config.smoothing_half_life_ms, max_dt_ms=config.max_dt_ms, warmup_updates=1
        )

    def reset(self) -> None:
        """Clear the smoother. Called on feed gaps and reconnects."""
        self._smoother.reset()

    def score(
        self, features: FeatureMap, *, regime: Regime, dt_ms: float
    ) -> CompositeResult:
        """Compute the composite score for one snapshot.

        Parameters
        ----------
        features:
            All features computed for this snapshot, with confidence filled in.
        regime:
            Current regime label, selecting the weight table.
        dt_ms:
            Interval since the previous snapshot, for the time-aware smoother.
        """
        config = self._config
        weights = self._effective_weights(features, regime=regime)
        if not weights:
            return _EMPTY_RESULT

        numerator = 0.0
        weight_mass = 0.0
        contributing_weight = 0.0
        used = 0
        for name, weight in weights.items():
            feature = features[name]
            mass = weight * feature.confidence
            if mass <= 0.0:
                continue
            numerator += feature.value * mass
            weight_mass += mass
            contributing_weight += weight
            used += 1

        # Weighted-average confidence: scale-free, so the validity test does not
        # depend on the absolute magnitudes in the weight table.
        aggregate_confidence = safe_div(weight_mass, contributing_weight)
        if used < config.min_features or aggregate_confidence < config.min_confidence:
            return CompositeResult(
                score=0.0,
                smoothed=self._smoother.value,
                confidence=aggregate_confidence,
                weight_mass=weight_mass,
                used_features=used,
                contributions=(),
                valid=False,
            )

        score = clamp_unit(numerator / weight_mass)
        smoothed = clamp_unit(self._smoother.update(score, dt_ms))

        contributions = tuple(
            FeatureContribution(
                name=name,
                value=features[name].value,
                weight=weight,
                confidence=features[name].confidence,
                contribution=features[name].value
                * weight
                * features[name].confidence
                / weight_mass,
            )
            for name, weight in weights.items()
            if weight * features[name].confidence > 0.0
        )

        return CompositeResult(
            score=score,
            smoothed=smoothed,
            confidence=aggregate_confidence,
            weight_mass=weight_mass,
            used_features=used,
            contributions=contributions,
            valid=True,
        )

    # -- weights ----------------------------------------------------------- #

    def _effective_weights(
        self, features: FeatureMap, *, regime: Regime
    ) -> dict[str, float]:
        """Select the regime weight table and apply collinearity caps."""
        table = self._config.weights_for(regime.value)
        weights: dict[str, float] = {}
        total = 0.0
        for name, feature in features.items():
            if feature.kind is not FeatureKind.DIRECTIONAL or not feature.valid:
                continue
            weight = table.get(name, 0.0)
            if weight <= 0.0:
                continue
            weights[name] = weight
            total += weight
        if total <= 0.0:
            return {}
        self._apply_collinearity_caps(weights, total)
        return weights

    def _apply_collinearity_caps(
        self, weights: dict[str, float], total: float
    ) -> None:
        """Scale down groups of near-duplicate features, in place."""
        for group, cap in self._config.collinear_groups.items():
            members = [name for name in group.split("|") if name in weights]
            if len(members) < 2:
                continue
            group_weight = sum(weights[name] for name in members)
            allowed = cap * total
            if group_weight <= allowed or group_weight <= 0.0:
                continue
            scale = allowed / group_weight
            for name in members:
                weights[name] *= scale


__all__ = ["CompositeScorer"]
