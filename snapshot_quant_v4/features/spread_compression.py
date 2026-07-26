"""Spread compression.

A narrowing spread means market makers are competing more aggressively for the
touch, which raises the reliability of every level-based feature and often
precedes a directional resolution. A widening spread means the opposite.

Compression is measured as ``slow_ema(spread) - fast_ema(spread)`` in ticks, so a
positive value means the spread is currently tighter than its recent baseline.
Both estimators are time-aware: a sample-count EMA would represent a different
number of seconds depending on how busy the feed is, and compression is
inherently a statement about time.

Like :mod:`features.spread`, this is a *quality* feature. Compression tells you
how much to trust a directional reading and, empirically, when a resolution is
near; it does not tell you which way. It is therefore mapped into ``[0, 1]`` with
0.5 as the neutral point and consumed by the confidence model and the console
reasons, never added to the signed score.
"""

from __future__ import annotations

from ..buffers.rolling_ema import EMAPair
from ..config import SpreadCompressionConfig
from ..utils.constants import F_SPREAD_COMPRESSION
from ..utils.math_utils import clamp, tanh_scale
from ..utils.types import FeatureKind, FeatureValue
from .base import Feature, FeatureContext


class SpreadCompressionFeature(Feature):
    """Detects a shrinking spread relative to its own recent baseline."""

    name = F_SPREAD_COMPRESSION
    kind = FeatureKind.QUALITY

    __slots__ = ("_config", "_emas")

    def __init__(self, config: SpreadCompressionConfig) -> None:
        self._config = config
        self._emas = EMAPair(
            config.fast_half_life_ms,
            config.slow_half_life_ms,
            max_dt_ms=config.max_dt_ms,
            warmup_updates=4,
        )

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return compression mapped into ``[0, 1]`` with 0.5 as neutral."""
        snapshot = context.snapshot
        self._emas.update(snapshot.spread_ticks, snapshot.dt_ms)
        if not self._emas.ready:
            return self._invalid("spread EMAs still warming up")

        # ``EMAPair.spread`` is fast - slow; compression is the negation, so that
        # a positive number always means "tighter than baseline".
        compression_ticks = -self._emas.spread
        scaled = tanh_scale(compression_ticks, self._config.scale_ticks)
        value = clamp(0.5 * (1.0 + scaled), 0.0, 1.0)

        return self._value(
            raw=compression_ticks,
            value=value,
            detail=(
                f"compression={compression_ticks:+.2f}t "
                f"fast={self._emas.fast.value:.2f}t slow={self._emas.slow.value:.2f}t"
            ),
        )

    def reset(self) -> None:
        """Clear both spread EMAs."""
        self._emas.reset()


__all__ = ["SpreadCompressionFeature"]
