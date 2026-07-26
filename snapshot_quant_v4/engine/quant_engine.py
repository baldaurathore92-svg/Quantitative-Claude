"""Per-instrument orchestration and the multi-instrument registry.

Pipeline, in the order the stages actually run::

    RawSnapshot
      -> validator                (structural checks, derived fields, gap flag)
      -> shared statistics        (update_pre: price/spread/depth/stability)
      -> features                 (declared dependency order)
      -> regime detector          (needs the momentum sign)
      -> quality gate             (needs the updated statistics)
      -> confidence model         (needs the regime and the quality report)
      -> composite scorer         (needs confidence)
      -> adaptive threshold
      -> state machine            (needs everything above)
      -> shared statistics        (update_post: imbalance dispersion)
      -> EngineOutput

Two ordering decisions are worth stating explicitly.

*   The regime is computed *before* confidence because the regime supplies the
    confidence multiplier, and confidence is computed before the composite
    because the composite weights by confidence. There is no cycle: the regime
    depends only on price statistics and the momentum sign.
*   ``update_post`` records the imbalance *after* the features have run, so the
    dispersion used to normalise a feature never contains the observation being
    normalised.

State ownership
---------------
Every estimator lives inside a :class:`SymbolEngine`, one per token. There is no
module-level mutable state anywhere in the package, so two instruments cannot
contaminate each other and a test can construct an engine, drive it and throw it
away.

Gap handling is centralised here: on any gap the engine closes an open position
through the state machine, resets every estimator it owns, and restarts warmup.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..config import AppConfig, FeaturesConfig, SymbolConfig
from ..features.acceleration import AccelerationFeature
from ..features.base import Feature, FeatureContext
from ..features.depth_slope import DepthSlopeFeature
from ..features.ltp_confirmation import LtpConfirmationFeature
from ..features.microprice import MicropriceFeature
from ..features.momentum import MomentumFeature
from ..features.queue_persistence import QueuePersistenceFeature
from ..features.refill import RefillFeature
from ..features.spread import SpreadFeature
from ..features.spread_compression import SpreadCompressionFeature
from ..features.weighted_obi import WeightedObiFeature
from ..utils.clock import Clock, SystemClock
from ..utils.constants import (
    EPSILON,
    F_ACCELERATION,
    F_DEPTH_SLOPE,
    F_LTP_CONFIRMATION,
    F_MICROPRICE,
    F_MOMENTUM,
    F_QUEUE_PERSISTENCE,
    F_REFILL,
    F_SPREAD,
    F_SPREAD_COMPRESSION,
    F_WEIGHTED_OBI,
)
from ..utils.logging_utils import get_logger
from ..utils.math_utils import sign
from ..utils.types import (
    CompositeResult,
    EngineOutput,
    EngineStats,
    FeatureValue,
    QualityReport,
    RawSnapshot,
    Snapshot,
    TradeState,
)
from .composite import CompositeScorer
from .confidence import ConfidenceModel
from .execution import ExecutionModel
from .quality import MarketQualityFilter
from .regime import RegimeDetector
from .state_machine import TradingStateMachine
from .stats import SharedStatistics
from .threshold import AdaptiveThreshold
from .validator import SnapshotValidator

_LOGGER = get_logger(__name__)

#: Human labels used when explaining a score to an operator.
_REASON_LABELS: Mapping[str, str] = {
    F_MICROPRICE: "Microprice tilt",
    F_WEIGHTED_OBI: "Order-book imbalance",
    F_DEPTH_SLOPE: "Depth asymmetry",
    F_QUEUE_PERSISTENCE: "Queue persistence",
    F_REFILL: "Refill activity",
    F_MOMENTUM: "Pressure momentum",
    F_ACCELERATION: "Pressure acceleration",
    F_LTP_CONFIRMATION: "Traded-price confirmation",
}

#: How many feature contributions to surface in the reason list.
_MAX_REASON_FEATURES = 4
#: Spread-compression value above which the book counts as compressing.
_COMPRESSION_REASON_LEVEL = 0.60
#: Touch-stability value above which the queue counts as stable.
_STABILITY_REASON_LEVEL = 0.65


def build_features(config: FeaturesConfig) -> tuple[Feature, ...]:
    """Construct the default feature pipeline in dependency-safe order.

    Adding a feature means adding one line here plus a weight in the regime
    tables. The returned order is validated by :class:`SymbolEngine`.
    """
    return (
        SpreadFeature(config.spread),
        SpreadCompressionFeature(config.spread_compression),
        MicropriceFeature(config.microprice),
        WeightedObiFeature(config.weighted_obi),
        DepthSlopeFeature(config.depth_slope),
        QueuePersistenceFeature(config.queue_persistence),
        RefillFeature(config.refill),
        MomentumFeature(config.momentum),
        AccelerationFeature(config.acceleration),
        LtpConfirmationFeature(config.ltp_confirmation),
    )


@dataclass(slots=True)
class _Counters:
    """Mutable counters for one instrument. Never used for decisions."""

    accepted: int = 0
    rejected: int = 0
    blocked: int = 0
    gaps: int = 0
    resets: int = 0
    rejects_by_reason: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> EngineStats:
        """Return an immutable view for reporting."""
        return EngineStats(
            accepted=self.accepted,
            rejected=self.rejected,
            blocked=self.blocked,
            gaps=self.gaps,
            resets=self.resets,
            rejects_by_reason=dict(self.rejects_by_reason),
        )


class SymbolEngine:
    """The full pipeline for a single instrument.

    Parameters
    ----------
    symbol:
        Instrument configuration (token, tick size, quantity).
    config:
        Application configuration.
    clock:
        Injected clock. Defaults to :class:`~utils.clock.SystemClock`.
    features:
        Optional feature pipeline override, for tests and research. Defaults to
        :func:`build_features`.
    execution:
        Optional execution model override.
    """

    __slots__ = (
        "_clock",
        "_composite",
        "_confidence",
        "_config",
        "_counters",
        "_execution",
        "_features",
        "_index",
        "_last_gap_monotonic_ms",
        "_pending_gap_reason",
        "_previous_snapshot",
        "_quality",
        "_regime",
        "_state_machine",
        "_stats",
        "_symbol",
        "_threshold",
        "_validator",
    )

    def __init__(
        self,
        symbol: SymbolConfig,
        config: AppConfig,
        *,
        clock: Clock | None = None,
        features: Sequence[Feature] | None = None,
        execution: ExecutionModel | None = None,
    ) -> None:
        self._symbol = symbol
        self._config = config
        self._clock = clock if clock is not None else SystemClock()
        self._features = tuple(
            features if features is not None else build_features(config.features)
        )
        self._validate_feature_order(self._features)

        self._validator = SnapshotValidator(
            symbol.symbol, symbol.tick_size, config.validation
        )
        self._stats = SharedStatistics(
            tick_size=symbol.tick_size,
            confidence=config.confidence,
            regime=config.regime,
            threshold=config.threshold,
            quality=config.quality,
        )
        self._quality = MarketQualityFilter(config.quality)
        self._confidence = ConfidenceModel(config.confidence)
        self._regime = RegimeDetector(config.regime)
        self._composite = CompositeScorer(config.composite)
        self._threshold = AdaptiveThreshold(config.threshold)
        self._execution = (
            execution
            if execution is not None
            else ExecutionModel(config.execution, config.state_machine)
        )
        self._state_machine = TradingStateMachine(
            config.state_machine,
            config.features.ltp_confirmation,
            self._execution,
            self._clock,
            quantity=symbol.quantity,
        )

        self._previous_snapshot: Snapshot | None = None
        self._last_gap_monotonic_ms: float | None = None
        self._pending_gap_reason: str | None = None
        self._counters = _Counters()
        self._index = 0

    # -- introspection ----------------------------------------------------- #

    @property
    def symbol(self) -> SymbolConfig:
        """Instrument configuration."""
        return self._symbol

    @property
    def state(self) -> TradeState:
        """Current state-machine state."""
        return self._state_machine.state

    @property
    def statistics(self) -> EngineStats:
        """Immutable counter snapshot."""
        return self._counters.snapshot()

    @property
    def shared_statistics(self) -> SharedStatistics:
        """Rolling statistics, exposed for tests and diagnostics."""
        return self._stats

    @property
    def execution(self) -> ExecutionModel:
        """The execution model in use."""
        return self._execution

    # -- main entry point -------------------------------------------------- #

    def process(self, raw: RawSnapshot) -> EngineOutput | None:
        """Process one raw snapshot.

        Returns
        -------
        EngineOutput | None
            ``None`` when the snapshot was rejected by the validator. A rejected
            snapshot updates nothing except the counters, which is the point: a
            structurally invalid book must not touch a single estimator.
        """
        started_ns = time.perf_counter_ns()

        result = self._validator.validate(raw)
        if not result.accepted or result.snapshot is None:
            self._counters.rejected += 1
            reason = result.reason.value
            self._counters.rejects_by_reason[reason] = (
                self._counters.rejects_by_reason.get(reason, 0) + 1
            )
            if result.gap_detected:
                self._note_gap(f"validator: {reason}")
            _LOGGER.debug("%s: rejected (%s) %s", self._symbol.symbol, reason, result.detail)
            return None

        snapshot = result.snapshot
        if result.gap_detected:
            self._note_gap("feed gap")
        forced = self._consume_pending_gap(snapshot)

        self._counters.accepted += 1
        self._index += 1

        self._stats.update_pre(snapshot)
        features = self._evaluate_features(snapshot)

        momentum = features.get(F_MOMENTUM)
        # A constant-pressure trend should have zero pressure momentum. Floating-
        # point EMA cancellation can leave ~1e-14 residue; treating that residue
        # as opposition incorrectly turns a clean TREND into PULLBACK.
        momentum_sign = (
            sign(momentum.value, deadband=EPSILON)
            if momentum is not None and momentum.valid
            else 0
        )
        regime = self._regime.update(self._stats, momentum_sign=momentum_sign)

        quality = self._quality.evaluate(
            snapshot, self._stats, ms_since_gap=self._ms_since_gap()
        )
        if not quality.tradable:
            self._counters.blocked += 1

        self._confidence.apply(
            features,
            spread_ticks=snapshot.spread_ticks,
            stats=self._stats,
            quality=quality,
            regime=regime.regime,
        )
        composite = self._composite.score(
            features, regime=regime.regime, dt_ms=snapshot.dt_ms
        )
        threshold = self._threshold.compute(self._stats)

        machine = self._state_machine.update(
            snapshot=snapshot,
            composite=composite,
            threshold=threshold,
            quality=quality,
            features=features,
        )

        imbalance = features.get(F_WEIGHTED_OBI)
        if imbalance is not None and imbalance.valid:
            self._stats.update_post(imbalance.value)

        self._previous_snapshot = snapshot
        # The decision path ends here. ``compute_us`` measures exactly that, and
        # deliberately excludes the operator-facing explanation below: reason
        # strings are presentation, they cannot influence any decision, and
        # including them would make the latency figure describe formatting rather
        # than computation.
        compute_us = (time.perf_counter_ns() - started_ns) / 1000.0
        reasons = self._build_reasons(
            features=features,
            composite=composite,
            quality=quality,
            machine_reasons=machine.reasons,
            forced_flat=forced,
        )

        return EngineOutput(
            symbol=snapshot.symbol,
            token=snapshot.token,
            exchange_timestamp_ms=snapshot.exchange_timestamp_ms,
            wall_ms=self._clock.wall_ms(),
            snapshot=snapshot,
            features=features,
            composite=composite,
            regime=regime,
            threshold=threshold,
            quality=quality,
            state=machine.state,
            transition=machine.transition,
            position=machine.position,
            pnl=machine.pnl,
            entry_quote=machine.entry_quote,
            exit_quote=machine.exit_quote,
            reasons=reasons,
            compute_us=compute_us,
            snapshot_index=self._index,
        )

    # -- feature evaluation ------------------------------------------------ #

    def _evaluate_features(self, snapshot: Snapshot) -> dict[str, FeatureValue]:
        """Run the feature pipeline for one snapshot."""
        computed: dict[str, FeatureValue] = {}
        context = FeatureContext(
            snapshot=snapshot,
            previous=self._previous_snapshot,
            stats=self._stats,
            computed=computed,
        )
        for feature in self._features:
            computed[feature.name] = feature.compute(context)
        return computed

    @staticmethod
    def _validate_feature_order(features: Sequence[Feature]) -> None:
        """Fail fast when a feature is scheduled before one it depends on.

        Checked once at construction rather than per snapshot, so the hot path
        pays nothing for the guarantee.
        """
        seen: set[str] = set()
        for feature in features:
            if not feature.name:
                raise ValueError(f"{type(feature).__name__} has no name")
            if feature.name in seen:
                raise ValueError(f"duplicate feature name: {feature.name}")
            missing = [dep for dep in feature.depends_on if dep not in seen]
            if missing:
                raise ValueError(
                    f"feature {feature.name!r} depends on {missing} which are not "
                    "evaluated before it"
                )
            seen.add(feature.name)

    # -- gap handling ------------------------------------------------------ #

    def notify_gap(self, reason: str) -> None:
        """Record an externally detected discontinuity, such as a reconnect.

        Thread safety: this is called from the feed thread while the engine thread
        may be inside :meth:`process`. It only sets a flag and a timestamp, both
        single-word stores, and the flag is consumed by the engine thread at a
        well-defined point (the start of the next accepted snapshot). No lock is
        needed and none is taken, which keeps the feed thread free of any
        possibility of blocking on the engine.
        """
        self._note_gap(reason)

    def _note_gap(self, reason: str) -> None:
        """Record a gap and schedule the reset for the next usable snapshot."""
        self._counters.gaps += 1
        self._last_gap_monotonic_ms = self._clock.monotonic_ms()
        self._pending_gap_reason = reason
        _LOGGER.warning("%s: %s; state will be reset", self._symbol.symbol, reason)

    def _consume_pending_gap(self, snapshot: Snapshot) -> bool:
        """Apply a pending reset now that a usable snapshot is available.

        The reset is deferred to the first accepted snapshot because closing a
        position requires a book to price the exit against.
        """
        reason = self._pending_gap_reason
        if reason is None:
            return False
        self._pending_gap_reason = None
        self._state_machine.force_flat(snapshot, reason)
        self._reset_estimators()
        self._counters.resets += 1
        return True

    def _reset_estimators(self) -> None:
        """Clear every rolling estimator owned by this engine."""
        self._stats.reset()
        self._regime.reset()
        self._composite.reset()
        for feature in self._features:
            feature.reset()
        self._previous_snapshot = None
        self._validator.reset()

    def _ms_since_gap(self) -> float | None:
        """Milliseconds since the last gap, or ``None`` if there was none."""
        if self._last_gap_monotonic_ms is None:
            return None
        return self._clock.monotonic_ms() - self._last_gap_monotonic_ms

    # -- explanations ------------------------------------------------------ #

    def _build_reasons(
        self,
        *,
        features: Mapping[str, FeatureValue],
        composite: CompositeResult,
        quality: QualityReport,
        machine_reasons: tuple[str, ...],
        forced_flat: bool,
    ) -> tuple[str, ...]:
        """Assemble the operator-facing explanation for this snapshot.

        Reasons are generated from the same numbers that drove the decision — the
        realised contributions, not a parallel narrative — so the display can
        never disagree with the score.
        """
        reasons: list[str] = []
        if forced_flat:
            reasons.append("! state reset after feed gap")

        contributions = composite.contributions
        ranked = sorted(
            contributions, key=lambda item: abs(item.contribution), reverse=True
        )
        for item in ranked[:_MAX_REASON_FEATURES]:
            if item.contribution == 0.0:
                continue
            label = _REASON_LABELS.get(item.name, item.name)
            marker = "+" if item.contribution > 0.0 else "-"
            reasons.append(
                f"{marker} {label} {item.value:+.2f} "
                f"(w={item.weight:.2f} c={item.confidence:.2f})"
            )

        compression = features.get(F_SPREAD_COMPRESSION)
        if (
            compression is not None
            and compression.valid
            and compression.value >= _COMPRESSION_REASON_LEVEL
        ):
            reasons.append(f"+ Spread compressing ({compression.raw:+.2f}t)")

        spread = features.get(F_SPREAD)
        if (
            spread is not None
            and spread.valid
            and spread.raw > self._config.quality.max_signal_spread_ticks
        ):
            reasons.append(f"- Spread wide ({spread.raw:.2f}t)")

        if self._stats.touch_stability >= _STABILITY_REASON_LEVEL:
            reasons.append(f"+ Queue stable ({self._stats.touch_stability:.2f})")

        reasons.extend(f"- Blocked: {block.value}" for block in quality.reasons)

        reasons.extend(machine_reasons)
        return tuple(reasons)


class EngineRegistry:
    """Owns one :class:`SymbolEngine` per subscribed token.

    Routing by token is the only multi-instrument logic in the engine; all state
    stays inside the per-symbol engines, which is what keeps instruments
    independent.

    Parameters
    ----------
    config:
        Application configuration; ``config.symbols`` defines the instruments.
    clock:
        Injected clock, shared by all instruments.
    """

    __slots__ = ("_clock", "_config", "_engines", "_unknown_tokens")

    def __init__(self, config: AppConfig, *, clock: Clock | None = None) -> None:
        if not config.symbols:
            raise ValueError("configuration defines no symbols to subscribe")
        self._config = config
        self._clock = clock if clock is not None else SystemClock()
        self._engines: dict[str, SymbolEngine] = {
            symbol.token: SymbolEngine(symbol, config, clock=self._clock)
            for symbol in config.symbols
        }
        self._unknown_tokens: dict[str, int] = {}

    @property
    def engines(self) -> Mapping[str, SymbolEngine]:
        """Read-only view of the per-token engines."""
        return self._engines

    def process(self, raw: RawSnapshot) -> EngineOutput | None:
        """Route a snapshot to its instrument engine.

        An unknown token is counted and logged once rather than raising: a feed
        can legitimately deliver a token that was unsubscribed a moment earlier,
        and that must not stop the process.
        """
        engine = self._engines.get(raw.token)
        if engine is None:
            seen = self._unknown_tokens.get(raw.token, 0)
            self._unknown_tokens[raw.token] = seen + 1
            if seen == 0:
                _LOGGER.warning("received snapshot for unsubscribed token %s", raw.token)
            return None
        return engine.process(raw)

    def notify_gap(self, reason: str) -> None:
        """Propagate a feed-wide discontinuity to every instrument."""
        for engine in self._engines.values():
            engine.notify_gap(reason)

    def statistics(self) -> Mapping[str, EngineStats]:
        """Per-token counter snapshots."""
        return {token: engine.statistics for token, engine in self._engines.items()}

    @property
    def unknown_tokens(self) -> Mapping[str, int]:
        """Counts of snapshots received for unsubscribed tokens."""
        return dict(self._unknown_tokens)


__all__ = ["EngineRegistry", "SymbolEngine", "build_features"]
