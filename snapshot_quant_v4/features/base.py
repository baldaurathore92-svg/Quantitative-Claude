"""Feature contract.

Every feature is an object with three members: ``name``, ``kind`` and
``compute``. The contract has two properties that matter for this project.

**No feature can reach the network.** A feature receives a
:class:`FeatureContext` — an immutable snapshot, the previous snapshot, the
shared rolling statistics and the already-computed features — and returns a
:class:`~snapshot_quant_v4.utils.types.FeatureValue`. There is no path from a
feature to the adapter, so every feature is unit-testable by constructing two
snapshots by hand.

**Dependencies are declared, not implicit.** ``depends_on`` lists the features
whose values are read from ``context.computed``. The engine verifies at
construction time that the evaluation order satisfies every declaration, so a
future reordering fails loudly at startup instead of silently reading a stale
value from the previous snapshot.

Features own their own estimators. Two consequences are intentional:

*   ``reset()`` must clear them, and the engine calls it on every feed gap.
*   Normalisation dispersion is per-feature, which keeps each feature's scaling
    independent of every other feature's behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..engine.stats import SharedStatistics
from ..utils.types import FeatureKind, FeatureValue, Snapshot


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Everything a feature is allowed to see.

    Attributes
    ----------
    snapshot:
        The validated snapshot being processed.
    previous:
        The previously accepted snapshot, or ``None`` for the first snapshot of
        a sequence (including the first after a gap). Features that need a
        previous state must treat ``None`` as "not yet valid" rather than
        substituting zeros.
    stats:
        Shared rolling statistics, already updated for this snapshot.
    computed:
        Features already evaluated for this snapshot, keyed by name.
    """

    snapshot: Snapshot
    previous: Snapshot | None
    stats: SharedStatistics
    computed: Mapping[str, FeatureValue] = field(default_factory=dict)

    @property
    def dt_ms(self) -> float:
        """Interval since the previous snapshot, in milliseconds."""
        return self.snapshot.dt_ms

    @property
    def tick_size(self) -> float:
        """Instrument tick size in rupees."""
        return self.snapshot.tick_size

    def require(self, name: str) -> FeatureValue:
        """Return a previously computed feature.

        Raises
        ------
        KeyError
            If the feature has not been computed yet, which indicates a
            declared-dependency violation rather than a data problem.
        """
        try:
            return self.computed[name]
        except KeyError as exc:
            raise KeyError(
                f"feature {name!r} was requested before it was computed; "
                "check depends_on and the evaluation order"
            ) from exc


class Feature(ABC):
    """Base class for all features.

    Subclasses must define :attr:`name` and :attr:`kind` as class attributes and
    implement :meth:`compute` and :meth:`reset`.
    """

    #: Canonical feature name; must match the constants in ``utils.constants``.
    name: str = ""
    #: Whether the feature is directional or a quality metric.
    kind: FeatureKind = FeatureKind.DIRECTIONAL
    #: Features whose values are read from ``context.computed``.
    depends_on: tuple[str, ...] = ()

    @abstractmethod
    def compute(self, context: FeatureContext) -> FeatureValue:
        """Evaluate the feature for one snapshot.

        Implementations must be O(1) with respect to history, must not allocate
        unbounded memory, and must return ``valid=False`` rather than a fabricated
        value when their estimators are not yet usable.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state. Called on feed gaps and reconnects."""

    # -- helpers for subclasses -------------------------------------------- #

    def _value(
        self,
        *,
        raw: float,
        value: float,
        valid: bool = True,
        local_confidence: float = 1.0,
        detail: str = "",
    ) -> FeatureValue:
        """Construct a :class:`FeatureValue` carrying this feature's identity."""
        return FeatureValue(
            name=self.name,
            kind=self.kind,
            raw=raw,
            value=value,
            local_confidence=local_confidence,
            valid=valid,
            detail=detail,
        )

    def _invalid(self, reason: str) -> FeatureValue:
        """Construct a neutral, invalid :class:`FeatureValue`.

        An invalid feature is *excluded* from the composite rather than
        contributing zero: contributing zero would drag the normalised score
        towards neutral and would misrepresent an unknown as a balanced book.
        """
        return FeatureValue(
            name=self.name,
            kind=self.kind,
            raw=0.0,
            value=0.0,
            local_confidence=0.0,
            valid=False,
            detail=reason,
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind.value})"


__all__ = ["Feature", "FeatureContext"]
