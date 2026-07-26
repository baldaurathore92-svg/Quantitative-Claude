"""Queue persistence: survival of a *price level* between snapshots.

What this can and cannot be
---------------------------
A snapshot feed publishes no order identities and no queue positions, so the
position of a hypothetical order in the queue is unknowable and is not estimated
here. What *is* observable is whether the price level that was at the touch one
snapshot ago still exists, and whether it still holds comparable size. That is a
level-survival statistic, and it is what this feature reports.

Price-keyed comparison
----------------------
The comparison is keyed on **price**, never on ladder index. Comparing
``bids[0]`` across two snapshots is one of the most common defects in retail
order-book code: when the best bid improves by a tick, index zero refers to a
different price, and the apparent quantity collapse is an artefact of the
re-indexing rather than a real event. This feature looks up the *previous best
price* in the current ladder using
:meth:`~snapshot_quant_v4.utils.types.Snapshot.level_at` and compares like with
like. A level counts as persisted when it is still present and retains at least
``1 - tolerance`` of its previous quantity.

The directional value is ``persistence_bid - persistence_ask``: a bid side that
holds its levels while the ask side keeps vanishing is a book in which downside
liquidity is sticky and upside liquidity is not.
"""

from __future__ import annotations

from ..buffers.rolling_mean import RollingMean
from ..config import QueuePersistenceConfig
from ..utils.constants import F_QUEUE_PERSISTENCE
from ..utils.math_utils import clamp_unit
from ..utils.types import FeatureKind, FeatureValue, Side, Snapshot
from .base import Feature, FeatureContext


class QueuePersistenceFeature(Feature):
    """Survival rate of the touch price level, per side."""

    name = F_QUEUE_PERSISTENCE
    kind = FeatureKind.DIRECTIONAL

    __slots__ = ("_ask_survival", "_bid_survival", "_config")

    def __init__(self, config: QueuePersistenceConfig) -> None:
        self._config = config
        self._bid_survival = RollingMean(config.window, min_samples=config.min_samples)
        self._ask_survival = RollingMean(config.window, min_samples=config.min_samples)

    def compute(self, context: FeatureContext) -> FeatureValue:
        """Return the bid-minus-ask level survival asymmetry."""
        previous = context.previous
        if previous is None:
            return self._invalid("no previous snapshot")

        snapshot = context.snapshot
        self._bid_survival.update(self._survived(snapshot, previous, Side.BID))
        self._ask_survival.update(self._survived(snapshot, previous, Side.ASK))

        if not (self._bid_survival.ready and self._ask_survival.ready):
            return self._invalid("survival window still filling")

        bid_persistence = self._bid_survival.value
        ask_persistence = self._ask_survival.value
        value = clamp_unit(bid_persistence - ask_persistence)

        return self._value(
            raw=bid_persistence - ask_persistence,
            value=value,
            detail=f"bid={bid_persistence:.2f} ask={ask_persistence:.2f}",
        )

    def reset(self) -> None:
        """Clear both survival windows."""
        self._bid_survival.reset()
        self._ask_survival.reset()

    # -- components -------------------------------------------------------- #

    def _survived(self, snapshot: Snapshot, previous: Snapshot, side: Side) -> float:
        """Return ``1.0`` when the previous touch level is still intact.

        The previous best *price* is looked up in the current ladder. Three
        outcomes are possible and all three are meaningful:

        *   the price is still present with comparable size -> survived;
        *   the price is present but has lost more than ``tolerance`` of its
            size -> not survived;
        *   the price is absent from the visible ladder -> not survived. Note
            that a level can also leave the visible window because four better
            prices appeared in front of it; that is rare within a five-level
            window and is treated as non-survival, which is the conservative
            direction.
        """
        previous_level = previous.best_bid if side is Side.BID else previous.best_ask
        if previous_level.quantity <= 0:
            return 0.0
        current_level = snapshot.level_at(side, previous_level.price_paise)
        if current_level is None:
            return 0.0
        threshold = previous_level.quantity * (1.0 - self._config.tolerance)
        return 1.0 if current_level.quantity >= threshold else 0.0


__all__ = ["QueuePersistenceFeature"]
