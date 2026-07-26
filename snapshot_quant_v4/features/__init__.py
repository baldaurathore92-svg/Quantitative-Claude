"""Feature implementations, one concern per module.

Every feature in this package is a pure transformation of snapshots plus its own
private, resettable estimators. None of them can see the websocket, the state
machine or another feature's internals; cross-feature reads go through the
declared ``depends_on`` mechanism in :mod:`features.base`.

Directional features (they carry a sign and enter the composite score):

*   :class:`~features.microprice.MicropriceFeature`
*   :class:`~features.weighted_obi.WeightedObiFeature`
*   :class:`~features.depth_slope.DepthSlopeFeature`
*   :class:`~features.queue_persistence.QueuePersistenceFeature`
*   :class:`~features.refill.RefillFeature`
*   :class:`~features.momentum.MomentumFeature`
*   :class:`~features.acceleration.AccelerationFeature`
*   :class:`~features.ltp_confirmation.LtpConfirmationFeature`

Quality features (magnitude without direction; they feed confidence, never the
score):

*   :class:`~features.spread.SpreadFeature`
*   :class:`~features.spread_compression.SpreadCompressionFeature`

To add a feature: implement :class:`~features.base.Feature`, add its name to
``utils.constants``, register it in ``engine.quant_engine.build_features`` and
give it a weight in ``config.DEFAULT_REGIME_WEIGHTS``. Nothing else changes.
"""

from __future__ import annotations

from .acceleration import AccelerationFeature
from .base import Feature, FeatureContext
from .depth_slope import DepthSlopeFeature
from .ltp_confirmation import LtpConfirmationFeature
from .microprice import MicropriceFeature
from .momentum import MomentumFeature
from .queue_persistence import QueuePersistenceFeature
from .refill import RefillFeature
from .spread import SpreadFeature
from .spread_compression import SpreadCompressionFeature
from .weighted_obi import WeightedObiFeature

__all__ = [
    "AccelerationFeature",
    "DepthSlopeFeature",
    "Feature",
    "FeatureContext",
    "LtpConfirmationFeature",
    "MicropriceFeature",
    "MomentumFeature",
    "QueuePersistenceFeature",
    "RefillFeature",
    "SpreadCompressionFeature",
    "SpreadFeature",
    "WeightedObiFeature",
]
