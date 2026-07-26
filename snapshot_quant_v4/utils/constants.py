"""Immutable, exchange-level constants for the Snapshot Quant Engine.

Nothing in this module may be mutated at runtime. Every value that a strategy
might reasonably want to tune lives in :mod:`snapshot_quant_v4.config`, not
here. Only facts about the exchange / the Angel One SmartAPI wire format
belong in this file.

Design notes
------------
*   **Tick size.** NSE cash market equities trade on a 0.05 rupee grid. All
    price differences produced by the engine are expressed in *ticks* rather
    than rupees so that features are comparable across a 150 rupee stock and
    a 5000 rupee stock. The tick size is still overridable per symbol in the
    configuration because a handful of instruments (and other segments) use a
    different grid.
*   **Paise scaling.** SmartAPI V2 transmits prices as integers scaled by a
    per-segment divisor. For the equity/derivative segments the divisor is
    100 (paise). Currency segments use 10000000. We therefore never hardcode a
    single multiplier anywhere in the feature code; the adapter converts once,
    at the boundary, using :data:`PRICE_DIVISOR_BY_EXCHANGE`.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Final

# --------------------------------------------------------------------------- #
# Segment / wire-format facts
# --------------------------------------------------------------------------- #


class ExchangeType(IntEnum):
    """SmartAPI V2 websocket ``exchange_type`` codes."""

    NSE_CM = 1
    NSE_FO = 2
    BSE_CM = 3
    BSE_FO = 4
    MCX_FO = 5
    NCX_FO = 7
    CDE_FO = 13


class SubscriptionMode(IntEnum):
    """SmartAPI V2 websocket subscription modes."""

    LTP = 1
    QUOTE = 2
    SNAP_QUOTE = 3
    DEPTH_20 = 4


#: Integer divisor that converts a wire price into rupees, per segment.
PRICE_DIVISOR_BY_EXCHANGE: Final[Mapping[int, int]] = MappingProxyType(
    {
        ExchangeType.NSE_CM: 100,
        ExchangeType.BSE_CM: 100,
        ExchangeType.NSE_FO: 100,
        ExchangeType.BSE_FO: 100,
        ExchangeType.MCX_FO: 100,
        ExchangeType.NCX_FO: 100,
        ExchangeType.CDE_FO: 10_000_000,
    }
)

#: Fallback divisor used when a segment is not present in the mapping above.
DEFAULT_PRICE_DIVISOR: Final[int] = 100

#: Default NSE cash-market tick size, in rupees.
DEFAULT_TICK_SIZE: Final[float] = 0.05

#: Number of price levels published by SnapQuote (mode 3) on each side.
SNAPQUOTE_DEPTH_LEVELS: Final[int] = 5

#: Levels that carry the *primary* directional signal. Deeper levels are used
#: only to modulate confidence (see :mod:`features.weighted_obi`).
PRIMARY_DEPTH_LEVELS: Final[int] = 2

#: IANA timezone of the exchange. Used for display and session windows only;
#: all internal arithmetic uses epoch milliseconds.
EXCHANGE_TIMEZONE: Final[str] = "Asia/Kolkata"

# --------------------------------------------------------------------------- #
# Canonical feature names
# --------------------------------------------------------------------------- #

F_MICROPRICE: Final[str] = "microprice"
F_WEIGHTED_OBI: Final[str] = "weighted_obi"
F_DEPTH_SLOPE: Final[str] = "depth_slope"
F_SPREAD: Final[str] = "spread"
F_SPREAD_COMPRESSION: Final[str] = "spread_compression"
F_QUEUE_PERSISTENCE: Final[str] = "queue_persistence"
F_MOMENTUM: Final[str] = "momentum"
F_ACCELERATION: Final[str] = "acceleration"
F_REFILL: Final[str] = "refill"
F_LTP_CONFIRMATION: Final[str] = "ltp_confirmation"

#: Every feature name known to the engine, in dependency-safe evaluation order.
FEATURE_ORDER: Final[tuple[str, ...]] = (
    F_SPREAD,
    F_SPREAD_COMPRESSION,
    F_MICROPRICE,
    F_WEIGHTED_OBI,
    F_DEPTH_SLOPE,
    F_QUEUE_PERSISTENCE,
    F_REFILL,
    F_MOMENTUM,
    F_ACCELERATION,
    F_LTP_CONFIRMATION,
)

# --------------------------------------------------------------------------- #
# Numerical guards
# --------------------------------------------------------------------------- #

#: Smallest denominator tolerated by :func:`utils.math_utils.safe_div`.
EPSILON: Final[float] = 1e-12

#: Natural log of 2, used by the time-aware EMA decay.
LN2: Final[float] = 0.6931471805599453
