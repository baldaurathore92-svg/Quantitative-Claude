"""Small, allocation-free numerical helpers.

Every function here is a pure function of its arguments, is O(1), and performs
no logging or I/O so that it can be called on the hot path. The functions are
deliberately tiny: the goal is that the *same* normalisation logic is used by
every feature instead of each feature inventing its own scaling.
"""

from __future__ import annotations

import math

from .constants import EPSILON, LN2

__all__ = [
    "bps_to_ticks",
    "clamp",
    "clamp_unit",
    "ema_alpha",
    "from_ticks",
    "is_tick_aligned",
    "linear_scale",
    "log_ratio",
    "robust_scale",
    "round_to_tick",
    "safe_div",
    "sign",
    "tanh_scale",
    "ticks_to_bps",
    "to_ticks",
]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``.

    Raises
    ------
    ValueError
        If ``low > high``, which would make the result undefined.
    """
    if low > high:
        raise ValueError(f"clamp bounds inverted: low={low} high={high}")
    if value < low:
        return low
    if value > high:
        return high
    return value


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into the canonical feature range ``[-1, +1]``."""
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is negligible.

    Used instead of bare division so that an empty book or a zero-volume
    interval can never raise ``ZeroDivisionError`` on the hot path.
    """
    if -EPSILON < denominator < EPSILON:
        return default
    return numerator / denominator


def sign(value: float, deadband: float = 0.0) -> int:
    """Return ``-1``, ``0`` or ``+1``.

    ``deadband`` allows callers to treat small magnitudes as flat, which keeps
    the state machine from flip-flopping on numerical dust.
    """
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def to_ticks(price_difference: float, tick_size: float) -> float:
    """Convert a rupee price difference into a (possibly fractional) tick count."""
    return safe_div(price_difference, tick_size)


def from_ticks(ticks: float, tick_size: float) -> float:
    """Convert a tick count back into a rupee price difference."""
    return ticks * tick_size


def bps_to_ticks(bps: float, reference_price: float, tick_size: float) -> float:
    """Convert basis points of ``reference_price`` into ticks.

    This is the only sanctioned way to express an aggression or slippage
    parameter in relative terms: a fixed bps value corresponds to a wildly
    different number of ticks at 150 rupees versus 5000 rupees, so the
    conversion must happen against a live reference price.
    """
    return safe_div(reference_price * bps * 1e-4, tick_size)


def ticks_to_bps(ticks: float, reference_price: float, tick_size: float) -> float:
    """Convert a tick count into basis points of ``reference_price``."""
    return safe_div(ticks * tick_size * 1e4, reference_price)


def ema_alpha(dt_ms: float, half_life_ms: float, max_dt_ms: float) -> float:
    """Return the time-aware EMA smoothing factor for an interval.

    ``alpha = 1 - exp(-ln2 * dt / half_life)``

    Snapshot feeds are event driven, so the interval between two updates is not
    constant. A fixed per-sample ``alpha`` would therefore represent a
    different amount of *time* depending on how busy the book is, making every
    derived feature non-stationary. Deriving ``alpha`` from the elapsed time
    fixes the effective window in seconds instead of in samples.

    ``dt_ms`` is clamped into ``(0, max_dt_ms]`` so that a stalled feed cannot
    produce ``alpha == 1`` (a full reset) and a duplicate timestamp cannot
    produce ``alpha == 0`` (a frozen estimator).
    """
    if half_life_ms <= 0.0:
        raise ValueError("half_life_ms must be positive")
    if max_dt_ms <= 0.0:
        raise ValueError("max_dt_ms must be positive")
    effective = clamp(dt_ms, 0.0, max_dt_ms)
    if effective <= 0.0:
        return 0.0
    return 1.0 - math.exp(-LN2 * effective / half_life_ms)


def tanh_scale(value: float, scale: float) -> float:
    """Map an unbounded value into ``[-1, +1]`` with a soft saturation.

    ``tanh`` is preferred over hard clipping because it is smooth (so the
    composite score does not develop a plateau) and because it preserves the
    ordering of extreme observations instead of collapsing them onto the
    boundary.

    The mathematical range is open, but in binary floating point ``tanh``
    saturates to exactly 1.0 once the argument exceeds roughly 19, so callers
    must treat the bound as inclusive.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return math.tanh(value / scale)


def robust_scale(value: float, dispersion: float, floor: float, k: float = 2.0) -> float:
    """Normalise ``value`` by its own recent dispersion.

    Parameters
    ----------
    value:
        The raw observation.
    dispersion:
        A recent standard deviation of the same series, supplied by the
        caller's own rolling estimator.
    floor:
        Minimum dispersion. Without a floor, a quiet book drives the divisor
        towards zero and every micro-fluctuation saturates the feature.
    k:
        Number of dispersion units that should map to roughly ``tanh(1)``.

    Returns
    -------
    float
        A value in ``(-1, +1)``.
    """
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    return math.tanh(value / (max(dispersion, floor) * k))


def linear_scale(value: float, low: float, high: float) -> float:
    """Map ``[low, high]`` onto ``[0, 1]`` with clamping outside the range."""
    if high <= low:
        raise ValueError(f"linear_scale requires high > low, got {low}..{high}")
    return clamp((value - low) / (high - low), 0.0, 1.0)


def log_ratio(numerator: float, denominator: float, offset: float = 1.0) -> float:
    """Return ``log((numerator + offset) / (denominator + offset))``.

    The offset keeps the ratio finite when a queue empties completely, which is
    a routine event at the touch.
    """
    num = max(numerator, 0.0) + offset
    den = max(denominator, 0.0) + offset
    return math.log(num / den)


def round_to_tick(price: float, tick_size: float) -> float:
    """Round ``price`` onto the exchange tick grid.

    The tick count is computed as an integer and the product is then normalised
    to six decimal places. The normalisation is not cosmetic: ``16001 * 0.05``
    evaluates to ``800.0500000000001``, whereas a price reconstructed from the
    wire integer (``80005 / 100``) is the nearest double to ``800.05``. Without
    normalisation those two representations of the same exchange price compare as
    unequal, and a stop or target sitting exactly on a tick would be missed by a
    hair. Six decimals is far finer than any Indian equity tick and far coarser
    than the error being removed.
    """
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")
    return round(round(price / tick_size) * tick_size, 6)


def is_tick_aligned(price: float, tick_size: float, tolerance: float = 1e-6) -> bool:
    """Return ``True`` when ``price`` sits on the exchange tick grid."""
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")
    ticks = price / tick_size
    return abs(ticks - round(ticks)) <= tolerance
