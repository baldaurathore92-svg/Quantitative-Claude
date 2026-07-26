"""Presentation model: turns an :class:`EngineOutput` into display-ready text.

Both renderers consume this module, so the layout logic exists exactly once. A
renderer decides *how* to draw (rich table versus ANSI text); it never decides
*what* the numbers mean or how they are formatted.

Keeping the formatting out of the engine also keeps it out of the measured
latency path: a view is built at the render frame rate, not once per snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

from ..utils.constants import EXCHANGE_TIMEZONE
from ..utils.types import EngineOutput, TradeState

#: India Standard Time as a fixed offset. Used when the platform has no tz
#: database, which is common in minimal containers. IST has no daylight saving,
#: so the fixed offset is exact rather than an approximation.
_IST_FALLBACK: Final[timezone] = timezone(timedelta(hours=5, minutes=30), "IST")


def _exchange_timezone() -> timezone:
    """Return the exchange timezone, falling back to a fixed IST offset."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(EXCHANGE_TIMEZONE)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - any tzdata failure must fall back cleanly
        return _IST_FALLBACK


_TZ = _exchange_timezone()

#: Semantic colour names, mapped to concrete codes by each renderer.
COLOUR_NEUTRAL: Final[str] = "neutral"
COLOUR_LONG: Final[str] = "long"
COLOUR_SHORT: Final[str] = "short"
COLOUR_WARN: Final[str] = "warn"
COLOUR_MUTED: Final[str] = "muted"

_STATE_COLOURS: Final[dict[TradeState, str]] = {
    TradeState.WARMUP: COLOUR_MUTED,
    TradeState.NEUTRAL: COLOUR_NEUTRAL,
    TradeState.WATCH_LONG: COLOUR_WARN,
    TradeState.WATCH_SHORT: COLOUR_WARN,
    TradeState.LONG: COLOUR_LONG,
    TradeState.SHORT: COLOUR_SHORT,
    TradeState.EXIT_LONG: COLOUR_WARN,
    TradeState.EXIT_SHORT: COLOUR_WARN,
    TradeState.COOLDOWN: COLOUR_MUTED,
}

#: Width of the composite score bar, in characters, per side of centre.
_BAR_HALF_WIDTH: Final[int] = 12


def format_time(output: EngineOutput) -> str:
    """Format the exchange timestamp in exchange-local time.

    The exchange timestamp is shown rather than the local clock: when they
    disagree, the exchange's view is the one that explains the numbers.
    """
    if output.exchange_timestamp_ms <= 0:
        return "--:--:--.---"
    moment = datetime.fromtimestamp(output.exchange_timestamp_ms / 1000.0, tz=_TZ)
    return moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


def score_bar(score: float, threshold: float) -> str:
    """Render a centred bar for a score in ``[-1, +1]`` with threshold markers.

    A bar communicates "how far past the threshold" at a glance, which a bare
    number does not; the ``|`` markers show where the adaptive entry threshold
    currently sits on both sides.
    """
    width = _BAR_HALF_WIDTH
    cells = ["·"] * (width * 2 + 1)
    cells[width] = "┃"

    threshold_offset = min(width, max(1, round(threshold * width)))
    for offset in (width - threshold_offset, width + threshold_offset):
        if 0 <= offset < len(cells):
            cells[offset] = "|"

    magnitude = min(1.0, abs(score))
    filled = round(magnitude * width)
    for step in range(1, filled + 1):
        index = width + step if score >= 0.0 else width - step
        if 0 <= index < len(cells):
            cells[index] = "█"
    return "".join(cells)


@dataclass(frozen=True, slots=True)
class SymbolView:
    """Everything a renderer needs for one instrument, pre-formatted."""

    time: str
    symbol: str
    state: str
    state_colour: str
    regime: str
    composite: str
    composite_bar: str
    threshold: str
    confidence: str
    quality: str
    book: str
    features: tuple[tuple[str, str, str], ...]
    position: str
    exit_price: str
    pnl: str
    pnl_colour: str
    reasons: tuple[str, ...]
    latency: str


def build_view(output: EngineOutput) -> SymbolView:
    """Build the display model for one engine output."""
    snapshot = output.snapshot
    composite = output.composite
    threshold = output.threshold

    features = tuple(
        (
            contribution.name,
            f"{contribution.value:+.2f}",
            f"w{contribution.weight:.2f} c{contribution.confidence:.2f} "
            f"-> {contribution.contribution:+.3f}",
        )
        for contribution in sorted(
            composite.contributions,
            key=lambda item: abs(item.contribution),
            reverse=True,
        )
    )

    position = output.position
    if position is not None and position.is_open:
        position_text = (
            f"{position.direction.name} {position.quantity} @ {position.entry_price:.2f} "
            f"(stop {position.stop_price:.2f} / target {position.target_price:.2f})"
        )
    else:
        position_text = "flat"

    exit_text = (
        f"{output.exit_quote.price:.2f}" if output.exit_quote is not None else "-"
    )
    pnl = output.pnl
    if pnl is None:
        pnl_text = "-"
        pnl_colour = COLOUR_MUTED
    else:
        pnl_text = (
            f"{pnl.gross_ticks:+.1f}t gross {pnl.gross_rupees:+.2f} "
            f"cost {pnl.cost_rupees:.2f} net {pnl.net_rupees:+.2f} "
            f"({pnl.net_bps:+.1f}bps)"
        )
        pnl_colour = COLOUR_LONG if pnl.net_rupees >= 0.0 else COLOUR_SHORT

    blocked = (
        "OK"
        if output.quality.tradable
        else "+".join(reason.value for reason in output.quality.reasons)
    )

    return SymbolView(
        time=format_time(output),
        symbol=output.symbol,
        state=output.state.value,
        state_colour=_STATE_COLOURS.get(output.state, COLOUR_NEUTRAL),
        regime=output.regime.regime.value,
        composite=(
            f"{composite.smoothed:+.3f} (raw {composite.score:+.3f}, "
            f"{composite.used_features} features)"
            if composite.valid
            else "n/a"
        ),
        composite_bar=score_bar(composite.smoothed, threshold.entry),
        threshold=(
            f"entry {threshold.entry:.3f} watch {threshold.watch:.3f} "
            f"exit {threshold.exit:.3f}"
            + (" [floored]" if threshold.floor_applied else "")
        ),
        confidence=f"{composite.confidence * 100.0:.0f}%",
        quality=blocked,
        book=(
            f"{snapshot.best_bid.quantity}@{snapshot.best_bid.price:.2f} / "
            f"{snapshot.best_ask.price:.2f}@{snapshot.best_ask.quantity} "
            f"spread {snapshot.spread_ticks:.1f}t mid {snapshot.mid:.2f}"
        ),
        features=features,
        position=position_text,
        exit_price=exit_text,
        pnl=pnl_text,
        pnl_colour=pnl_colour,
        reasons=output.reasons,
        latency=f"{output.compute_us:.0f}us",
    )


__all__ = [
    "COLOUR_LONG",
    "COLOUR_MUTED",
    "COLOUR_NEUTRAL",
    "COLOUR_SHORT",
    "COLOUR_WARN",
    "SymbolView",
    "build_view",
    "format_time",
    "score_bar",
]
