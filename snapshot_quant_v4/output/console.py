"""Console renderers.

Three implementations satisfy
:class:`~snapshot_quant_v4.utils.types.Renderer`:

``RichRenderer``
    Uses ``rich`` when it is installed: a live-updating table with colour.
``PlainRenderer``
    Standard-library ANSI renderer with the same content and colours. This is not
    a degraded stub; it is a complete implementation, because a trading process
    must not depend on an optional presentation library being present.
``NullRenderer``
    Draws nothing, for headless runs and benchmarks.

:func:`create_renderer` selects one from configuration, defaulting to ``rich``
when available and ANSI otherwise.

Both visible renderers draw at the runner's frame rate, never per snapshot. A
terminal write costs on the order of milliseconds; performing one inside the
per-snapshot path would make presentation, not computation, the dominant term in
the latency budget.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any, Final, TextIO

from ..utils.logging_utils import get_logger
from ..utils.types import EngineOutput
from .views import (
    COLOUR_LONG,
    COLOUR_MUTED,
    COLOUR_NEUTRAL,
    COLOUR_SHORT,
    COLOUR_WARN,
    SymbolView,
    build_view,
)

_LOGGER = get_logger(__name__)

_ANSI_CODES: Final[dict[str, str]] = {
    COLOUR_NEUTRAL: "\x1b[36m",
    COLOUR_LONG: "\x1b[32m",
    COLOUR_SHORT: "\x1b[31m",
    COLOUR_WARN: "\x1b[33m",
    COLOUR_MUTED: "\x1b[90m",
}
_ANSI_RESET: Final[str] = "\x1b[0m"
_ANSI_BOLD: Final[str] = "\x1b[1m"
_ANSI_CLEAR_HOME: Final[str] = "\x1b[2J\x1b[H"
_ANSI_HIDE_CURSOR: Final[str] = "\x1b[?25l"
_ANSI_SHOW_CURSOR: Final[str] = "\x1b[?25h"

_RICH_STYLES: Final[dict[str, str]] = {
    COLOUR_NEUTRAL: "cyan",
    COLOUR_LONG: "bold green",
    COLOUR_SHORT: "bold red",
    COLOUR_WARN: "yellow",
    COLOUR_MUTED: "grey62",
}

_TITLE: Final[str] = "Snapshot Quant Engine V4 - SmartAPI SnapQuote (mode 3)"
#: Feature rows shown per instrument.
_MAX_FEATURE_ROWS: Final[int] = 6
#: Reason lines shown per instrument.
_MAX_REASON_ROWS: Final[int] = 6


class PlainRenderer:
    """ANSI console renderer built only on the standard library.

    Parameters
    ----------
    stream:
        Output stream. Defaults to ``stdout``.
    colour:
        Force colour on or off. ``None`` enables colour only for a TTY, so
        redirected output stays clean for later parsing.
    """

    __slots__ = ("_colour", "_started", "_stream")

    def __init__(self, *, stream: TextIO | None = None, colour: bool | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        if colour is None:
            colour = bool(getattr(self._stream, "isatty", lambda: False)())
        self._colour = colour
        self._started = False

    def start(self) -> None:
        """Clear the screen and hide the cursor."""
        self._started = True
        if self._colour:
            self._stream.write(_ANSI_HIDE_CURSOR)
        self._stream.write(_ANSI_CLEAR_HOME)
        self._stream.flush()

    def stop(self) -> None:
        """Restore the cursor."""
        if self._colour and self._started:
            self._stream.write(_ANSI_SHOW_CURSOR)
        self._stream.write("\n")
        self._stream.flush()
        self._started = False

    def render(self, outputs: Iterable[EngineOutput], footer: str = "") -> None:
        """Redraw the whole frame."""
        lines: list[str] = [self._bold(_TITLE), ""]
        for output in outputs:
            lines.extend(self._render_view(build_view(output)))
            lines.append("")
        if footer:
            lines.append(self._paint(footer, COLOUR_MUTED))
        self._stream.write(_ANSI_CLEAR_HOME + "\n".join(lines) + "\n")
        self._stream.flush()

    def _render_view(self, view: SymbolView) -> list[str]:
        """Format one instrument block."""
        state = self._paint(f"{view.state:<12}", view.state_colour)
        lines = [
            f"{self._bold(view.symbol):<18} {view.time}  {state} "
            f"regime {view.regime:<9} conf {view.confidence:>4} "
            f"lat {view.latency:>7}",
            f"  book      {view.book}",
            f"  composite {view.composite_bar}  {view.composite}",
            f"  threshold {view.threshold}",
            f"  quality   {view.quality}",
        ]
        if view.features:
            lines.append("  features")
            lines.extend(
                f"    {name:<20} {value:>7}  {detail}"
                for name, value, detail in view.features[:_MAX_FEATURE_ROWS]
            )
        lines.append(f"  position  {view.position}")
        lines.append(
            f"  exit      {view.exit_price}   "
            f"pnl {self._paint(view.pnl, view.pnl_colour)}"
        )
        if view.reasons:
            lines.append("  reasons")
            lines.extend(
                f"    {self._mark(reason)}"
                for reason in view.reasons[:_MAX_REASON_ROWS]
            )
        return lines

    def _mark(self, reason: str) -> str:
        """Colour a reason line by its leading marker."""
        if reason.startswith("+"):
            return self._paint("+ " + reason[1:].strip(), COLOUR_LONG)
        if reason.startswith("-"):
            return self._paint("- " + reason[1:].strip(), COLOUR_SHORT)
        if reason.startswith("!"):
            return self._paint("! " + reason[1:].strip(), COLOUR_WARN)
        return self._paint(reason, COLOUR_MUTED)

    def _paint(self, text: str, colour: str) -> str:
        """Wrap ``text`` in an ANSI colour when colour is enabled."""
        if not self._colour:
            return text
        code = _ANSI_CODES.get(colour, "")
        return f"{code}{text}{_ANSI_RESET}" if code else text

    def _bold(self, text: str) -> str:
        """Embolden ``text`` when colour is enabled."""
        return f"{_ANSI_BOLD}{text}{_ANSI_RESET}" if self._colour else text


class RichRenderer:
    """Renderer backed by ``rich``.

    Parameters
    ----------
    console:
        A ``rich.console.Console``. Injected rather than constructed so the
        renderer can be tested with a recording console.

    Notes
    -----
    Construct through :func:`create_renderer`, which falls back to
    :class:`PlainRenderer` when ``rich`` is not installed.
    """

    __slots__ = ("_console", "_live", "_panel_factory", "_table_factory")

    def __init__(self, console: Any) -> None:
        from rich.panel import Panel
        from rich.table import Table

        self._console = console
        self._table_factory = Table
        self._panel_factory = Panel
        self._live: Any = None

    def start(self) -> None:
        """Clear the console."""
        self._console.clear()

    def stop(self) -> None:
        """Print a trailing newline so the shell prompt is not overwritten."""
        self._console.print()

    def render(self, outputs: Iterable[EngineOutput], footer: str = "") -> None:
        """Redraw the whole frame as a set of panels."""
        self._console.clear()
        self._console.rule(_TITLE)
        for output in outputs:
            self._console.print(self._panel(build_view(output)))
        if footer:
            self._console.print(footer, style=_RICH_STYLES[COLOUR_MUTED])

    def _panel(self, view: SymbolView) -> Any:
        """Build one instrument panel."""
        table = self._table_factory(show_header=False, box=None, pad_edge=False)
        table.add_column("field", style=_RICH_STYLES[COLOUR_MUTED], no_wrap=True)
        table.add_column("value")

        table.add_row("time", view.time)
        table.add_row(
            "state",
            f"[{_RICH_STYLES[view.state_colour]}]{view.state}[/] "
            f"regime {view.regime} confidence {view.confidence}",
        )
        table.add_row("book", view.book)
        table.add_row("composite", f"{view.composite_bar}  {view.composite}")
        table.add_row("threshold", view.threshold)
        table.add_row("quality", view.quality)
        for name, value, detail in view.features[:_MAX_FEATURE_ROWS]:
            table.add_row(name, f"{value:>7}  {detail}")
        table.add_row("position", view.position)
        table.add_row(
            "exit",
            f"{view.exit_price}   pnl "
            f"[{_RICH_STYLES[view.pnl_colour]}]{view.pnl}[/]",
        )
        for reason in view.reasons[:_MAX_REASON_ROWS]:
            table.add_row("", self._mark(reason))
        table.add_row("latency", view.latency)
        return self._panel_factory(table, title=view.symbol, expand=False)

    @staticmethod
    def _mark(reason: str) -> str:
        """Apply rich markup to a reason line based on its marker."""
        if reason.startswith("+"):
            return f"[{_RICH_STYLES[COLOUR_LONG]}]{reason}[/]"
        if reason.startswith("-"):
            return f"[{_RICH_STYLES[COLOUR_SHORT]}]{reason}[/]"
        if reason.startswith("!"):
            return f"[{_RICH_STYLES[COLOUR_WARN]}]{reason}[/]"
        return f"[{_RICH_STYLES[COLOUR_MUTED]}]{reason}[/]"


class NullRenderer:
    """Renderer that draws nothing. Used for headless runs and benchmarks."""

    __slots__ = ()

    def start(self) -> None:
        """No-op."""

    def render(self, outputs: Iterable[EngineOutput], footer: str = "") -> None:
        """Consume the outputs without drawing.

        The iterable is still drained so that a generator-based caller behaves
        identically with and without a renderer.
        """
        del footer  # part of the Renderer protocol; nothing to draw it on
        for _ in outputs:
            pass

    def stop(self) -> None:
        """No-op."""


def create_renderer(mode: str = "auto", *, stream: TextIO | None = None) -> Any:
    """Construct a renderer from a configuration string.

    Parameters
    ----------
    mode:
        ``auto`` prefers ``rich`` and falls back to ANSI; ``rich`` requires it and
        falls back with a warning; ``plain`` forces ANSI; ``none`` disables
        rendering.
    stream:
        Output stream for the ANSI renderer.

    Raises
    ------
    ValueError
        For an unknown mode.
    """
    if mode not in ("auto", "rich", "plain", "none"):
        raise ValueError(f"unknown renderer mode: {mode!r}")
    if mode == "none":
        return NullRenderer()
    if mode == "plain":
        return PlainRenderer(stream=stream)
    try:
        from rich.console import Console

        return RichRenderer(Console(file=stream))
    except ImportError:
        if mode == "rich":
            _LOGGER.warning(
                "renderer 'rich' requested but the package is not installed; "
                "using the ANSI renderer instead"
            )
        return PlainRenderer(stream=stream)


__all__ = [
    "NullRenderer",
    "PlainRenderer",
    "RichRenderer",
    "create_renderer",
]
