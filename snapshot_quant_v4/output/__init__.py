"""Presentation layer.

Strictly one-way dependency: this package imports from ``utils`` only. The engine
never imports the renderer, so the decision path cannot be affected by, or
blocked on, terminal I/O.
"""

from __future__ import annotations

from .console import NullRenderer, PlainRenderer, RichRenderer, create_renderer
from .views import SymbolView, build_view, format_time, score_bar

__all__ = [
    "NullRenderer",
    "PlainRenderer",
    "RichRenderer",
    "SymbolView",
    "build_view",
    "create_renderer",
    "format_time",
    "score_bar",
]
