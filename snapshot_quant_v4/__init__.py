"""Snapshot Quant Engine V4.

A deterministic, snapshot-only quantitative engine for the NSE cash market,
built strictly on top of Angel One SmartAPI V2 SnapQuote (subscription mode 3).

Scope and honesty about the data
--------------------------------
A retail snapshot feed publishes a periodic photograph of the top of the book.
It does **not** publish the order event stream. The following quantities are
therefore not computable and are not attempted anywhere in this package:

*   aggressor side of a trade, and any ratio derived from it;
*   exact iceberg or spoof detection, both of which require order identities and
    cancellation events;
*   full order-book imbalance at per-level granularity beyond the five
    published levels;
*   queue position of a hypothetical order.

For each of those, the engine implements a clearly named, documented
*approximation* instead, or omits the concept entirely. Every approximation
carries a docstring that states what it can and cannot detect.

Package layout
--------------
``utils``
    Constants, immutable data model, clock, maths, logging.
``buffers``
    O(1) incremental rolling estimators.
``features``
    One module per feature. No feature imports the adapter or the engine.
``engine``
    Validation, quality gating, confidence, regime, composite, threshold, state
    machine, execution model, orchestration.
``adapter``
    SmartAPI V2 websocket source and an offline replay source.
``output``
    Console rendering, decoupled from the compute path.
"""

from __future__ import annotations

__version__ = "4.0.0"

__all__ = ["__version__"]
