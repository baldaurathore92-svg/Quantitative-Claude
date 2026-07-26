"""Market-data sources.

Every source satisfies
:class:`~snapshot_quant_v4.utils.types.MarketDataSource`, so the runner and the
engine are written against the protocol and never against a specific transport:

``AngelOneAdapter``
    Live SmartAPI V2 SnapQuote feed. Requires ``smartapi-python``, imported
    lazily so that nothing else in the package depends on it.
``ReplaySource``
    Deterministic playback of a recorded JSONL session. The basis for tests,
    benchmarks and defect reproduction.
``SyntheticSource``
    Seeded synthetic book, for demonstrations and benchmarks. Explicitly not
    market data.
``ScenarioSource``
    Exactly reproducible upward/downward streams plus three noise and three
    seeded random pattern families. Also explicitly not market data.

``SnapshotRecorder`` writes the engine's normalised snapshot form, which is what
``ReplaySource`` consumes.
"""

from __future__ import annotations

from .angel_v2 import AdapterUnavailableError, AngelOneAdapter, LoginError
from .parsing import PayloadError, parse_snapshot, snapshot_from_json, snapshot_to_json
from .queueing import SnapshotQueue
from .replay import ReplaySource, SnapshotRecorder
from .scenarios import (
    MarketScenario,
    NoisePattern,
    RandomPattern,
    ScenarioConfig,
    ScenarioSource,
)
from .synthetic import SyntheticConfig, SyntheticSource

__all__ = [
    "AdapterUnavailableError",
    "AngelOneAdapter",
    "LoginError",
    "MarketScenario",
    "NoisePattern",
    "PayloadError",
    "RandomPattern",
    "ReplaySource",
    "ScenarioConfig",
    "ScenarioSource",
    "SnapshotQueue",
    "SnapshotRecorder",
    "SyntheticConfig",
    "SyntheticSource",
    "parse_snapshot",
    "snapshot_from_json",
    "snapshot_to_json",
]
