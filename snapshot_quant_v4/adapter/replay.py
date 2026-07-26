"""Offline replay source and session recorder.

Replay is not a convenience feature. Without a deterministic source there is no
way to unit test the pipeline, no way to benchmark it honestly, and no way to
reproduce a defect that appeared at 14:37 in a live session. The recorder writes
the engine's own normalised snapshot form, so replaying a recording exercises
exactly the input the engine saw.

Pacing
------
``speed`` controls playback:

*   ``0`` replays as fast as the CPU allows, which is what tests and benchmarks
    want;
*   ``1.0`` reproduces the original inter-snapshot intervals from the exchange
    timestamps, which is what a realistic dry run wants;
*   any other positive value scales those intervals.

Feature values are unaffected by the choice, because every time-aware estimator
is driven by the exchange timestamp inside the snapshot rather than by wall-clock
arrival time. That property is precisely why the engine uses exchange time for
mathematics and monotonic time only for timeouts.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import TextIO

from ..utils.logging_utils import get_logger
from ..utils.types import RawSnapshot
from .parsing import PayloadError, snapshot_from_json, snapshot_to_json

_LOGGER = get_logger(__name__)


class ReplaySource:
    """Replays snapshots from a JSONL file.

    Implements :class:`~snapshot_quant_v4.utils.types.MarketDataSource`.

    Parameters
    ----------
    path:
        JSONL file produced by :class:`SnapshotRecorder`.
    speed:
        Playback speed. ``0`` means unpaced.
    max_snapshots:
        Stop after this many snapshots. ``0`` means no limit.
    strict:
        When ``True`` a malformed record raises; when ``False`` it is counted and
        skipped. Tests use strict mode; an operator replaying a truncated
        recording usually does not.
    """

    __slots__ = ("_emitted", "_errors", "_max_snapshots", "_path", "_speed", "_stop", "_strict")

    def __init__(
        self,
        path: str | Path,
        *,
        speed: float = 0.0,
        max_snapshots: int = 0,
        strict: bool = False,
    ) -> None:
        if speed < 0.0:
            raise ValueError(f"speed must be >= 0, got {speed}")
        if max_snapshots < 0:
            raise ValueError(f"max_snapshots must be >= 0, got {max_snapshots}")
        self._path = Path(path).expanduser()
        self._speed = float(speed)
        self._max_snapshots = int(max_snapshots)
        self._strict = strict
        self._stop = threading.Event()
        self._errors = 0
        self._emitted = 0

    @property
    def errors(self) -> int:
        """Number of records that could not be parsed."""
        return self._errors

    @property
    def emitted(self) -> int:
        """Number of snapshots yielded."""
        return self._emitted

    def start(self) -> None:
        """Verify the file exists before the consumer begins.

        Raises
        ------
        FileNotFoundError
            If the recording is missing. Failing here, rather than inside the
            iterator, gives a clean startup error.
        """
        if not self._path.is_file():
            raise FileNotFoundError(f"replay file not found: {self._path}")
        self._stop.clear()

    def stop(self) -> None:
        """Request that iteration end at the next record."""
        self._stop.set()

    def snapshots(self) -> Iterator[RawSnapshot]:
        """Yield snapshots from the recording."""
        previous_exchange_ms: int | None = None
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if self._stop.is_set():
                    return
                text = line.strip()
                if not text:
                    continue
                snapshot = self._parse_line(text, line_number)
                if snapshot is None:
                    continue
                if self._speed > 0.0 and previous_exchange_ms is not None:
                    self._pace(previous_exchange_ms, snapshot.exchange_timestamp_ms)
                previous_exchange_ms = snapshot.exchange_timestamp_ms
                self._emitted += 1
                yield snapshot
                if self._max_snapshots and self._emitted >= self._max_snapshots:
                    return

    def _parse_line(self, text: str, line_number: int) -> RawSnapshot | None:
        """Parse one JSONL record, honouring the strict flag."""
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            self._handle_error(line_number, f"invalid JSON: {exc.msg}")
            return None
        if not isinstance(record, dict):
            self._handle_error(line_number, "record is not an object")
            return None
        try:
            return snapshot_from_json(
                record, received_monotonic_ms=time.monotonic_ns() / 1_000_000.0
            )
        except PayloadError as exc:
            self._handle_error(line_number, str(exc))
            return None

    def _handle_error(self, line_number: int, detail: str) -> None:
        """Record a malformed-record error, or raise it in strict mode.

        Raises
        ------
        PayloadError
            When ``strict`` is set.
        """
        self._errors += 1
        message = f"{self._path}:{line_number}: {detail}"
        if self._strict:
            raise PayloadError(message)
        if self._errors <= 5:
            _LOGGER.warning("skipping malformed replay record: %s", message)
        return

    def _pace(self, previous_ms: int, current_ms: int) -> None:
        """Sleep to reproduce the original inter-snapshot interval."""
        delta_ms = current_ms - previous_ms
        if delta_ms <= 0:
            return
        delay_s = (delta_ms / 1000.0) / self._speed
        if delay_s > 0.0:
            self._stop.wait(delay_s)


class SnapshotRecorder:
    """Appends snapshots to a JSONL file for later replay.

    Writes are buffered by the file object and flushed every
    ``flush_interval`` records, so recording adds a bounded, predictable cost to
    the engine thread rather than a synchronous write per snapshot.
    """

    __slots__ = ("_flush_interval", "_handle", "_path", "_written")

    def __init__(self, path: str | Path, *, flush_interval: int = 100) -> None:
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        self._path = Path(path).expanduser()
        self._handle: TextIO | None = None
        self._written = 0
        self._flush_interval = int(flush_interval)

    @property
    def written(self) -> int:
        """Number of records written."""
        return self._written

    @property
    def path(self) -> Path:
        """Destination file."""
        return self._path

    def open(self) -> None:
        """Create the destination directory and open the file for appending."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def write(self, snapshot: RawSnapshot) -> None:
        """Append one snapshot.

        Raises
        ------
        RuntimeError
            If the recorder was not opened, which is a programming error rather
            than a runtime condition.
        """
        handle = self._handle
        if handle is None:
            raise RuntimeError("recorder is not open")
        handle.write(snapshot_to_json(snapshot))
        handle.write("\n")
        self._written += 1
        if self._written % self._flush_interval == 0:
            handle.flush()

    def close(self) -> None:
        """Flush and close. Safe to call more than once."""
        handle = self._handle
        if handle is None:
            return
        try:
            handle.flush()
        finally:
            handle.close()
            self._handle = None
            _LOGGER.info("recorded %d snapshot(s) to %s", self._written, self._path)

    def __enter__(self) -> SnapshotRecorder:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["ReplaySource", "SnapshotRecorder"]
