"""Bounded hand-off queue between the feed thread and the engine thread.

The websocket callback runs on the client library's own thread. Doing the feature
computation there would couple socket draining to compute latency and would
eventually apply backpressure to the TCP connection, which for a market data feed
means the exchange's snapshots queue up in the kernel and every one of them
arrives late.

The callback therefore does the minimum — stamp, parse, enqueue — and a dedicated
engine thread consumes the queue.

Overflow policy: **drop the oldest**. For a snapshot feed the newest photograph of
the book strictly dominates an older one; keeping a stale snapshot while
discarding the current one would be the wrong trade in every case. Drops are
counted and reported rather than hidden, because a non-zero drop count is a
capacity signal that the operator must see.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from ..utils.logging_utils import get_logger
from ..utils.types import RawSnapshot

_LOGGER = get_logger(__name__)

#: Sentinel pushed to release a blocked consumer on shutdown.
_SHUTDOWN = object()


class SnapshotQueue:
    """A bounded, drop-oldest queue of :class:`RawSnapshot` records.

    Parameters
    ----------
    maxsize:
        Capacity. Must be positive.
    """

    __slots__ = ("_closed", "_dropped", "_lock", "_queue")

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def dropped(self) -> int:
        """Number of snapshots discarded because the queue was full."""
        with self._lock:
            return self._dropped

    @property
    def depth(self) -> int:
        """Approximate current queue depth, for monitoring only."""
        return self._queue.qsize()

    def put(self, snapshot: RawSnapshot) -> None:
        """Enqueue a snapshot, discarding the oldest entry when full.

        Never blocks and never raises, because it is called from the library's
        callback thread where an exception would be swallowed or would tear down
        the connection.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            # Not an error condition: a full queue is the documented trigger for
            # the drop-oldest policy below. Handled, counted and reported, never
            # silently ignored.
            self._make_room()
        else:
            return

        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            # Another producer refilled the slot between the two operations. The
            # newest snapshot is dropped instead of blocking the feed thread, and
            # the drop is counted.
            self._record_drop()

    def _make_room(self) -> None:
        """Discard the oldest queued snapshot to make space for a newer one."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            # A consumer drained the queue between the failed put and this get,
            # so space already exists and nothing was dropped.
            return
        total = self._record_drop()
        if total == 1 or total % 1000 == 0:
            _LOGGER.warning(
                "snapshot queue saturated; dropped %d snapshot(s) so far", total
            )

    def _record_drop(self) -> int:
        """Increment and return the drop counter."""
        with self._lock:
            self._dropped += 1
            return self._dropped

    def close(self) -> None:
        """Signal end of stream, releasing any blocked consumer.

        The sentinel is only enqueued when there is room. Making space for it by
        discarding a queued snapshot would lose data that has already been
        accepted, so a full queue simply relies on the consumer's poll timeout to
        observe the closed flag.
        """
        self._closed = True
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            _LOGGER.debug(
                "queue full at close; consumer will observe the closed flag on its "
                "next poll"
            )

    def drain(self, *, timeout: float = 0.25) -> Iterator[RawSnapshot]:
        """Yield snapshots until :meth:`close` is called.

        The timeout exists so that a consumer loop can also check its own stop
        flag; a blocking ``get`` with no timeout cannot be interrupted.
        """
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if item is _SHUTDOWN:
                # Drain whatever is still queued before finishing, so a shutdown
                # never discards snapshots that were already accepted.
                while True:
                    try:
                        remaining = self._queue.get_nowait()
                    except queue.Empty:
                        return
                    if isinstance(remaining, RawSnapshot):
                        yield remaining
            if isinstance(item, RawSnapshot):
                yield item


__all__ = ["SnapshotQueue"]
