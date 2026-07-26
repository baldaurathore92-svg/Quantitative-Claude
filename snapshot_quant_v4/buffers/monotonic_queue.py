"""Sliding-window extrema in amortised O(1) via monotonic deques.

The naive rolling minimum rescans the window on every update, which is O(N) per
snapshot and is exactly the pattern the performance requirements forbid.

The monotonic-deque algorithm keeps, for the maximum, a deque of candidate
samples in strictly decreasing order:

*   On insert, every tail element that is not greater than the new sample can
    never be the maximum again (the new sample is both larger and newer), so it
    is discarded.
*   The head is therefore always the window maximum.
*   Expiry is by sequence index: the head is dropped once its index falls out of
    the window.

Each sample is pushed and popped at most once, so N updates cost O(N) total —
amortised O(1) per update. The minimum uses the mirrored comparison.
"""

from __future__ import annotations

from collections import deque


class MonotonicWindow:
    """Maintains the minimum and maximum of the last ``window`` samples.

    Parameters
    ----------
    window:
        Window length in samples.
    """

    __slots__ = ("_count", "_index", "_max_deque", "_min_deque", "_window")

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self._window = int(window)
        # Each entry is (sequence_index, value).
        self._max_deque: deque[tuple[int, float]] = deque()
        self._min_deque: deque[tuple[int, float]] = deque()
        self._index = 0
        self._count = 0

    # -- state ------------------------------------------------------------- #

    def __len__(self) -> int:
        return self._count

    @property
    def window(self) -> int:
        """Configured window length."""
        return self._window

    @property
    def ready(self) -> bool:
        """``True`` once at least one sample is present."""
        return self._count > 0

    @property
    def full(self) -> bool:
        """``True`` once the window is completely populated."""
        return self._count >= self._window

    @property
    def maximum(self) -> float:
        """Window maximum.

        Raises
        ------
        IndexError
            If no samples are present.
        """
        if not self._max_deque:
            raise IndexError("MonotonicWindow is empty")
        return self._max_deque[0][1]

    @property
    def minimum(self) -> float:
        """Window minimum.

        Raises
        ------
        IndexError
            If no samples are present.
        """
        if not self._min_deque:
            raise IndexError("MonotonicWindow is empty")
        return self._min_deque[0][1]

    @property
    def range(self) -> float:
        """``maximum - minimum``, or ``0.0`` when empty."""
        if not self._max_deque:
            return 0.0
        return self._max_deque[0][1] - self._min_deque[0][1]

    def position_of(self, value: float) -> float:
        """Where ``value`` sits inside the window range, in ``[0, 1]``.

        Returns ``0.5`` for a degenerate (zero-width) window so that callers do
        not have to special-case a flat market.
        """
        if not self._max_deque:
            return 0.5
        low = self._min_deque[0][1]
        high = self._max_deque[0][1]
        span = high - low
        if span <= 0.0:
            return 0.5
        position = (value - low) / span
        if position < 0.0:
            return 0.0
        if position > 1.0:
            return 1.0
        return position

    # -- mutation ---------------------------------------------------------- #

    def update(self, sample: float) -> None:
        """Push ``sample`` and expire samples that left the window."""
        index = self._index
        self._index += 1
        if self._count < self._window:
            self._count += 1

        max_deque = self._max_deque
        while max_deque and max_deque[-1][1] <= sample:
            max_deque.pop()
        max_deque.append((index, sample))

        min_deque = self._min_deque
        while min_deque and min_deque[-1][1] >= sample:
            min_deque.pop()
        min_deque.append((index, sample))

        expiry = index - self._window
        if max_deque[0][0] <= expiry:
            max_deque.popleft()
        if min_deque[0][0] <= expiry:
            min_deque.popleft()

    def reset(self) -> None:
        """Clear all state."""
        self._max_deque.clear()
        self._min_deque.clear()
        self._index = 0
        self._count = 0


__all__ = ["MonotonicWindow"]
