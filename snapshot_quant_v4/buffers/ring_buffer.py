"""Fixed-capacity ring buffer of floats.

The buffer is the substrate for every rolling statistic in the engine. Its
contract is deliberately narrow:

*   :meth:`RingBuffer.push` is O(1) and returns the evicted value, which is
    exactly what an incremental sum/sum-of-squares estimator needs. This is why
    the engine never needs ``del values[:index]`` or a ``list(buffer)`` copy.
*   Random access is by *age*: ``at(0)`` is the newest sample. Callers never
    need to reason about the physical write cursor.
*   Iteration exists (:meth:`iter_values`) but is documented as an
    O(N) operation reserved for periodic drift correction and for tests. It is
    never called on the per-snapshot path.

``array('d')`` is used rather than a Python list so the storage is a flat block
of C doubles: no per-element object, no reference counting during eviction.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator


class RingBuffer:
    """A pre-allocated circular buffer of ``float`` values.

    Parameters
    ----------
    capacity:
        Maximum number of samples retained. Must be positive.

    Raises
    ------
    ValueError
        If ``capacity`` is not positive.
    """

    __slots__ = ("_capacity", "_count", "_data", "_index")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._data = array("d", [0.0]) * self._capacity
        self._index = 0
        self._count = 0

    # -- introspection ----------------------------------------------------- #

    @property
    def capacity(self) -> int:
        """Maximum number of retained samples."""
        return self._capacity

    def __len__(self) -> int:
        """Number of samples currently stored."""
        return self._count

    @property
    def full(self) -> bool:
        """``True`` once ``capacity`` samples have been pushed."""
        return self._count == self._capacity

    @property
    def fill_ratio(self) -> float:
        """Fraction of the window that is populated, in ``[0, 1]``."""
        return self._count / self._capacity

    # -- mutation ---------------------------------------------------------- #

    def push(self, value: float) -> float | None:
        """Append ``value``, returning the evicted sample if the buffer was full.

        Returns
        -------
        float | None
            The value that just left the window, or ``None`` while the window
            is still filling.
        """
        evicted: float | None = None
        if self._count == self._capacity:
            evicted = self._data[self._index]
        else:
            self._count += 1
        self._data[self._index] = value
        self._index += 1
        if self._index == self._capacity:
            self._index = 0
        return evicted

    def clear(self) -> None:
        """Drop every sample. Used on feed gaps and reconnects."""
        self._index = 0
        self._count = 0

    # -- access ------------------------------------------------------------ #

    def at(self, age: int) -> float:
        """Return the sample ``age`` positions behind the newest one.

        ``at(0)`` is the newest sample, ``at(1)`` the one before it.

        Raises
        ------
        IndexError
            If ``age`` is negative or not yet populated.
        """
        if age < 0 or age >= self._count:
            raise IndexError(f"age {age} out of range for {self._count} samples")
        position = self._index - 1 - age
        if position < 0:
            position += self._capacity
        return self._data[position]

    @property
    def newest(self) -> float:
        """The most recently pushed sample.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        return self.at(0)

    @property
    def oldest(self) -> float:
        """The oldest sample still inside the window.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        return self.at(self._count - 1)

    def iter_values(self) -> Iterator[float]:
        """Iterate from oldest to newest.

        O(N). Reserved for periodic drift correction and tests; never called
        per snapshot.
        """
        start = self._index - self._count
        if start < 0:
            start += self._capacity
        for offset in range(self._count):
            position = start + offset
            if position >= self._capacity:
                position -= self._capacity
            yield self._data[position]


__all__ = ["RingBuffer"]
