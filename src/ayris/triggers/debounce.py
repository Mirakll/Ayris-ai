"""Small monotonic guards against duplicate events and trigger storms."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Hashable


class Debouncer:
    """Accept the first occurrence of a key in a configurable time window."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seen: dict[Hashable, float] = {}

    def accept(self, key: Hashable, window: float) -> bool:
        now = self._clock()
        previous = self._seen.get(key)
        if previous is not None and now - previous < max(0.0, window):
            return False
        self._seen[key] = now
        if len(self._seen) > 2048:
            cutoff = now - max(window, 60.0)
            self._seen = {item: stamp for item, stamp in self._seen.items() if stamp >= cutoff}
        return True

    def clear(self) -> None:
        self._seen.clear()


class RateLimiter:
    """Bound launches per key inside a rolling monotonic interval."""

    def __init__(
        self,
        limit: int = 10,
        interval: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = max(1, limit)
        self._interval = max(0.001, interval)
        self._clock = clock
        self._hits: dict[Hashable, deque[float]] = {}

    def accept(self, key: Hashable) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self._interval
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True

    def clear(self) -> None:
        self._hits.clear()
