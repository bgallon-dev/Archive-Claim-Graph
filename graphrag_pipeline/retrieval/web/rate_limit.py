"""Simple in-memory per-user token bucket rate limiter.

Designed for a low-concurrency single-process server (solo-operated SaaS).
State is stored in a plain dict in RAM — not persistent across restarts and
not shared between processes.  Sufficient for the intended deployment.

The sliding window evicts expired timestamps on every call, so memory stays
bounded even under long-running processes with many distinct user keys.
"""
from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """Per-key sliding-window rate limiter.

    Parameters
    ----------
    max_calls:
        Number of calls allowed per *period_seconds* (default 20).
    period_seconds:
        Rolling window in seconds (default 60).
    """

    def __init__(self, max_calls: int = 20, period_seconds: float = 60.0) -> None:
        self._max = max_calls
        self._period = period_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Return True if *key* is within the rate limit, False if exceeded."""
        now = time.monotonic()
        cutoff = now - self._period
        with self._lock:
            timestamps = self._buckets.get(key, [])
            # Evict expired timestamps from the sliding window.
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= self._max:
                self._buckets[key] = timestamps
                return False
            timestamps.append(now)
            self._buckets[key] = timestamps
            return True
