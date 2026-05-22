"""Token-bucket rate limiter for the news HTTP adapter. See Plan 3 §T20a."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Thread-safe token bucket for HTTP rate limiting.

    rate=4, per_seconds=60 yields a 4-token bucket that refills at 4/60 tokens
    per second (one token every 15s). acquire() blocks (via sleep_fn) until at
    least one token is available, then deducts one.

    Inject clock_fn and sleep_fn so tests run instantly and deterministically.
    """

    def __init__(
        self,
        *,
        rate: int,
        per_seconds: float,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = rate
        self._per_seconds = per_seconds
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._tokens: float = float(rate)  # start full
        self._last_checked: float = clock_fn()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        with self._lock:
            now = self._clock_fn()
            elapsed = now - self._last_checked
            self._tokens = min(
                float(self._rate),
                self._tokens + elapsed * self._rate / self._per_seconds,
            )
            self._last_checked = now

            if self._tokens < 1.0:
                # Time needed to accumulate one full token
                wait = (1.0 - self._tokens) * self._per_seconds / self._rate
                self._sleep_fn(wait)
                self._tokens = 0.0
                self._last_checked = self._clock_fn()
            else:
                self._tokens -= 1.0
