"""Clock injection for deterministic eval. See design spec §5.4."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


class ReplayClock:
    def __init__(self, fixed: datetime) -> None:
        if fixed.tzinfo is None:
            raise ValueError("ReplayClock requires timezone-aware datetime")
        self._t = fixed

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: int) -> None:
        self._t = self._t + timedelta(seconds=seconds)

    def set(self, t: datetime) -> None:
        if t.tzinfo is None:
            raise ValueError("ReplayClock.set requires timezone-aware datetime")
        self._t = t
