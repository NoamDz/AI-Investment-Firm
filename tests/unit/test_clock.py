from datetime import datetime, timedelta, timezone

import pytest

from firm.core.clock import Clock, WallClock, ReplayClock


def test_wallclock_returns_utc():
    c: Clock = WallClock()
    t = c.now()
    assert t.tzinfo is not None
    assert t.utcoffset() == timedelta(0)


def test_replayclock_is_fixed_until_advanced():
    fixed = datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc)
    c = ReplayClock(fixed)
    assert c.now() == fixed
    assert c.now() == fixed  # idempotent

    c.advance(60)
    assert c.now() == fixed + timedelta(seconds=60)


def test_replayclock_set():
    c = ReplayClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    new_time = datetime(2024, 8, 5, 9, 30, tzinfo=timezone.utc)
    c.set(new_time)
    assert c.now() == new_time


def test_replayclock_rejects_naive_on_init():
    with pytest.raises(ValueError, match="timezone-aware"):
        ReplayClock(datetime(2024, 1, 1))


def test_replayclock_rejects_naive_on_set():
    c = ReplayClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="timezone-aware"):
        c.set(datetime(2024, 6, 1))
