"""Unit tests for the monitor heartbeat node."""
from datetime import datetime, timezone

from firm.core.clock import ReplayClock
from firm.agents.monitor import make_monitor


FIXED_DT = datetime(2026, 1, 15, 9, 30, 0, tzinfo=timezone.utc)


def test_monitor_returns_heartbeat_at():
    clock = ReplayClock(FIXED_DT)
    monitor = make_monitor(clock)
    result = monitor({})
    assert result == {"heartbeat_at": FIXED_DT.isoformat()}
