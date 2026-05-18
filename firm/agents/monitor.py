"""Monitor agent: emits a heartbeat timestamp into the working state."""
from __future__ import annotations

from typing import Callable

from firm.core.clock import Clock
from firm.orchestrator.state import WorkingState


def make_monitor(clock: Clock) -> Callable[[WorkingState], dict]:
    def monitor(state: WorkingState) -> dict:
        return {"heartbeat_at": clock.now().isoformat()}
    return monitor
