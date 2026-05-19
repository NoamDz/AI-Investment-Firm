"""Monitor agent: emits a heartbeat timestamp into the working state."""
from __future__ import annotations

from typing import Any, Callable

from firm.core.clock import Clock
from firm.orchestrator.state import WorkingState


def make_monitor(clock: Clock) -> Callable[[WorkingState], dict[str, Any]]:
    def monitor(state: WorkingState) -> dict[str, Any]:
        return {"heartbeat_at": clock.now().isoformat()}
    return monitor
