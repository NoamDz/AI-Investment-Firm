"""Reporter — minimal JSONL summary for Plan 1. Markdown+XLSX in Plan 3."""
from __future__ import annotations

import json
from pathlib import Path

from firm.core.clock import Clock


def make_reporter(*, reports_root: Path, clock: Clock):
    def reporter(state: dict) -> dict:
        now = clock.now()
        date_dir = reports_root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "decisions.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now.isoformat(), **state}, default=str) + "\n")
        return {"report_path": str(path)}
    return reporter
