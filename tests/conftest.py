"""Root pytest conftest — session-scope OTel sync-mode setup.

Activates ``SimpleSpanProcessor`` for the entire test session so that spans
are flushed synchronously on ``span.end()`` and test assertions can read them
immediately from the JSONL files without waiting for a batch flush.
"""
from __future__ import annotations

import pytest

from firm.obs.tracer import use_sync_exporter


@pytest.fixture(scope="session", autouse=True)
def _otel_sync_mode() -> None:
    """Force synchronous OTel span flushing for all tests."""
    use_sync_exporter()
