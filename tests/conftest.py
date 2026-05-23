"""Root pytest conftest — session-scope OTel sync-mode setup and VCR defaults.

Activates ``SimpleSpanProcessor`` for the entire test session so that spans
are flushed synchronously on ``span.end()`` and test assertions can read them
immediately from the JSONL files without waiting for a batch flush.

Also pins ``FIRM_VCR_MODE=replay`` for all tests (defense-in-depth: if any
test accidentally constructs ``CachedAnthropicClient.from_env()`` with
``FIRM_LLM_MODE=live`` or ``record``, the cassette layer intercepts and raises
``CassetteMissError`` instead of hitting the live API).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from firm.obs.tracer import use_sync_exporter


@pytest.fixture(scope="session", autouse=True)
def _otel_sync_mode() -> None:
    """Force synchronous OTel span flushing for all tests."""
    use_sync_exporter()


@pytest.fixture(scope="session", autouse=True)
def _pin_vcr_replay_mode() -> None:
    """Pin FIRM_VCR_MODE=replay for the entire test session.

    Uses ``setdefault`` so an explicit env var override is still respected
    (e.g. a developer running with ``FIRM_VCR_MODE=record`` to re-record
    cassettes).
    """
    os.environ.setdefault("FIRM_VCR_MODE", "replay")


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    """Temporary cassette directory for record-mode tests."""
    d = tmp_path / "cassettes"
    d.mkdir()
    return d
