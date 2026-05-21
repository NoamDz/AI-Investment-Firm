"""T22: Litestream replicator — structural and unit tests.

Test A: static structural validation of docker-compose.yml and
        config/litestream.yml (always runs, no Docker required).
Test B: per-connection wal_autocheckpoint PRAGMA assertion (always runs).

The full docker-up + replication-catch-up scenario from the plan spec
is exercised by `make litestream-drill` in T23, not here.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path


# ---------------------------------------------------------------------------
# Test A — static structural validation (always runs)
# ---------------------------------------------------------------------------


def test_litestream_service_in_docker_compose():
    import yaml

    root = Path(__file__).parent.parent.parent
    with open(root / "docker-compose.yml", encoding="utf-8") as f:
        compose = yaml.safe_load(f)
    assert "litestream" in compose["services"]
    svc = compose["services"]["litestream"]
    assert svc["command"].startswith("replicate")
    assert "/data" in str(svc["volumes"]).replace("\\", "/")
    assert svc["depends_on"]["firm"]["condition"] == "service_started"


def test_litestream_config_has_max_wal_size():
    import yaml

    root = Path(__file__).parent.parent.parent
    with open(root / "config/litestream.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["max-wal-size"] == "16MB"
    assert cfg["dbs"][0]["path"] == "/data/firm.db"
    # First replica must be the file destination — S3 is optional/opt-in.
    file_replicas = [r for r in cfg["dbs"][0]["replicas"] if r["type"] == "file"]
    assert len(file_replicas) == 1
    assert "/data/litestream" in file_replicas[0]["path"]


# ---------------------------------------------------------------------------
# Test B — connection-level WAL autocheckpoint (always runs)
# ---------------------------------------------------------------------------


def test_wal_autocheckpoint_set_on_connection(tmp_path: Path):
    from firm.db.connection import get_conn

    db = tmp_path / "firm.db"
    with closing(get_conn(db)) as conn:
        result = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        assert result == 1000


# The plan's `docker compose up firm litestream && stop firm && verify litestream
# caught up` invariant is covered by `make litestream-drill` (T23), which restores
# from the replica into a temp DB and asserts row counts match. No stub test here.
