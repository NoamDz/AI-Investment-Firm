from pathlib import Path
from firm.core.config import PolicyConfig, UniverseConfig, load_policy, load_universe


def test_load_policy_from_repo():
    p = load_policy(Path("config/policy.yaml"))
    assert isinstance(p, PolicyConfig)
    assert p.limits.max_position_pct == 0.10
    assert p.limits.max_trades_per_day == 20
    assert p.hitl.trade_threshold_pct == 0.03


def test_load_universe_from_repo():
    u = load_universe(Path("config/universe.yaml"))
    assert isinstance(u, UniverseConfig)
    assert len(u.tickers) == 30
    assert "AAPL" in u.tickers
    assert u.sector_map["AAPL"] == "tech"


def test_universe_rejects_unmapped_ticker(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("as_of: 2023-11-01\ntickers: [AAPL, XYZ]\nsector_map: {AAPL: tech}\n")
    import pytest
    with pytest.raises(ValueError, match="XYZ"):
        load_universe(bad)
