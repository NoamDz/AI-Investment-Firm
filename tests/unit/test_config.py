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


def test_policy_rejects_negative_pct(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "limits:\n"
        "  max_position_pct: -0.1\n"
        "  max_sector_pct: 0.3\n"
        "  max_gross_exposure: 1.0\n"
        "  max_trade_pct: 0.05\n"
        "  max_trades_per_day: 20\n"
        "  min_cash_pct: 0.05\n"
        "  max_daily_loss_pct: 0.03\n"
        "  stale_quote_seconds: 60\n"
        "  stale_filing_days: 90\n"
        "hitl:\n"
        "  trade_threshold_pct: 0.03\n"
        "  escalate_new_ticker: true\n"
    )
    import pytest
    with pytest.raises(ValueError):
        load_policy(bad)


def test_policy_rejects_pct_above_one(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "limits:\n"
        "  max_position_pct: 1.5\n"
        "  max_sector_pct: 0.3\n"
        "  max_gross_exposure: 1.0\n"
        "  max_trade_pct: 0.05\n"
        "  max_trades_per_day: 20\n"
        "  min_cash_pct: 0.05\n"
        "  max_daily_loss_pct: 0.03\n"
        "  stale_quote_seconds: 60\n"
        "  stale_filing_days: 90\n"
        "hitl:\n"
        "  trade_threshold_pct: 0.03\n"
        "  escalate_new_ticker: true\n"
    )
    import pytest
    with pytest.raises(ValueError):
        load_policy(bad)


def test_universe_rejects_empty_tickers(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("as_of: 2023-11-01\ntickers: []\nsector_map: {}\n")
    import pytest
    with pytest.raises(ValueError):
        load_universe(bad)


def test_universe_rejects_duplicate_tickers(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "as_of: 2023-11-01\n"
        "tickers: [AAPL, AAPL, MSFT]\n"
        "sector_map: {AAPL: tech, MSFT: tech}\n"
    )
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        load_universe(bad)


def test_universe_rejects_orphan_sector_map_key(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "as_of: 2023-11-01\n"
        "tickers: [AAPL]\n"
        "sector_map: {AAPL: tech, EXTRA: tech}\n"
    )
    import pytest
    with pytest.raises(ValueError, match="EXTRA"):
        load_universe(bad)
