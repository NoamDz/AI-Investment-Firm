"""Unit tests for firm.eval.sufficiency (Plan 4 T12).

Covers:
  * load_dev_set against the shipped fixture
  * compute_sufficiency_metrics under perfect / half-precision / half-recall /
    empty-produced / empty-gold / empty-dev-set conditions
  * cross-claim leakage rejection
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from firm.eval.sufficiency import (
    DEFAULT_DEV_SET_PATH,
    SufficiencyDevCase,
    compute_sufficiency_metrics,
    load_dev_set,
)


# ---------------------------------------------------------------------------
# load_dev_set
# ---------------------------------------------------------------------------


def test_load_dev_set_reads_shipped_fixture() -> None:
    cases = load_dev_set(DEFAULT_DEV_SET_PATH)
    # 6 lines in tests/fixtures/sufficiency_dev_set.jsonl
    assert len(cases) >= 5
    assert all(isinstance(c, SufficiencyDevCase) for c in cases)
    # First case in the fixture is the AAPL revenue claim with 2 golds.
    first = cases[0]
    assert first.claim_id == "dev-001"
    assert "AAPL" in first.text
    assert len(first.gold_citations) == 2


def test_load_dev_set_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "dev.jsonl"
    p.write_text(
        '\n'
        + json.dumps({"claim_id": "c1", "text": "t1", "gold_citations": ["g1"]})
        + '\n\n'
        + json.dumps({"claim_id": "c2", "text": "t2", "gold_citations": ["g2"]})
        + '\n',
        encoding="utf-8",
    )
    cases = load_dev_set(p)
    assert [c.claim_id for c in cases] == ["c1", "c2"]


def test_load_dev_set_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dev_set(tmp_path / "does_not_exist.jsonl")


def test_load_dev_set_raises_on_missing_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"claim_id": "c1", "text": "t1"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_dev_set(p)


# ---------------------------------------------------------------------------
# compute_sufficiency_metrics
# ---------------------------------------------------------------------------


def _case(claim_id: str, golds: list[str]) -> SufficiencyDevCase:
    return SufficiencyDevCase(
        claim_id=claim_id, text=f"text-{claim_id}", gold_citations=tuple(golds)
    )


def test_compute_perfect_precision_recall() -> None:
    dev = [_case("c1", ["g1", "g2"]), _case("c2", ["g3"])]
    produced = {"c1": ["g1", "g2"], "c2": ["g3"]}
    p, r = compute_sufficiency_metrics(dev, produced)
    assert p == 1.0
    assert r == 1.0


def test_compute_half_precision() -> None:
    # Produced 2 chunks for c1, only 1 is gold → precision = 1/2 = 0.5
    dev = [_case("c1", ["g1"])]
    produced = {"c1": ["g1", "noise"]}
    p, r = compute_sufficiency_metrics(dev, produced)
    assert p == 0.5
    assert r == 1.0


def test_compute_half_recall() -> None:
    # 2 golds, only 1 produced → recall = 1/2 = 0.5
    dev = [_case("c1", ["g1", "g2"])]
    produced = {"c1": ["g1"]}
    p, r = compute_sufficiency_metrics(dev, produced)
    assert p == 1.0
    assert r == 0.5


def test_compute_empty_produced() -> None:
    dev = [_case("c1", ["g1"])]
    p, r = compute_sufficiency_metrics(dev, {})
    assert p == 0.0
    assert r == 0.0


def test_compute_empty_gold() -> None:
    # Dev case with NO gold citations: every produced chunk is a false
    # positive → precision = 0.0; recall is undefined but reported as 0.0.
    dev = [_case("c1", [])]
    p, r = compute_sufficiency_metrics(dev, {"c1": ["x"]})
    assert p == 0.0
    assert r == 0.0


def test_compute_empty_dev_set() -> None:
    p, r = compute_sufficiency_metrics([], {"c1": ["g1"]})
    assert p == 0.0
    assert r == 0.0


def test_compute_cross_claim_leakage_rejected() -> None:
    # If c1's gold chunk_id g1 is produced under c2, it must NOT count as a
    # true positive — pairs are (claim_id, chunk_id), not raw chunk_ids.
    dev = [_case("c1", ["g1"]), _case("c2", ["g2"])]
    produced = {"c1": ["g2"], "c2": ["g1"]}  # swapped
    p, r = compute_sufficiency_metrics(dev, produced)
    assert p == 0.0
    assert r == 0.0


def test_compute_iterable_chunk_ids_accepted() -> None:
    # produced_citations values can be any Iterable[str], not just list.
    dev = [_case("c1", ["g1", "g2"])]
    produced = {"c1": iter(["g1", "g2"])}
    p, r = compute_sufficiency_metrics(dev, produced)
    assert p == 1.0
    assert r == 1.0
