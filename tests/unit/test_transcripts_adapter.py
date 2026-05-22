"""Tests for TranscriptsCorpusSource adapter (Plan 3 T19)."""
from __future__ import annotations

from pathlib import Path

import pytest
import pydantic

from firm.rag.source import CorpusSource
from firm.rag.transcripts import TranscriptsCorpusSource

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "transcripts_two.jsonl"

_VALID_ROW: dict[str, object] = {
    "ticker": "AAPL",
    "quarter": 3,
    "fiscal_year": 2024,
    "published_at": "2024-08-01T16:00:00+00:00",
    "body": "Apple Q3 FY2024 earnings call transcript text.",
}


def _row_loader(row: dict[str, object]):
    """Return a dataset_loader callable that yields a single row."""
    return lambda: [row]


# ---------------------------------------------------------------------------
# 1. Loads from local fixture file
# ---------------------------------------------------------------------------


def test_loads_from_local_fixture() -> None:
    source = TranscriptsCorpusSource(path=FIXTURE_PATH)
    docs = list(source.iter_docs())
    assert len(docs) == 2
    doc_ids = {d.doc_id for d in docs}
    assert "transcript-AAPL-FY2024-Q3" in doc_ids
    assert "transcript-NVDA-FY2024-Q2" in doc_ids
    tickers = {d.ticker for d in docs}
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    for doc in docs:
        assert doc.published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 2. Implements CorpusSource protocol
# ---------------------------------------------------------------------------


def test_implements_corpus_source_protocol() -> None:
    source = TranscriptsCorpusSource(path=FIXTURE_PATH)
    assert isinstance(source, CorpusSource)
    assert source.name == "transcripts"


# ---------------------------------------------------------------------------
# 3–7. Missing required field raises ValueError
# ---------------------------------------------------------------------------


def test_missing_published_at_raises() -> None:
    row = {k: v for k, v in _VALID_ROW.items() if k != "published_at"}
    with pytest.raises(ValueError, match="published_at"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


def test_missing_ticker_raises() -> None:
    row = {k: v for k, v in _VALID_ROW.items() if k != "ticker"}
    with pytest.raises(ValueError, match="ticker"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


def test_missing_body_raises() -> None:
    row = {k: v for k, v in _VALID_ROW.items() if k != "body"}
    with pytest.raises(ValueError, match="body"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


def test_missing_quarter_raises() -> None:
    row = {k: v for k, v in _VALID_ROW.items() if k != "quarter"}
    with pytest.raises(ValueError, match="quarter"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


def test_missing_fiscal_year_raises() -> None:
    row = {k: v for k, v in _VALID_ROW.items() if k != "fiscal_year"}
    with pytest.raises(ValueError, match="fiscal_year"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


# ---------------------------------------------------------------------------
# 8. Naive published_at raises (FilingDoc Pydantic validator)
# ---------------------------------------------------------------------------


def test_naive_published_at_raises() -> None:
    row = {**_VALID_ROW, "published_at": "2024-08-01T16:00:00"}  # no tz offset
    with pytest.raises((ValueError, pydantic.ValidationError)):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


# ---------------------------------------------------------------------------
# 9. Invalid quarter raises (out-of-range and non-positive)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_quarter", [5, 0, -1])
def test_invalid_quarter_raises(bad_quarter: int) -> None:
    row = {**_VALID_ROW, "quarter": bad_quarter}
    with pytest.raises(ValueError, match="quarter"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


# ---------------------------------------------------------------------------
# 9b. Invalid fiscal_year raises (zero and negative)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_fy", [0, -1])
def test_invalid_fiscal_year_raises(bad_fy: int) -> None:
    row = {**_VALID_ROW, "fiscal_year": bad_fy}
    with pytest.raises(ValueError, match="fiscal_year"):
        list(TranscriptsCorpusSource(dataset_loader=_row_loader(row)).iter_docs())


# ---------------------------------------------------------------------------
# 10. published_at is parsed as tz-aware
# ---------------------------------------------------------------------------


def test_published_at_parsed_tz_aware() -> None:
    source = TranscriptsCorpusSource(path=FIXTURE_PATH)
    docs = list(source.iter_docs())
    for doc in docs:
        assert doc.published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 11. Constructor requires exactly one of path or dataset_loader
# ---------------------------------------------------------------------------


def test_constructor_requires_path_or_loader_not_both() -> None:
    with pytest.raises(ValueError):
        TranscriptsCorpusSource(
            path=FIXTURE_PATH,
            dataset_loader=lambda: [],
        )


def test_constructor_requires_path_or_loader_not_neither() -> None:
    with pytest.raises(ValueError):
        TranscriptsCorpusSource()


# ---------------------------------------------------------------------------
# 12. Blank lines in JSONL are skipped silently
# ---------------------------------------------------------------------------


def test_blank_lines_in_jsonl_are_skipped(tmp_path: Path) -> None:
    jsonl = tmp_path / "transcripts.jsonl"
    jsonl.write_text(
        '{"ticker": "AAPL", "quarter": 3, "fiscal_year": 2024, '
        '"published_at": "2024-08-01T16:00:00+00:00", "body": "text A."}\n'
        "\n"
        '{"ticker": "NVDA", "quarter": 2, "fiscal_year": 2024, '
        '"published_at": "2024-08-21T16:00:00+00:00", "body": "text B."}\n',
        encoding="utf-8",
    )
    source = TranscriptsCorpusSource(path=jsonl)
    docs = list(source.iter_docs())
    assert len(docs) == 2


# ---------------------------------------------------------------------------
# 13. UTF-8 BOM at file head is transparently stripped
# ---------------------------------------------------------------------------


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    """Common Windows/Excel exports prefix a BOM; loader must handle it."""
    jsonl = tmp_path / "transcripts_bom.jsonl"
    row = (
        '{"ticker": "AAPL", "quarter": 3, "fiscal_year": 2024, '
        '"published_at": "2024-08-01T16:00:00+00:00", "body": "text."}\n'
    )
    jsonl.write_bytes(b"\xef\xbb\xbf" + row.encode("utf-8"))
    docs = list(TranscriptsCorpusSource(path=jsonl).iter_docs())
    assert len(docs) == 1
    assert docs[0].ticker == "AAPL"
