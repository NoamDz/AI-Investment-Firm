"""Tests for FinanceBenchSource adapter."""
from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path

import pytest

from firm.rag.financebench import FinanceBenchSource, _row_to_filing_doc
from firm.rag.source import CorpusSource, FilingDoc

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "financebench_two_docs.json"


def _fixture_loader() -> list[dict[str, object]]:
    """Translate FilingDoc-shaped fixture JSON into HF-row-shaped dicts."""
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for doc in raw["docs"]:
        rows.append(
            {
                "doc_name": doc["doc_id"],
                "company": doc["ticker"],
                "doc_type": doc["filing_type"],
                "filing_date": doc["published_at"],
                "text": doc["html"],
                "title": doc.get("title"),
            }
        )
    return rows


class TestAdapterLoadsFromLocalFixture:
    def test_adapter_loads_from_local_fixture(self) -> None:
        source = FinanceBenchSource(dataset_loader=_fixture_loader)
        docs = list(source.iter_docs())
        assert len(docs) == 2
        assert all(isinstance(d, FilingDoc) for d in docs)
        doc_ids = {d.doc_id for d in docs}
        assert "aapl-10k-2024q3" in doc_ids
        assert "nvda-10q-2024q3" in doc_ids

    def test_adapter_implements_corpus_source_protocol(self) -> None:
        source = FinanceBenchSource(dataset_loader=_fixture_loader)
        assert isinstance(source, CorpusSource)
        assert source.name == "financebench"


class TestAdapterSkipsEvalQaSet:
    def test_adapter_skips_eval_qa_set(self, tmp_path: Path) -> None:
        holdout_file = tmp_path / "holdout.json"
        holdout_file.write_text(json.dumps(["nvda-10q-2024q3"]), encoding="utf-8")

        source = FinanceBenchSource(
            dataset_loader=_fixture_loader,
            eval_holdout_file=holdout_file,
        )
        docs = list(source.iter_docs())
        assert len(docs) == 1
        assert docs[0].doc_id == "aapl-10k-2024q3"

    def test_empty_holdout_yields_all_docs(self, tmp_path: Path) -> None:
        holdout_file = tmp_path / "holdout.json"
        holdout_file.write_text(json.dumps([]), encoding="utf-8")

        source = FinanceBenchSource(
            dataset_loader=_fixture_loader,
            eval_holdout_file=holdout_file,
        )
        docs = list(source.iter_docs())
        assert len(docs) == 2


class TestPublishedAtIsParsedTzAware:
    def test_published_at_is_parsed_tz_aware(self) -> None:
        source = FinanceBenchSource(dataset_loader=_fixture_loader)
        docs = list(source.iter_docs())
        for doc in docs:
            assert doc.published_at.tzinfo is not None
            assert doc.published_at.tzinfo == timezone.utc

    def test_naive_date_string_becomes_utc(self) -> None:
        def loader() -> list[dict[str, object]]:
            return [
                {
                    "doc_name": "test-naive-date",
                    "company": "ACME",
                    "doc_type": "10-K",
                    "filing_date": "2024-01-15",
                    "text": "body",
                }
            ]

        source = FinanceBenchSource(dataset_loader=loader)
        (doc,) = list(source.iter_docs())
        assert doc.published_at.tzinfo == timezone.utc

    def test_missing_filing_date_raises(self) -> None:
        row: dict[str, object] = {
            "doc_name": "bad-row",
            "company": "ACME",
            "doc_type": "10-K",
            "text": "body",
        }
        with pytest.raises(ValueError, match="missing filing_date"):
            _row_to_filing_doc(row)


@pytest.mark.skipif(
    os.environ.get("FIRM_ALLOW_HF_DOWNLOAD") != "1",
    reason="live HF download disabled; set FIRM_ALLOW_HF_DOWNLOAD=1 to enable",
)
class TestLiveHfDownload:
    def test_live_hf_first_doc_is_filing_doc(self) -> None:
        source = FinanceBenchSource()
        doc = next(iter(source.iter_docs()))
        assert isinstance(doc, FilingDoc)
        assert doc.published_at.tzinfo is not None
