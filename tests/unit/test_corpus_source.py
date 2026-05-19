"""Tests for FilingDoc model and CorpusSource protocol."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from firm.rag.source import CorpusSource, FilingDoc


def _make_filing_doc(**overrides: object) -> FilingDoc:
    defaults: dict[str, object] = {
        "doc_id": "doc-001",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "published_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "title": "Annual Report",
        "html": "<html><body>content</body></html>",
    }
    defaults.update(overrides)
    return FilingDoc.model_validate(defaults)


class TestFilingDocTzValidation:
    def test_filing_doc_requires_published_at_tz_aware(self) -> None:
        with pytest.raises(ValidationError):
            FilingDoc.model_validate(
                {
                    "doc_id": "doc-001",
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "published_at": None,
                    "title": "Annual Report",
                    "html": "<html/>",
                }
            )

        with pytest.raises(ValidationError):
            FilingDoc.model_validate(
                {
                    "doc_id": "doc-001",
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "published_at": datetime(2024, 1, 1, 0, 0, 0),
                    "title": "Annual Report",
                    "html": "<html/>",
                }
            )

        doc = _make_filing_doc(published_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
        assert doc.published_at.tzinfo is not None


class TestCorpusSourceProtocol:
    def test_corpus_source_protocol_is_iterable(self) -> None:
        class _FakeSource:
            name: str = "fake"

            def iter_docs(self) -> Iterator[FilingDoc]:
                docs: list[FilingDoc] = [_make_filing_doc()]
                return iter(docs)

        class _MissingIterDocs:
            name: str = "missing"

        fake = _FakeSource()
        assert isinstance(fake, CorpusSource)

        yielded = list(fake.iter_docs())
        assert len(yielded) == 1
        assert isinstance(yielded[0], FilingDoc)

        assert not isinstance(_MissingIterDocs(), CorpusSource)


class TestFilingDocRoundTrip:
    def test_filing_doc_round_trips_pydantic(self) -> None:
        original = _make_filing_doc(
            doc_id="doc-rt-001",
            ticker="MSFT",
            filing_type="10-Q",
            url="https://example.com/filing",
            metadata={"source": "edgar"},
        )

        json_str = original.model_dump_json()
        restored = FilingDoc.model_validate_json(json_str)

        assert restored.doc_id == original.doc_id
        assert restored.ticker == original.ticker
        assert restored.filing_type == original.filing_type
        assert restored.title == original.title
        assert restored.html == original.html
        assert restored.url == original.url
        assert restored.metadata == original.metadata
        assert restored.published_at == original.published_at
        assert restored.published_at.tzinfo is not None
