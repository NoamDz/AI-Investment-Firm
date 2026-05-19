"""FinanceBench corpus adapter.

Loads SEC filings from the PatronusAI/financebench Hugging Face dataset and
exposes them via the CorpusSource protocol.

Note: ``ticker`` is the dataset's ``company`` string (e.g. "Apple Inc."); symbol
normalisation is deferred to a later task.

The eval holdout set (plan 4 will populate the full 150 Q&A pairs) is stored in
``config/financebench_eval_holdout.json`` and referenced via ``eval_holdout_file``.
Rows whose ``doc_name`` appears in that list are skipped so the eval set stays
unseen during ingest.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from firm.rag.source import CorpusSource, FilingDoc


def _load_from_hf() -> Iterable[dict[str, object]]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("PatronusAI/financebench")
    yield from ds["train"]


def _row_to_filing_doc(row: dict[str, object]) -> FilingDoc:
    doc_name = str(row.get("doc_name") or row.get("document_name") or "")
    if not doc_name:
        raise ValueError("missing doc_name")

    filing_date_raw = row.get("filing_date")
    if not filing_date_raw:
        raise ValueError(f"missing filing_date for {doc_name}")

    dt = datetime.fromisoformat(str(filing_date_raw).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    ticker = str(row.get("company") or row.get("ticker_name") or "")
    doc_type = str(row.get("doc_type") or "")
    text = str(row.get("text") or row.get("pdf_content") or "")
    url_val = row.get("doc_link") or row.get("pdf_url")
    url = str(url_val) if url_val else None
    title_val = row.get("title")
    title = str(title_val) if title_val else f"{ticker} {doc_type} {filing_date_raw}"

    return FilingDoc(
        doc_id=doc_name,
        ticker=ticker,
        filing_type=doc_type,
        published_at=dt,
        title=title,
        html=text,
        url=url,
    )


class FinanceBenchSource:
    """CorpusSource backed by PatronusAI/financebench.

    Parameters
    ----------
    dataset_loader:
        Callable that returns an iterable of raw dataset rows.  Defaults to
        ``_load_from_hf``; override in tests to avoid network I/O.
    eval_holdout_file:
        Path to a JSON file containing a list of ``doc_name`` strings to skip.
        Rows whose doc_name is in this set are excluded so plan-4 evaluation
        can reuse the same documents without training contamination.
        TODO(plan4): populate the full 150 Q&A pair holdout list here.
    """

    name = "financebench"

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[dict[str, object]]] = _load_from_hf,
        eval_holdout_file: Path | None = None,
    ) -> None:
        self._loader = dataset_loader
        self._holdout: set[str] = (
            set(json.loads(eval_holdout_file.read_text(encoding="utf-8")))
            if eval_holdout_file
            else set()
        )

    def iter_docs(self) -> Iterator[FilingDoc]:
        for row in self._loader():
            doc_name = str(row.get("doc_name") or row.get("document_name") or "")
            if doc_name in self._holdout:
                continue
            yield _row_to_filing_doc(row)


# Satisfy the runtime-checkable CorpusSource protocol.
assert isinstance(FinanceBenchSource(), CorpusSource)
