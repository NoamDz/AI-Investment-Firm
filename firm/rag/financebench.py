"""FinanceBench corpus adapter. See spec §5.2 and plan T7.

The live PatronusAI/financebench dataset exposes only ``doc_period`` (year)
and ``doc_type`` (10k/10q/8k/Earnings) — no explicit filing date. We derive
``published_at`` from those two fields using SEC filing conventions: a 10-K
covering FY N is filed ~Q1 of N+1, a 10-Q ~mid Q1 of N+1, an 8-K within the
event year (approximated as N-12-31), and an Earnings release ~30 days after
period close. Fixture rows may still supply ``filing_date`` directly and that
takes precedence. All timestamps are coerced to UTC midnight.

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
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from firm.rag.source import FilingDoc


def _load_from_hf() -> Iterable[dict[str, object]]:
    """Aggregate FinanceBench rows by doc_name so each doc yields a single
    pseudo-row with ``text`` set to the unique ``evidence_text_full_page``
    pages concatenated in page order.

    The HF dataset stores 150 Q&A pairs across 84 unique filings and does
    not include the full PDF text — only per-Q&A page-level evidence. We
    treat the union of those pages as the document corpus for retrieval.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("PatronusAI/financebench")

    first_row_by_doc: dict[str, dict[str, object]] = {}
    pages_by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_pages: dict[str, set[str]] = defaultdict(set)

    for row in ds["train"]:
        doc_name = str(row.get("doc_name") or "")
        if not doc_name:
            continue
        first_row_by_doc.setdefault(doc_name, dict(row))
        for ev in row.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            page_text = ev.get("evidence_text_full_page") or ev.get("evidence_text")
            if not page_text:
                continue
            page_text_str = str(page_text)
            if page_text_str in seen_pages[doc_name]:
                continue
            seen_pages[doc_name].add(page_text_str)
            try:
                page_num = int(ev.get("evidence_page_num") or 0)
            except (TypeError, ValueError):
                page_num = 0
            pages_by_doc[doc_name].append((page_num, page_text_str))

    for doc_name, base in first_row_by_doc.items():
        pages = sorted(pages_by_doc[doc_name], key=lambda p: p[0])
        out = dict(base)
        out["text"] = "\n\n".join(text for _, text in pages)
        yield out


def _derive_filing_date(doc_period: object, doc_type: str) -> datetime:
    """Approximate a filing date from a fiscal-year period + filing type."""
    year = int(str(doc_period).strip())
    normalised = doc_type.strip().lower().replace("-", "")
    if normalised == "10k":
        return datetime(year + 1, 3, 31, tzinfo=timezone.utc)
    if normalised == "10q":
        return datetime(year + 1, 2, 15, tzinfo=timezone.utc)
    if normalised == "8k":
        return datetime(year, 12, 31, tzinfo=timezone.utc)
    return datetime(year + 1, 1, 31, tzinfo=timezone.utc)


def _row_to_filing_doc(row: dict[str, object]) -> FilingDoc:
    doc_name = str(row.get("doc_name") or row.get("document_name") or "")
    if not doc_name:
        raise ValueError("missing doc_name")

    doc_type = str(row.get("doc_type") or "")
    filing_date_raw = row.get("filing_date")
    if filing_date_raw:
        dt = datetime.fromisoformat(str(filing_date_raw).strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        date_label: object = filing_date_raw
    elif row.get("doc_period") is not None:
        dt = _derive_filing_date(row["doc_period"], doc_type)
        date_label = dt.date().isoformat()
    else:
        raise ValueError(f"missing filing_date and doc_period for {doc_name}")

    ticker = str(row.get("company") or row.get("ticker_name") or "")
    text = str(row.get("text") or row.get("pdf_content") or "")
    url_val = row.get("doc_link") or row.get("pdf_url")
    url = str(url_val) if url_val else None
    title_val = row.get("title")
    title = str(title_val) if title_val else f"{ticker} {doc_type} {date_label}"

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


