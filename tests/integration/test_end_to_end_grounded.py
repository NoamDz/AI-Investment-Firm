"""End-to-end grounded demo integration test (Plan 2 §T30).

Flow under test
---------------
1. Seed ``data/precomputed/`` fixture parquets (required by the CLI's tool
   constructors — paths are hardcoded relative to the repo root).
2. Run ``firm ingest`` as a subprocess to populate a local Qdrant instance
   with AAPL + MSFT fixture chunks and to warm the LLM cache for the
   contextual augmentation step.
3. In-process retrieval: load NomicEmbedder + BM25 + BgeReranker + Qdrant
   and run a real retrieval for the AAPL research question.  This gives us the
   *exact* chunks (in the *exact* reranked order) that ``firm run`` will pass
   to the extractor — so the prompt hashes we seed into llm_cache will match.
4. Drive capturing stubs through the real agent classes (extractor, judge, PM
   voter) using those retrieved chunks to extract canonical
   ``(system, messages, tools)`` triples.
5. Seed the LLM cache with canned responses keyed on those prompt hashes.
6. Run ``firm run --once`` as a subprocess with ``FIRM_LLM_MODE=cached``,
   ``FIRM_BROKER=FAKE``, and ``QDRANT_LOCAL_PATH`` pointing at the same
   local Qdrant directory.
7. Assert:
   * subprocess exit code == 0
   * ``outbox`` table has 1 confirmed row
   * ``decisions`` table has >= 3 rows
   * The JSONL report for 2024-03-13 contains at least one citation with a
     ``chunk_id`` field traceable to a seeded chunk.

All LLM calls are served from the SQLite llm_cache (no network).
All Qdrant ops use the local filesystem (no Docker / network Qdrant).

Markers
-------
``@pytest.mark.requires_models`` — requires sentence-transformers
(NomicEmbedder) and the bge-reranker-v2-m3 reranker.  Skip with::

    pytest -m "not requires_models"
"""
from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")
pytest.importorskip("sentence_transformers")

import pyarrow as pa  # type: ignore[import-untyped]  # noqa: E402
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from firm.agents.hitl import mark_approved  # noqa: E402
from firm.agents.pm import PmLens, PmVoter  # noqa: E402
from firm.agents.reporter import _persist_decisions_from_state  # noqa: E402
from firm.core.clock import ReplayClock  # noqa: E402
from firm.core.models import Claim, Decision  # noqa: E402
from firm.db.connection import get_conn  # noqa: E402
from firm.db.migrations import init_db  # noqa: E402
from firm.grounding.judge import SufficiencyJudge  # noqa: E402
from firm.llm.cache import LlmCache, hash_prompt  # noqa: E402
from firm.llm.citations import AnthropicCitationsExtractor  # noqa: E402
from firm.llm.client import CompletionResponse  # noqa: E402
from firm.orchestrator.graph import _FIRM_SERDE  # noqa: E402
from firm.orchestrator.state import WorkingState  # noqa: E402
from firm.rag.chunk import Chunk, chunk_document  # noqa: E402
from firm.rag.contextual import ContextualAugmenter  # noqa: E402
from firm.rag.source import FilingDoc  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The FIRM_REPLAY_AT passed to the subprocess.
# Filing dates must be within stale_filing_days=90 of this timestamp.
_REPLAY_AT = "2024-03-13T14:30:00+00:00"
_REPLAY_DT = datetime.fromisoformat(_REPLAY_AT)

# Fixture clock for LlmCache.put() calls (created_at is not used for cache lookup).
_FIXTURE_CLOCK = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))

# Chunk params — must match the fixture rag.yaml written inside the test.
_FIXTURE_CHUNK_TARGET_TOKENS = 64
_FIXTURE_CHUNK_OVERLAP_TOKENS = 8

# LLM models — must match the fixture llm.yaml written inside the test.
_RESEARCH_MODEL = "claude-sonnet-4-6"
_JUDGE_MODEL = "claude-haiku-4-5"
_PM_MODEL = "claude-sonnet-4-6"
_AUGMENTATION_MODEL = "claude-haiku-4-5"

# Qdrant collection name — must match the fixture rag.yaml.
_COLLECTION = "e2e_grounded_chunks"

# The canned doc_summary produced by the augmentation cache entry.
# This is the text returned by the _CapturingMessagesClient's CompletionResponse.
_CANNED_DOC_SUMMARY = "Fixture contextual summary."

# Repo root (two parents up from tests/integration/).
_REPO_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Fixture corpus rows
# ---------------------------------------------------------------------------

# Two docs with filing_date within 90 days of the replay clock (2024-03-13).
_FIXTURE_ROWS: list[dict[str, object]] = [
    {
        "doc_name": "AAPL_2024_10K_e2e",
        "filing_date": "2024-01-15",
        "company": "Apple Inc.",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Apple Inc. reported strong quarterly performance across product lines. "
            "Revenue increased materially year over year driven by Services and iPhone. "
            "Operating margin expanded and management reiterated full-year guidance. "
            "Capital returns continued through share repurchases and dividends. "
            "The company cash position remained robust supporting future investments.</p>"
            "</body></html>"
        ),
    },
    {
        "doc_name": "MSFT_2024_10K_e2e",
        "filing_date": "2024-01-16",
        "company": "Microsoft Corporation",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Microsoft Corporation delivered robust cloud revenue growth in Azure. "
            "Commercial cloud revenues reached record levels driven by enterprise demand. "
            "Operating income grew alongside disciplined cost management and margin expansion. "
            "The intelligent cloud segment was a primary growth driver for the quarter. "
            "Management expects continued momentum in AI-driven productivity tools.</p>"
            "</body></html>"
        ),
    },
]


# ---------------------------------------------------------------------------
# Capturing stub
# ---------------------------------------------------------------------------


class _CapturingMessagesClient:
    """Stub implementing both AnthropicMessagesClient and AnthropicClient protocols.

    Captures every ``messages_create`` / ``complete`` call so the test can
    compute canonical prompt hashes without hitting the real API.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        self.calls.append({
            "model": model,
            "system": system,
            "messages": list(messages),
            "tools": list(tools) if tools is not None else None,
        })
        return {
            "content": [{"type": "text", "text": "stub"}],
            "model": model,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, object]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        self.calls.append({
            "model": model,
            "system": system if system is not None else "",
            "messages": list(messages),
            "tools": None,
        })
        return CompletionResponse(text=_CANNED_DOC_SUMMARY, input_tokens=0,
                                  output_tokens=0, cache_hit=False)


# ---------------------------------------------------------------------------
# Helper: row → FilingDoc
# ---------------------------------------------------------------------------


def _row_to_filing_doc(row: dict[str, object]) -> FilingDoc:
    doc_name = str(row.get("doc_name", ""))
    filing_date_raw = str(row.get("filing_date", ""))
    dt = datetime.fromisoformat(filing_date_raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    company = str(row.get("company", ""))
    doc_type = str(row.get("doc_type", ""))
    text = str(row.get("text") or "")
    title = f"{company} {doc_type} {filing_date_raw}"
    return FilingDoc(
        doc_id=doc_name,
        ticker=company,
        filing_type=doc_type,
        published_at=dt,
        title=title,
        html=text,
    )


# ---------------------------------------------------------------------------
# Helper: ensure data/precomputed/ parquets exist
# ---------------------------------------------------------------------------


def _ensure_precomputed_parquets() -> None:
    """Write fixture parquets to data/precomputed/ if absent.

    The CLI hardcodes ``Path("data/precomputed/fundamentals.parquet")`` and
    ``Path("data/precomputed/risk_metrics.parquet")``.  These paths are
    resolved relative to the subprocess cwd (the repo root).  We write minimal
    deterministic rows covering the AAPL + MSFT tickers at dates within the
    PIT window of the replay clock (2024-03-13).
    """
    from datetime import date
    from decimal import Decimal

    precomputed_dir = _REPO_ROOT / "data" / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    fund_path = precomputed_dir / "fundamentals.parquet"
    if not fund_path.exists():
        fund_rows: list[tuple[str, str, date, Decimal]] = [
            ("AAPL", "pe_ratio", date(2024, 1, 15), Decimal("27.5")),
            ("AAPL", "gross_margin", date(2024, 1, 15), Decimal("0.46")),
            ("AAPL", "revenue_yoy_growth", date(2024, 1, 15), Decimal("0.035")),
            ("AAPL", "debt_to_equity", date(2024, 1, 15), Decimal("1.60")),
            ("AAPL", "current_ratio", date(2024, 1, 15), Decimal("1.06")),
            ("MSFT", "pe_ratio", date(2024, 1, 16), Decimal("35.0")),
            ("MSFT", "gross_margin", date(2024, 1, 16), Decimal("0.70")),
            ("MSFT", "revenue_yoy_growth", date(2024, 1, 16), Decimal("0.16")),
            ("MSFT", "debt_to_equity", date(2024, 1, 16), Decimal("0.35")),
            ("MSFT", "current_ratio", date(2024, 1, 16), Decimal("1.80")),
        ]
        tickers, ratio_names, as_ofs, values = [], [], [], []
        for t, r, a, v in fund_rows:
            tickers.append(t)
            ratio_names.append(r)
            as_ofs.append(a)
            values.append(str(v))
        pq.write_table(
            pa.table({
                "ticker": pa.array(tickers, type=pa.string()),
                "ratio_name": pa.array(ratio_names, type=pa.string()),
                "as_of": pa.array(as_ofs, type=pa.date32()),
                "value": pa.array(values, type=pa.string()),
            }),
            str(fund_path),
        )

    risk_path = precomputed_dir / "risk_metrics.parquet"
    if not risk_path.exists():
        risk_rows: list[tuple[str, str, date, Decimal]] = [
            ("AAPL", "volatility_30d", date(2024, 1, 15), Decimal("0.19")),
            ("AAPL", "beta_180d", date(2024, 1, 15), Decimal("1.12")),
            ("AAPL", "max_drawdown_90d", date(2024, 1, 15), Decimal("0.08")),
            ("MSFT", "volatility_30d", date(2024, 1, 16), Decimal("0.20")),
            ("MSFT", "beta_180d", date(2024, 1, 16), Decimal("1.05")),
            ("MSFT", "max_drawdown_90d", date(2024, 1, 16), Decimal("0.07")),
        ]
        tickers2, metric_ids, as_ofs2, values2 = [], [], [], []
        for t, m, a, v in risk_rows:
            tickers2.append(t)
            metric_ids.append(m)
            as_ofs2.append(a)
            values2.append(str(v))
        pq.write_table(
            pa.table({
                "ticker": pa.array(tickers2, type=pa.string()),
                "metric_id": pa.array(metric_ids, type=pa.string()),
                "as_of": pa.array(as_ofs2, type=pa.date32()),
                "value": pa.array(values2, type=pa.string()),
            }),
            str(risk_path),
        )


# ---------------------------------------------------------------------------
# Helper: seed LLM cache for ingest (contextual augmentation)
# ---------------------------------------------------------------------------


def _seed_augmentation_cache(db_path: Path) -> None:
    """Pre-seed llm_cache for the ContextualAugmenter calls made during ingest.

    Drives the real augmenter with a capturing stub to record the canonical
    (system, messages) pairs, then writes a canned response for each.  The
    canned response text is ``_CANNED_DOC_SUMMARY`` — the same string returned
    by the ``complete()`` stub — so that chunks stored in Qdrant will carry
    that exact doc_summary value and ``_retrieve_aapl_chunks`` can apply the
    same prefix transformation that ``HybridRetriever`` does at retrieve time.
    """
    capturing = _CapturingMessagesClient()
    augmenter = ContextualAugmenter(
        client=capturing,
        model=_AUGMENTATION_MODEL,
    )
    cache = LlmCache(db_path=db_path, clock=_FIXTURE_CLOCK)

    for row in _FIXTURE_ROWS:
        doc = _row_to_filing_doc(row)
        chunks = chunk_document(
            doc,
            target_tokens=_FIXTURE_CHUNK_TARGET_TOKENS,
            overlap_tokens=_FIXTURE_CHUNK_OVERLAP_TOKENS,
        )
        if not chunks:
            chunks = [
                Chunk(
                    id=f"{doc.doc_id}-0",
                    doc_id=doc.doc_id,
                    ticker=doc.ticker,
                    section="body",
                    published_at=doc.published_at,
                    text=doc.html[:200],
                    char_span=(0, 200),
                    token_count=10,
                )
            ]
        augmenter.augment(doc, list(chunks))

    for call in capturing.calls:
        prompt_hash = hash_prompt(
            system=call["system"],
            messages=call["messages"],
            tools=None,
        )
        response_dict: dict[str, object] = {
            "content": [{"type": "text", "text": _CANNED_DOC_SUMMARY}],
            "model": call["model"],
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        cache.put(
            prompt_hash=prompt_hash,
            model=call["model"],
            response_json=json.dumps(response_dict),
            input_tokens=10,
            output_tokens=10,
        )


# ---------------------------------------------------------------------------
# Helper: retrieve AAPL chunks in-process (post-ingest)
# ---------------------------------------------------------------------------


def _retrieve_aapl_chunks(qdrant_local_path: Path) -> list[Chunk]:
    """Run in-process retrieval against the populated Qdrant to get the exact
    chunks (in the exact reranked order) that ``firm run`` will pass to the
    extractor.

    This step runs AFTER ``firm ingest`` has populated the Qdrant collection.
    By using the real embedder, BM25, and reranker here, we guarantee that the
    capturing extractor in ``_seed_extractor_cache`` produces the same
    (system, messages, tools) triple — and therefore the same prompt hash —
    as the real extractor will compute when ``firm run`` invokes it.
    """
    from firm.agents.research import _format_question
    from firm.rag.embed import BM25Sparse, NomicEmbedder
    from firm.rag.preprocess import tables_to_prose
    from firm.rag.qdrant_store import VectorStore
    from firm.rag.rerank import BgeReranker
    from firm.rag.retrieve import GroundedRetriever, HybridRetriever

    question = _format_question("AAPL")

    embedder = NomicEmbedder()
    sparse = BM25Sparse()

    # Fit BM25 on the same corpus texts used during ingest.
    all_texts: list[str] = []
    for row in _FIXTURE_ROWS:
        doc = _row_to_filing_doc(row)
        processed_html = tables_to_prose(doc.html)
        doc_processed = doc.model_copy(update={"html": processed_html})
        chunks = chunk_document(
            doc_processed,
            target_tokens=_FIXTURE_CHUNK_TARGET_TOKENS,
            overlap_tokens=_FIXTURE_CHUNK_OVERLAP_TOKENS,
        )
        all_texts.extend(c.text for c in chunks)
    if all_texts:
        sparse.fit(all_texts)
    else:
        sparse.fit(["placeholder"])

    qdrant_client = QdrantClient(path=str(qdrant_local_path))
    store = VectorStore(qdrant_client)

    hybrid = HybridRetriever(
        store=store,
        embedder=embedder,
        sparse=sparse,
        collection=_COLLECTION,
        k_retrieve=8,
    )
    reranker = BgeReranker(model_id="BAAI/bge-reranker-v2-m3", score_floor=0.0)
    retriever = GroundedRetriever(hybrid=hybrid, reranker=reranker, k_final=4)

    retrieved = retriever.retrieve(question, as_of=_REPLAY_DT)
    qdrant_client.close()
    return [rc.chunk for rc in retrieved]


# ---------------------------------------------------------------------------
# Helper: build tool payload (mirrors _build_llm_stack in cli.py)
# ---------------------------------------------------------------------------


def _build_tool_payload() -> list[dict[str, object]]:
    """Construct the tools= list that AnthropicCitationsExtractor will pass.

    Mirrors the construction in ``firm.cli._build_llm_stack`` so that the
    prompt hash computed from the capturing extractor matches the hash computed
    by the real extractor during ``firm run``.
    """
    from firm.tools.fundamentals import FundamentalsTool
    from firm.tools.risk_metrics import RiskMetricsTool

    return [
        {
            "name": FundamentalsTool.tool_def.name,
            "description": FundamentalsTool.tool_def.description,
            "input_schema": copy.deepcopy(dict(FundamentalsTool.tool_def.input_schema)),
        },
        {
            "name": RiskMetricsTool.tool_def.name,
            "description": RiskMetricsTool.tool_def.description,
            "input_schema": copy.deepcopy(dict(RiskMetricsTool.tool_def.input_schema)),
        },
    ]


# ---------------------------------------------------------------------------
# Helper: seed LLM cache for research extractor
# ---------------------------------------------------------------------------


def _seed_extractor_cache(db_path: Path, chunks: list[Chunk]) -> list[Claim]:
    """Seed llm_cache for the AnthropicCitationsExtractor call.

    Uses ``chunks`` (the output of in-process retrieval) so the document blocks
    in the captured messages match exactly what ``firm run``'s extractor will
    build.  The canned response has NO tool_use blocks (single-turn path) and
    two text+citations blocks pointing at ``chunks[0]`` — so the extractor
    emits two Claims with ``source_chunk_id = chunks[0].id``.

    Returns the Claims that the canned response would produce, so callers can
    use them to seed the judge and PM voter caches.
    """
    capturing = _CapturingMessagesClient()
    extractor = AnthropicCitationsExtractor(
        client=capturing,
        model=_RESEARCH_MODEL,
        max_tokens=4096,
    )
    # Override _tool_payload so the capturing call records tools=tool_payload,
    # matching what the CLI extractor (which is initialised with real Tool
    # objects) will produce.
    tool_payload = _build_tool_payload()
    extractor._tool_payload = tool_payload  # noqa: SLF001

    from firm.agents.research import _format_question

    question = _format_question("AAPL")
    extractor.extract(query=question, chunks=chunks, as_of=_REPLAY_DT)

    assert len(capturing.calls) >= 1, "extractor made no calls to capturing stub"
    call = capturing.calls[0]

    # Canned response: two text blocks with citations referencing chunks[0].
    first_chunk = chunks[0]
    canned_content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": "Apple Inc. reported strong revenue growth driven by Services and iPhone.",
            "citations": [
                {
                    "type": "char_location",
                    "document_index": 0,
                    "document_title": first_chunk.doc_id,
                    "start_char_index": 0,
                    "end_char_index": min(50, len(first_chunk.text)),
                    "cited_text": first_chunk.text[:50],
                }
            ],
        },
        {
            "type": "text",
            "text": "Operating margin expanded year over year with capital returns continued.",
            "citations": [
                {
                    "type": "char_location",
                    "document_index": 0,
                    "document_title": first_chunk.doc_id,
                    "start_char_index": 0,
                    "end_char_index": min(80, len(first_chunk.text)),
                    "cited_text": first_chunk.text[:80],
                }
            ],
        },
    ]
    response_dict: dict[str, object] = {
        "content": canned_content,
        "model": call["model"],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    prompt_hash = hash_prompt(
        system=call["system"],
        messages=call["messages"],
        tools=call["tools"],
    )
    cache = LlmCache(db_path=db_path, clock=_FIXTURE_CLOCK)
    cache.put(
        prompt_hash=prompt_hash,
        model=call["model"],
        response_json=json.dumps(response_dict),
        input_tokens=100,
        output_tokens=50,
    )

    # Return the Claims the canned response would produce.
    return [
        Claim(
            text="Apple Inc. reported strong revenue growth driven by Services and iPhone.",
            source_chunk_id=first_chunk.id,
            source_span=(0, min(50, len(first_chunk.text))),
        ),
        Claim(
            text="Operating margin expanded year over year with capital returns continued.",
            source_chunk_id=first_chunk.id,
            source_span=(0, min(80, len(first_chunk.text))),
        ),
    ]


# ---------------------------------------------------------------------------
# Helper: seed LLM cache for sufficiency judge
# ---------------------------------------------------------------------------


def _seed_judge_cache(db_path: Path, claims: list[Claim]) -> None:
    """Seed llm_cache for the SufficiencyJudge call.

    All claims are marked SUPPORTED so the research agent proceeds with the
    BUY path (sufficiency.aggregate_status() == "ok").
    """
    capturing = _CapturingMessagesClient()
    judge = SufficiencyJudge(
        client=capturing,
        model=_JUDGE_MODEL,
    )

    from firm.agents.research import _format_question
    from firm.grounding.judge import JudgeResponseError

    question = _format_question("AAPL")
    # The capturing client returns a stub response that is not valid JSON, so
    # ``assess`` raises ``JudgeResponseError`` *after* the prompt has been
    # captured by ``messages_create``.  We only need the captured prompt here.
    try:
        judge.assess(question=question, claims=claims)
    except JudgeResponseError:
        pass

    assert len(capturing.calls) == 1, (
        f"judge made {len(capturing.calls)} calls, expected 1"
    )
    call = capturing.calls[0]

    canned_text = json.dumps({
        "assessments": [
            {
                "claim_id": f"c{i + 1}",
                "status": "SUPPORTED",
                "rationale": "The claim is directly supported by the cited text.",
            }
            for i in range(len(claims))
        ],
        "overall_reasoning": "All claims are fully supported by the retrieved evidence.",
    })
    response_dict: dict[str, object] = {
        "content": [{"type": "text", "text": canned_text}],
        "model": call["model"],
        "usage": {"input_tokens": 50, "output_tokens": 30},
    }

    prompt_hash = hash_prompt(
        system=call["system"],
        messages=call["messages"],
        tools=call["tools"],
    )
    cache = LlmCache(db_path=db_path, clock=_FIXTURE_CLOCK)
    cache.put(
        prompt_hash=prompt_hash,
        model=call["model"],
        response_json=json.dumps(response_dict),
        input_tokens=50,
        output_tokens=30,
    )


# ---------------------------------------------------------------------------
# Helper: seed LLM cache for PM voter (3 lenses)
# ---------------------------------------------------------------------------


def _seed_pm_voter_cache(db_path: Path, claims: list[Claim]) -> None:
    """Seed llm_cache for the three PmVoter calls (quality, valuation, catalyst).

    ``make_pm`` uses ``research.rationale`` as the question argument for each
    ``voter.vote()`` call.  The research rationale (for the ok/BUY path) is::

        " ".join(c.text for c in claims)

    We use the same claims returned by ``_seed_extractor_cache`` so the
    research_rationale string matches exactly.

    All three lenses vote BUY so aggregate_votes produces a unanimous BUY →
    the PM Decision triggers risk → execution → confirmed outbox row.
    """
    research_rationale = " ".join(c.text for c in claims)

    capturing = _CapturingMessagesClient()
    voter = PmVoter(client=capturing, model=_PM_MODEL)
    cache = LlmCache(db_path=db_path, clock=_FIXTURE_CLOCK)

    claim_ids = [f"c{i + 1}" for i in range(len(claims))]
    canned_buy_response = json.dumps({
        "vote": "BUY",
        "confidence": 0.82,
        "rationale": "Strong evidence of quality / value / catalyst in the cited claims.",
        "cited_claim_ids": claim_ids,
    })

    from firm.agents.pm import PmVoteSchemaError

    for lens in (PmLens.QUALITY, PmLens.VALUATION, PmLens.CATALYST):
        # The capturing client returns a stub response that is not valid JSON,
        # so ``voter.vote`` raises ``PmVoteSchemaError`` *after* the prompt has
        # been captured.  We only need the captured prompt here.
        try:
            voter.vote(
                lens=lens,
                question=research_rationale,
                claims=claims,
                research_rationale=research_rationale,
            )
        except PmVoteSchemaError:
            pass
        call = capturing.calls[-1]

        response_dict: dict[str, object] = {
            "content": [{"type": "text", "text": canned_buy_response}],
            "model": call["model"],
            "usage": {"input_tokens": 60, "output_tokens": 20},
        }
        prompt_hash = hash_prompt(
            system=call["system"],
            messages=call["messages"],
            tools=call["tools"],
        )
        cache.put(
            prompt_hash=prompt_hash,
            model=call["model"],
            response_json=json.dumps(response_dict),
            input_tokens=60,
            output_tokens=20,
        )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.requires_models
def test_grounded_demo_produces_confirmed_trade_with_citations(
    tmp_path: Path,
) -> None:
    """End-to-end grounded demo: ingest → retrieve → seed cache → run → assert.

    Verifies the full Plan 2 grounded pipeline without any network access:
    all LLM calls are served from the SQLite cache and all Qdrant ops use a
    local filesystem instance.
    """
    # ------------------------------------------------------------------ #
    # Phase 0: precomputed parquets (CLI tool constructors need them)     #
    # ------------------------------------------------------------------ #
    _ensure_precomputed_parquets()

    # ------------------------------------------------------------------ #
    # Phase 1: layout                                                      #
    # ------------------------------------------------------------------ #
    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_local_path = tmp_path / "qdrant"
    qdrant_local_path.mkdir()

    reports_root = tmp_path / "reports"
    db_path = tmp_path / "firm.db"

    # ------------------------------------------------------------------ #
    # Phase 2: fixture YAMLs                                               #
    # ------------------------------------------------------------------ #
    fixture_rag_yaml = tmp_path / "rag.yaml"
    fixture_rag_yaml.write_text(
        "\n".join([
            "corpus:",
            "  financebench:",
            "    split: train",
            "    max_docs: 2",
            "",
            "chunk:",
            f"  target_tokens: {_FIXTURE_CHUNK_TARGET_TOKENS}",
            f"  overlap_tokens: {_FIXTURE_CHUNK_OVERLAP_TOKENS}",
            "",
            "embedding:",
            "  dense_model: nomic-ai/nomic-embed-text-v1.5",
            "  dense_dim: 768",
            "  sparse: bm25",
            "",
            "retrieval:",
            "  top_k_retrieve: 8",
            "  top_k_rerank: 4",
            "",
            "rerank:",
            "  model: BAAI/bge-reranker-v2-m3",
            "  score_floor: 0.0",
            "",
            "contextual:",
            f"  summary_model: {_AUGMENTATION_MODEL}",
            "",
            "qdrant:",
            f"  collection: {_COLLECTION}",
            "  url_env: QDRANT_URL",
        ]),
        encoding="utf-8",
    )

    fixture_llm_yaml = tmp_path / "llm.yaml"
    fixture_llm_yaml.write_text(
        "\n".join([
            "research:",
            f"  model: {_RESEARCH_MODEL}",
            "  max_tokens: 4096",
            "  temperature: 0.0",
            "",
            "judge:",
            f"  model: {_JUDGE_MODEL}",
            "  max_tokens: 2048",
            "  temperature: 0.0",
            "",
            "pm:",
            f"  model: {_PM_MODEL}",
            "  max_tokens: 1024",
            "  temperature: 0.0",
        ]),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Phase 3: init DB + seed augmentation cache                          #
    # ------------------------------------------------------------------ #
    init_db(db_path)
    _seed_augmentation_cache(db_path)

    # ------------------------------------------------------------------ #
    # Phase 4: common subprocess environment                              #
    # ------------------------------------------------------------------ #
    env = os.environ.copy()
    env["FIRM_DB_PATH"] = str(db_path)
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_local_path)
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_RAG_CONFIG"] = str(fixture_rag_yaml)
    env["FIRM_LLM_CONFIG"] = str(fixture_llm_yaml)
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_REPLAY_AT"] = _REPLAY_AT
    env["FIRM_REPORTS_ROOT"] = str(reports_root)
    # Seed a non-zero AAPL position so Risk's escalate_new_ticker check does not
    # trigger (escalate_new_ticker is now true; this test verifies the BUY
    # pass-through path, not the HITL path — that is covered by T31).
    env["FIRM_INITIAL_POSITIONS"] = '{"AAPL": "10"}'
    # ANTHROPIC_API_KEY is not needed in cached mode, but from_env() reads it.
    # Set a dummy so no ValueError is raised.
    env.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-cached-mode")
    # Force UTF-8 stdio so we can decode subprocess output on Windows (the
    # default cp1252 codec chokes on non-ASCII characters in tracebacks).
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("QDRANT_URL", None)

    # ------------------------------------------------------------------ #
    # Phase 5: firm ingest                                                #
    # ------------------------------------------------------------------ #
    ingest_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "ingest", "--config", str(fixture_rag_yaml)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    assert ingest_result.returncode == 0, (
        f"firm ingest exited {ingest_result.returncode}\n"
        f"stdout:\n{ingest_result.stdout}\n"
        f"stderr:\n{ingest_result.stderr}"
    )

    # Confirm Qdrant was populated.
    _qc = QdrantClient(path=str(qdrant_local_path))
    chunk_count = _qc.count(collection_name=_COLLECTION, exact=True).count
    _qc.close()
    assert chunk_count > 0, "Qdrant collection empty after ingest"

    # ------------------------------------------------------------------ #
    # Phase 6: in-process retrieval to get the exact chunks               #
    # ------------------------------------------------------------------ #
    retrieved_chunks = _retrieve_aapl_chunks(qdrant_local_path)
    assert retrieved_chunks, (
        "In-process retrieval returned no AAPL chunks; "
        "check that ingest seeded the collection correctly."
    )

    # ------------------------------------------------------------------ #
    # Phase 7: seed LLM cache for run-time agents                        #
    # ------------------------------------------------------------------ #
    claims = _seed_extractor_cache(db_path, retrieved_chunks)
    _seed_judge_cache(db_path, claims)
    _seed_pm_voter_cache(db_path, claims)

    # ------------------------------------------------------------------ #
    # Phase 8: firm run --once                                            #
    # ------------------------------------------------------------------ #
    run_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    assert run_result.returncode == 0, (
        f"firm run exited {run_result.returncode}\n"
        f"stdout:\n{run_result.stdout}\n"
        f"stderr:\n{run_result.stderr}"
    )

    # ------------------------------------------------------------------ #
    # Phase 9: assertions                                                 #
    # ------------------------------------------------------------------ #
    # 9a: outbox has 1 confirmed row.
    with closing(get_conn(db_path)) as conn:
        outbox_confirmed = conn.execute(
            "SELECT COUNT(*) AS n FROM outbox WHERE status='confirmed'"
        ).fetchone()["n"]

    assert outbox_confirmed == 1, (
        f"expected 1 confirmed outbox row, got {outbox_confirmed}\n"
        f"run stdout:\n{run_result.stdout}\n"
        f"run stderr:\n{run_result.stderr}"
    )

    # 9b: decisions table has >= 3 rows (research + pm + risk per heartbeat).
    # The reporter scans WorkingState for top-level Decision instances; one
    # heartbeat produces exactly three (research_decision, pm_decision,
    # risk_decision).  Plan 3 may emit additional per-voter / per-stage
    # Decisions; the assertion is intentionally permissive.
    with closing(get_conn(db_path)) as conn:
        decisions_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions"
        ).fetchone()["n"]

    assert decisions_count >= 3, (
        f"expected >= 3 decisions rows, got {decisions_count}\n"
        f"run stdout:\n{run_result.stdout}\n"
        f"run stderr:\n{run_result.stderr}"
    )

    # 9c: JSONL report exists and contains a citation with chunk_id.
    report_date_dir = reports_root / "2024-03-13"
    report_file = report_date_dir / "decisions.jsonl"
    assert report_file.exists(), (
        f"expected report file at {report_file}\n"
        f"run stdout:\n{run_result.stdout}\n"
        f"run stderr:\n{run_result.stderr}"
    )

    found_citation_with_chunk_id = False
    with report_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for _key, val in record.items():
                if not isinstance(val, dict):
                    continue
                citations_raw = val.get("citations")
                if not isinstance(citations_raw, list):
                    continue
                for cit in citations_raw:
                    if isinstance(cit, dict) and cit.get("chunk_id"):
                        found_citation_with_chunk_id = True
                        break
                if found_citation_with_chunk_id:
                    break
            if found_citation_with_chunk_id:
                break

    assert found_citation_with_chunk_id, (
        f"No citation with chunk_id found in {report_file}\n"
        f"Report contents:\n{report_file.read_text(encoding='utf-8')}"
    )


# ---------------------------------------------------------------------------
# Helper: read risk_decision from LangGraph checkpoint (T31)
# ---------------------------------------------------------------------------


def _read_risk_decision_from_checkpoint(db_path: Path, thread_id: str) -> Decision:
    """Deserialize the ``risk_decision`` from the LangGraph SQLite checkpoint.

    Opens a minimal graph shell (same topology, same ``_FIRM_SERDE``) so that
    ``graph.get_state()`` can deserialize firm's Decision types from the
    msgpack checkpoint without running any nodes.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn, serde=_FIRM_SERDE)

    # Mirror the real graph topology so get_state() can match the checkpoint.
    g = StateGraph(WorkingState)
    for node_name in ("monitor", "research", "pm", "risk", "hitl", "execution", "reporter"):
        g.add_node(node_name, lambda s: {})  # noqa: B023 -- dummy; never executed
    g.set_entry_point("monitor")
    g.add_edge("monitor", "research")
    g.add_edge("research", "pm")
    g.add_edge("pm", "risk")
    g.add_conditional_edges(
        "risk",
        lambda s: "hitl",
        {"hitl": "hitl", "execution": "execution"},
    )
    g.add_edge("hitl", "execution")
    g.add_edge("execution", "reporter")
    g.add_edge("reporter", END)
    graph = g.compile(checkpointer=saver, interrupt_before=["hitl"])

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    conn.close()

    risk_decision = state.values.get("risk_decision")
    if risk_decision is None:
        raise RuntimeError(
            f"risk_decision not found in checkpoint for thread_id={thread_id!r}; "
            f"state keys: {list(state.values.keys())}"
        )
    if not isinstance(risk_decision, Decision):
        raise TypeError(
            f"Expected Decision, got {type(risk_decision).__name__}: {risk_decision!r}"
        )
    return risk_decision


# ---------------------------------------------------------------------------
# T31: new-ticker → ESCALATE → HITL → mark_approved → resume → confirmed
# ---------------------------------------------------------------------------


@pytest.mark.requires_models
def test_grounded_demo_new_ticker_routes_through_hitl(
    tmp_path: Path,
) -> None:
    """T31: research on a new ticker (0 existing shares) routes through HITL.

    Flow:
    1. Ingest fixture corpus (subprocess).
    2. Retrieve AAPL chunks + seed LLM caches (in-process, same as T30).
    3. First ``firm run --once`` → Risk emits ESCALATE (new ticker, 0 shares)
       → LangGraph interrupt before hitl → subprocess exits 0.
    4. Read ``risk_decision`` from checkpoint. Persist it to ``decisions`` table
       (satisfying the FK on hitl_queue), then call ``mark_approved``.
    5. Second ``firm run --once`` with same thread_id (same FIRM_REPLAY_AT) →
       hitl resumes, finds pre-approved row → ``hitl_approved=True`` →
       execution unwraps EscalatePayload and submits the trade → outbox confirmed.
    6. Assert: hitl_queue has 1 approved row, outbox has 1 confirmed row.

    NOTE: ``FIRM_INITIAL_POSITIONS`` is intentionally NOT set so FakeBroker
    starts with 0 AAPL shares — triggering ``escalate_new_ticker`` in Risk.
    """
    # ------------------------------------------------------------------ #
    # Phase 0: precomputed parquets                                       #
    # ------------------------------------------------------------------ #
    _ensure_precomputed_parquets()

    # ------------------------------------------------------------------ #
    # Phase 1: layout                                                      #
    # ------------------------------------------------------------------ #
    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_local_path = tmp_path / "qdrant"
    qdrant_local_path.mkdir()

    reports_root = tmp_path / "reports"
    db_path = tmp_path / "firm.db"

    # ------------------------------------------------------------------ #
    # Phase 2: fixture YAMLs                                               #
    # ------------------------------------------------------------------ #
    fixture_rag_yaml = tmp_path / "rag.yaml"
    fixture_rag_yaml.write_text(
        "\n".join([
            "corpus:",
            "  financebench:",
            "    split: train",
            "    max_docs: 2",
            "",
            "chunk:",
            f"  target_tokens: {_FIXTURE_CHUNK_TARGET_TOKENS}",
            f"  overlap_tokens: {_FIXTURE_CHUNK_OVERLAP_TOKENS}",
            "",
            "embedding:",
            "  dense_model: nomic-ai/nomic-embed-text-v1.5",
            "  dense_dim: 768",
            "  sparse: bm25",
            "",
            "retrieval:",
            "  top_k_retrieve: 8",
            "  top_k_rerank: 4",
            "",
            "rerank:",
            "  model: BAAI/bge-reranker-v2-m3",
            "  score_floor: 0.0",
            "",
            "contextual:",
            f"  summary_model: {_AUGMENTATION_MODEL}",
            "",
            "qdrant:",
            f"  collection: {_COLLECTION}",
            "  url_env: QDRANT_URL",
        ]),
        encoding="utf-8",
    )

    fixture_llm_yaml = tmp_path / "llm.yaml"
    fixture_llm_yaml.write_text(
        "\n".join([
            "research:",
            f"  model: {_RESEARCH_MODEL}",
            "  max_tokens: 4096",
            "  temperature: 0.0",
            "",
            "judge:",
            f"  model: {_JUDGE_MODEL}",
            "  max_tokens: 2048",
            "  temperature: 0.0",
            "",
            "pm:",
            f"  model: {_PM_MODEL}",
            "  max_tokens: 1024",
            "  temperature: 0.0",
        ]),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Phase 3: init DB + seed augmentation cache                          #
    # ------------------------------------------------------------------ #
    init_db(db_path)
    _seed_augmentation_cache(db_path)

    # ------------------------------------------------------------------ #
    # Phase 4: common subprocess environment                              #
    # ------------------------------------------------------------------ #
    env = os.environ.copy()
    env["FIRM_DB_PATH"] = str(db_path)
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_local_path)
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_RAG_CONFIG"] = str(fixture_rag_yaml)
    env["FIRM_LLM_CONFIG"] = str(fixture_llm_yaml)
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_REPLAY_AT"] = _REPLAY_AT
    env["FIRM_REPORTS_ROOT"] = str(reports_root)
    # Intentionally no FIRM_INITIAL_POSITIONS — FakeBroker has 0 AAPL shares,
    # triggering escalate_new_ticker in Risk.
    env.pop("FIRM_INITIAL_POSITIONS", None)
    env.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-cached-mode")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("QDRANT_URL", None)

    # ------------------------------------------------------------------ #
    # Phase 5: firm ingest                                                #
    # ------------------------------------------------------------------ #
    ingest_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "ingest", "--config", str(fixture_rag_yaml)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    assert ingest_result.returncode == 0, (
        f"firm ingest exited {ingest_result.returncode}\n"
        f"stdout:\n{ingest_result.stdout}\n"
        f"stderr:\n{ingest_result.stderr}"
    )

    # Confirm Qdrant was populated.
    _qc = QdrantClient(path=str(qdrant_local_path))
    chunk_count = _qc.count(collection_name=_COLLECTION, exact=True).count
    _qc.close()
    assert chunk_count > 0, "Qdrant collection empty after ingest"

    # ------------------------------------------------------------------ #
    # Phase 6: in-process retrieval                                       #
    # ------------------------------------------------------------------ #
    retrieved_chunks = _retrieve_aapl_chunks(qdrant_local_path)
    assert retrieved_chunks, (
        "In-process retrieval returned no AAPL chunks; "
        "check that ingest seeded the collection correctly."
    )

    # ------------------------------------------------------------------ #
    # Phase 7: seed LLM cache for run-time agents                        #
    # ------------------------------------------------------------------ #
    claims = _seed_extractor_cache(db_path, retrieved_chunks)
    _seed_judge_cache(db_path, claims)
    _seed_pm_voter_cache(db_path, claims)

    # ------------------------------------------------------------------ #
    # Phase 8a: first firm run --once → interrupt before hitl            #
    # ------------------------------------------------------------------ #
    # The thread_id used by cli.py is clock.now().isoformat() which, for a
    # ReplayClock seeded with FIRM_REPLAY_AT, is always _REPLAY_AT.
    thread_id = _REPLAY_AT

    run1_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    # LangGraph interrupt_before exits cleanly (the invoke returns normally
    # after the checkpoint is saved); exit code must be 0.
    assert run1_result.returncode == 0, (
        f"firm run (first invoke) exited {run1_result.returncode}\n"
        f"stdout:\n{run1_result.stdout}\n"
        f"stderr:\n{run1_result.stderr}"
    )

    # ------------------------------------------------------------------ #
    # Phase 8b: read risk_decision, persist decisions row, pre-approve   #
    # ------------------------------------------------------------------ #
    # The LangGraph checkpoint holds risk_decision in the state after risk
    # ran but before hitl.  Deserialize it via a graph shell.
    risk_decision: Decision = _read_risk_decision_from_checkpoint(db_path, thread_id)

    assert risk_decision.action.value == "ESCALATE", (
        f"Expected ESCALATE from Risk (new ticker + 0 shares + escalate_new_ticker=true), "
        f"got {risk_decision.action.value!r}.  "
        f"Make sure escalate_new_ticker is true in config/policy.yaml and FIRM_INITIAL_POSITIONS is unset."
    )

    # The hitl_queue has a FK → decisions; insert the risk_decision row so
    # the FK is satisfied before mark_approved creates the pre-approved row.
    _persist_decisions_from_state(
        {"risk_decision": risk_decision}, db_path, _FIXTURE_CLOCK
    )

    # Pre-approve: insert an 'approved' hitl_queue row so that when the
    # graph resumes and hitl runs its INSERT OR IGNORE, it finds this row
    # and SELECT status returns 'approved' immediately.
    mark_approved(
        db_path=db_path,
        decision_id=risk_decision.id,
        approver="t31-test",
        clock=_FIXTURE_CLOCK,
    )

    # Verify pre-approval is visible.
    with closing(get_conn(db_path)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (risk_decision.id,)
        ).fetchone()
    assert row is not None, "hitl_queue row missing after mark_approved pre-approve"
    assert row["status"] == "approved", (
        f"Expected 'approved', got {row['status']!r}"
    )

    # ------------------------------------------------------------------ #
    # Phase 8c: second firm run --once → resume → execution → confirmed  #
    # ------------------------------------------------------------------ #
    run2_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    assert run2_result.returncode == 0, (
        f"firm run (second invoke / resume) exited {run2_result.returncode}\n"
        f"stdout:\n{run2_result.stdout}\n"
        f"stderr:\n{run2_result.stderr}"
    )

    # ------------------------------------------------------------------ #
    # Phase 9: assertions                                                 #
    # ------------------------------------------------------------------ #
    # 9a: hitl_queue shows the approved row.
    with closing(get_conn(db_path)) as conn:
        hitl_row = conn.execute(
            "SELECT status, approver FROM hitl_queue WHERE decision_id=?",
            (risk_decision.id,),
        ).fetchone()
    assert hitl_row is not None, "hitl_queue row missing after both runs"
    assert hitl_row["status"] == "approved", (
        f"hitl_queue status expected 'approved', got {hitl_row['status']!r}"
    )

    # 9b: outbox has 1 confirmed row (the HITL-approved AAPL trade executed).
    with closing(get_conn(db_path)) as conn:
        outbox_confirmed = conn.execute(
            "SELECT COUNT(*) AS n FROM outbox WHERE status='confirmed'"
        ).fetchone()["n"]
    assert outbox_confirmed == 1, (
        f"Expected 1 confirmed outbox row after HITL-approved resume, got {outbox_confirmed}\n"
        f"run2 stdout:\n{run2_result.stdout}\n"
        f"run2 stderr:\n{run2_result.stderr}"
    )

    # 9c: decisions table has >= 3 rows (research + pm + risk at minimum).
    with closing(get_conn(db_path)) as conn:
        decisions_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions"
        ).fetchone()["n"]
    assert decisions_count >= 3, (
        f"Expected >= 3 decisions rows, got {decisions_count}\n"
        f"run2 stdout:\n{run2_result.stdout}\n"
        f"run2 stderr:\n{run2_result.stderr}"
    )
