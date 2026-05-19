# Plan 2: RAG + Grounding + LLM Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 1's deterministic Research/PM stubs with LLM-backed agents grounded in real SEC filings via FinanceBench. At the end of Plan 2, `make ingest` populates a Qdrant vector store from the FinanceBench corpus, and `make demo` produces a confirmed paper trade whose research thesis is supported by span-anchored citations against actual filings, gated by a sufficiency judge, with numeric metrics sourced from MCP tools rather than the LLM, and a vote-of-3 PM synthesis. `escalate_new_ticker` flips back to `true` because the grounded research can now defend a new position.

**Architecture:** A two-phase pipeline. **Ingest** (offline, `make ingest`): FinanceBench filings → table-to-prose preprocessing → chunking → contextual augmentation (Haiku per-chunk doc summaries) → dense (nomic-embed) + sparse (BM25) embeddings → Qdrant collection with `published_at` payload. **Runtime** (`make demo`): Research agent issues a query → hybrid retrieve (top-50) with PIT filter → bge-reranker-v2-m3 (top-8) → Sonnet 4.6 + Citations API extracts `Claim` objects → Haiku sufficiency judge classifies SUPPORTED/PARTIAL/UNSUPPORTED → on all-SUPPORTED, three parallel Sonnet PM voters (quality/valuation/catalyst lenses) → deterministic Python aggregation → existing Risk/HITL/Execution/Reporter pipeline. An `llm_cache` SQLite table makes the demo deterministic across replays despite real API calls.

```
INGEST (offline, once per corpus)
  FinanceBench dataset
    └─> CorpusSource → tables→prose → chunker → doc_summary (Haiku) → embed (nomic + BM25)
                                                                       └─> Qdrant collection

RUNTIME (per heartbeat)
  Research query
    └─> hybrid retrieve (50) → PIT filter → rerank (8) → Sonnet+Citations → Claim[]
                                                                              └─> Sufficiency judge (Haiku)
                                                                                    ├─ SUPPORTED → PM
                                                                                    ├─ PARTIAL  → ESCALATE
                                                                                    └─ UNSUPPORTED → REFUSE
  PM
    └─> 3× Sonnet (quality / valuation / catalyst) → Python aggregate → Risk → HITL → Execution → Reporter

  Tool calls (no LLM arithmetic): fundamentals.get_ratio, risk.get_metric
```

**Tech Stack (additions vs Plan 1):**
- `anthropic>=0.39` — Sonnet 4.6 + Haiku 4.5 client, Citations API
- `qdrant-client>=1.11` — vector DB client (Qdrant runs as docker-compose service)
- `sentence-transformers>=3.0` — local nomic-embed and bge-reranker-v2-m3 inference
- `datasets>=2.20` — Hugging Face `PatronusAI/financebench` loader
- `rank-bm25>=0.2.2` — sparse BM25 token weights for hybrid retrieval
- `beautifulsoup4>=4.12`, `lxml>=5.0` — HTML/table extraction during ingest
- `tiktoken>=0.7` — token-aware chunking
- New service in `docker-compose.yml`: `qdrant/qdrant:v1.11.0` with a named volume

---

## File Structure

New and modified files relative to Plan 1. Unchanged files are not listed.

```
ai-investment-firm/
├── pyproject.toml                          # MODIFIED: add anthropic, qdrant-client, sentence-transformers, datasets, rank-bm25, bs4, lxml, tiktoken
├── docker-compose.yml                      # MODIFIED: add qdrant service + named volume
├── Makefile                                # MODIFIED: add `ingest` target
├── .env.example                            # MODIFIED: ANTHROPIC_API_KEY, QDRANT_URL, FIRM_LLM_MODE
├── config/
│   ├── policy.yaml                         # MODIFIED at T32: escalate_new_ticker → true
│   ├── rag.yaml                            # NEW: corpus, embedding, retrieval params
│   └── llm.yaml                            # NEW: model IDs, max_tokens, temperatures
├── data/
│   └── precomputed/
│       └── fundamentals.parquet            # NEW: pre-computed ratios for MCP tool (T22)
├── firm/
│   ├── core/
│   │   └── models.py                       # MODIFIED: extend Citation, add ClaimSupport, SufficiencyResult, PMVote
│   ├── db/
│   │   └── schema.sql                      # MODIFIED: add llm_cache, ingest_runs tables
│   ├── orchestrator/
│   │   └── state.py                        # MODIFIED: add retrieved_chunks, claims, sufficiency_result, pm_votes
│   ├── rag/                                # NEW MODULE
│   │   ├── __init__.py
│   │   ├── chunk.py                        # Chunk dataclass + chunker
│   │   ├── preprocess.py                   # table-to-prose, ticker-aware tokenization
│   │   ├── source.py                       # CorpusSource Protocol + FilingDoc
│   │   ├── financebench.py                 # FinanceBench adapter
│   │   ├── embed.py                        # NomicEmbedder + BM25Sparse
│   │   ├── contextual.py                   # Haiku doc summary augmentation
│   │   ├── qdrant_store.py                 # VectorStore wrapper around qdrant-client
│   │   ├── retrieve.py                     # HybridRetriever (dense+sparse, PIT filter)
│   │   ├── rerank.py                       # BgeReranker
│   │   └── ingest.py                       # end-to-end ingest pipeline + CLI entry
│   ├── llm/                                # NEW MODULE
│   │   ├── __init__.py
│   │   ├── cache.py                        # SQLite-backed (prompt_hash, model) cache
│   │   ├── anthropic_client.py             # thin Anthropic wrapper using cache
│   │   ├── citations.py                    # CitedClaimExtractor Protocol + AnthropicCitationsExtractor
│   │   └── prompts.py                      # system/user prompt templates (research, judge, PM)
│   ├── grounding/                          # NEW MODULE
│   │   ├── __init__.py
│   │   ├── judge.py                        # SufficiencyJudge (Haiku, JSON-mode)
│   │   └── schema.py                       # ClaimSupport, SufficiencyResult Pydantic models
│   ├── tools/                              # NEW MODULE (least-privilege MCP-shaped tools)
│   │   ├── __init__.py
│   │   ├── fundamentals.py                 # get_ratio(ticker, ratio_name, as_of) -> Decimal
│   │   └── risk_metrics.py                 # get_metric(ticker, metric, window) -> Decimal
│   ├── agents/
│   │   ├── research.py                     # MODIFIED: LLM-backed grounded research
│   │   └── pm.py                           # MODIFIED: vote-of-3 + Python aggregation
│   └── cli.py                              # MODIFIED: add `ingest` subcommand
├── scripts/
│   └── precompute_fundamentals.py          # NEW: builds data/precomputed/fundamentals.parquet from FinanceBench
└── tests/
    ├── fixtures/
    │   ├── chunks_aapl_q3_2024.json        # NEW: 8 synthetic chunks for retrieval/judge tests
    │   └── financebench_two_docs.json      # NEW: 2-doc subset for ingest tests
    ├── unit/
    │   ├── test_chunker.py                 # NEW
    │   ├── test_preprocess.py              # NEW
    │   ├── test_corpus_source.py           # NEW
    │   ├── test_financebench_adapter.py    # NEW (offline; fixture-driven)
    │   ├── test_embed.py                   # NEW
    │   ├── test_contextual.py              # NEW
    │   ├── test_qdrant_store.py            # NEW (uses qdrant testcontainer or in-memory mode)
    │   ├── test_retrieve.py                # NEW
    │   ├── test_rerank.py                  # NEW
    │   ├── test_llm_cache.py               # NEW
    │   ├── test_anthropic_client.py        # NEW
    │   ├── test_citations.py               # NEW
    │   ├── test_judge.py                   # NEW
    │   ├── test_fundamentals_tool.py       # NEW
    │   ├── test_risk_metrics_tool.py       # NEW
    │   ├── test_pm_vote_aggregation.py     # NEW
    │   ├── test_research_agent.py          # NEW
    │   └── test_pm_agent.py                # NEW
    └── integration/
        ├── test_ingest_pipeline.py         # NEW
        ├── test_retrieval_pit.py           # NEW (PIT enforcement CI invariant)
        ├── test_research_end_to_end.py     # NEW
        └── test_end_to_end_grounded.py     # NEW (full demo with cached LLM responses)
```

---

## Task 1: Update dependencies and config scaffolding

**Files:**
- Modified: `pyproject.toml`, `.env.example`
- Created: `config/rag.yaml`, `config/llm.yaml`

**Tests:**
- `tests/unit/test_config.py` (extend existing): assert `rag.yaml` and `llm.yaml` load cleanly

- [ ] **Step 1: Extend `tests/unit/test_config.py`** with two new tests asserting `load_rag_config(Path("config/rag.yaml"))` returns a `RagConfig` with non-empty `corpus.financebench.split`, `embedding.dense_model`, `retrieval.top_k_retrieve == 50`, `retrieval.top_k_rerank == 8`, and `load_llm_config(Path("config/llm.yaml"))` returns an `LlmConfig` exposing `research.model`, `judge.model`, `pm.model`, plus `max_tokens` for each (4096/2048/1024).

- [ ] **Step 2: Run — verify fail** (`pytest tests/unit/test_config.py -v`).

- [ ] **Step 3: Add dependencies to `pyproject.toml`**: `anthropic>=0.39`, `qdrant-client>=1.11`, `sentence-transformers>=3.0`, `datasets>=2.20`, `rank-bm25>=0.2.2`, `beautifulsoup4>=4.12`, `lxml>=5.0`, `tiktoken>=0.7`. Keep all Plan 1 deps.

- [ ] **Step 4: Extend `.env.example`** with `ANTHROPIC_API_KEY=`, `QDRANT_URL=http://localhost:6333`, `FIRM_LLM_MODE=cached` (modes: `live`, `cached`, `record`), `FIRM_RAG_CONFIG=config/rag.yaml`, `FIRM_LLM_CONFIG=config/llm.yaml`.

- [ ] **Step 5: Create `config/rag.yaml`** with `corpus.financebench.split: train`, `corpus.financebench.max_docs: null`, `chunk.target_tokens: 512`, `chunk.overlap_tokens: 64`, `embedding.dense_model: nomic-ai/nomic-embed-text-v1.5`, `embedding.dense_dim: 768`, `embedding.sparse: bm25`, `retrieval.top_k_retrieve: 50`, `retrieval.top_k_rerank: 8`, `rerank.model: BAAI/bge-reranker-v2-m3`, `rerank.score_floor: 0.3`, `contextual.summary_model: claude-haiku-4-5`, `qdrant.collection: firm_chunks`, `qdrant.url_env: QDRANT_URL`.

- [ ] **Step 6: Create `config/llm.yaml`** with three sections: `research: {model: claude-sonnet-4-6, max_tokens: 4096, temperature: 0.0}`, `judge: {model: claude-haiku-4-5, max_tokens: 2048, temperature: 0.0}`, `pm: {model: claude-sonnet-4-6, max_tokens: 1024, temperature: 0.0}`. Each value typed via Pydantic.

- [ ] **Step 7: Extend `firm/core/config.py`** with `RagConfig`, `LlmConfig`, `load_rag_config`, `load_llm_config`. Mirror the `PolicyConfig` pattern (nested Pydantic models + `yaml.safe_load`).

- [ ] **Step 8: Install + run tests.**

**Verify with:**
```
pip install -e ".[dev]"
ruff check firm tests
mypy firm
pytest tests/unit/test_config.py -v
```

**Risks / notes:** Pin `nomic-embed-text-v1.5` to the sentence-transformers path; the HF page has both nomic-native and ST loaders. Choose one in T7 and stay with it.

---

## Task 2: Database schema additions (llm_cache, ingest_runs)

**Files:**
- Modified: `firm/db/schema.sql`
- Test: `tests/unit/test_migrations.py` (extend)

**Tests:**
- Extend `test_init_db_creates_all_tables` to also assert `llm_cache` and `ingest_runs` are present.
- New `test_llm_cache_unique_on_prompt_hash_and_model`: insert, then second insert with same `(prompt_hash, model)` must conflict.

- [ ] **Step 1: Update `tests/unit/test_migrations.py`** to expect the new tables and the unique constraint.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Edit `firm/db/schema.sql`** appending:
  ```sql
  CREATE TABLE IF NOT EXISTS llm_cache (
      prompt_hash    TEXT NOT NULL,
      model          TEXT NOT NULL,
      response_json  TEXT NOT NULL,
      input_tokens   INTEGER,
      output_tokens  INTEGER,
      created_at     TEXT NOT NULL,
      PRIMARY KEY (prompt_hash, model)
  );
  CREATE INDEX IF NOT EXISTS idx_llm_cache_created ON llm_cache(created_at);

  CREATE TABLE IF NOT EXISTS ingest_runs (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      started_at      TEXT NOT NULL,
      finished_at     TEXT,
      corpus          TEXT NOT NULL,
      docs_total      INTEGER NOT NULL DEFAULT 0,
      docs_completed  INTEGER NOT NULL DEFAULT 0,
      chunks_written  INTEGER NOT NULL DEFAULT 0,
      status          TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
      error           TEXT
  );
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.** `git add firm/db/schema.sql tests/unit/test_migrations.py && git commit -m "feat(db): llm_cache and ingest_runs tables"`

**Verify with:**
```
pytest tests/unit/test_migrations.py -v
ruff check firm tests
mypy firm
```

---

## Task 3: Extend core models (citation fields, RAG types)

**Files:**
- Modified: `firm/core/models.py`
- New: `firm/grounding/schema.py`
- Test: `tests/unit/test_models.py` (extend)

**Tests:**
- `test_citation_accepts_anthropic_fields`: Citation now carries optional `cited_text`, `document_index`, `document_title` so the Anthropic response can be persisted without lossy mapping.
- `test_claim_support_enum_values`: `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`.
- `test_sufficiency_result_aggregation_status`: helper `.aggregate_status()` returns `"ok"`/`"partial"`/`"insufficient"`.

- [ ] **Step 1: Write failing tests** in `tests/unit/test_models.py` and a new `tests/unit/test_grounding_schema.py`.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Extend `firm/core/models.py`**: add optional fields on `Citation` (`cited_text: str | None`, `document_index: int | None`, `document_title: str | None`). Keep `source_id`, `chunk_id`, `span` for back-compat. Existing Plan 1 callers (Risk/Reporter/etc.) construct Citation with positional args — preserve those positions; new fields must be keyword-only with defaults.

- [ ] **Step 4: Create `firm/grounding/schema.py`**:
  ```python
  from enum import StrEnum
  from pydantic import BaseModel, Field

  class ClaimSupport(StrEnum):
      SUPPORTED = "SUPPORTED"
      PARTIAL = "PARTIAL"
      UNSUPPORTED = "UNSUPPORTED"

  class ClaimAssessment(BaseModel):
      claim_id: str
      support: ClaimSupport
      reasoning: str = Field(min_length=1)

  class SufficiencyResult(BaseModel):
      claim_assessments: list[ClaimAssessment]
      overall_reasoning: str = ""

      def aggregate_status(self) -> str:
          supports = [a.support for a in self.claim_assessments]
          if any(s == ClaimSupport.UNSUPPORTED for s in supports):
              return "insufficient"
          if any(s == ClaimSupport.PARTIAL for s in supports):
              return "partial"
          return "ok"
  ```

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.** `git add firm/core/models.py firm/grounding/schema.py firm/grounding/__init__.py tests/unit/test_models.py tests/unit/test_grounding_schema.py && git commit -m "feat(core,grounding): extend Citation; ClaimSupport+SufficiencyResult"`

**Verify with:**
```
pytest tests/unit/test_models.py tests/unit/test_grounding_schema.py -v
mypy firm
```

**Risks / notes:** Plan 1's Risk/Reporter code instantiates `Citation` rarely (mostly `[]`); add the new fields with defaults to avoid touching their call sites.

---

## Task 4: Chunk dataclass and finance-aware chunker

**Files:**
- New: `firm/rag/__init__.py`, `firm/rag/chunk.py`
- Test: `tests/unit/test_chunker.py`

**Tests:**
- `test_chunk_target_size_within_tolerance`: chunks land within ±20% of `target_tokens` (512).
- `test_chunk_overlap_present`: adjacent chunks share the configured overlap.
- `test_chunk_preserves_published_at_and_metadata`: every chunk carries the source's `published_at`, `ticker`, `doc_id`, `section`.
- `test_chunker_rejects_doc_without_published_at`: raises `ValueError`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/chunk.py`** with:
  ```python
  class Chunk(BaseModel):
      id: str                 # f"{doc_id}::{idx:04d}"
      doc_id: str
      ticker: str
      published_at: datetime  # required, tz-aware
      section: str
      text: str
      char_span: tuple[int, int]   # offset into the parent doc
      token_count: int
      doc_summary: str | None = None  # filled by contextual augmentation (T8)
      metadata: dict[str, Any] = Field(default_factory=dict)

  def chunk_document(doc: FilingDoc, *, target_tokens: int, overlap_tokens: int) -> list[Chunk]: ...
  ```
  Use `tiktoken.encoding_for_model("gpt-4")` (`cl100k_base`) for stable counts. Reject docs whose `published_at` is None.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_chunker.py -v
ruff check firm/rag tests/unit/test_chunker.py
mypy firm/rag
```

**Risks / notes:** Use tiktoken's `cl100k_base` (Sonnet token count is close enough for chunk-size budgeting; we are not pricing tokens here).

---

## Task 5: Table-to-prose + ticker-aware preprocessing

**Files:**
- New: `firm/rag/preprocess.py`
- Test: `tests/unit/test_preprocess.py`, `tests/fixtures/financebench_two_docs.json`

**Tests:**
- `test_html_table_converted_to_prose`: an HTML `<table>` with `Q3 2024 / Revenue / 18,120` row produces prose like `"In Q3 2024, total revenue was $18,120 million"`.
- `test_ticker_tokens_preserved`: tokenizer keeps `$AAPL`, `BRK.B`, `10-K` as single tokens.
- `test_strips_boilerplate_and_normalizes_whitespace`.

- [ ] **Step 1: Author 2-doc fixture** `tests/fixtures/financebench_two_docs.json` with one 10-K excerpt (AAPL Q3 2024) and one 10-Q excerpt (NVDA Q3 2024). Include at least one HTML table per doc.

- [ ] **Step 2: Write tests.**

- [ ] **Step 3: Run — verify fail.**

- [ ] **Step 4: Implement `firm/rag/preprocess.py`** with `tables_to_prose(html: str) -> str` (uses `bs4` + `lxml`; iterates each `<table>`, replaces with deterministic prose) and `ticker_aware_tokens(text: str) -> list[str]` (regex preserving `\$[A-Z]+`, `[A-Z]+\.[A-Z]`, `\d+-[A-Z]`). Also NFKC-normalize + strip zero-width chars (spec §8.3 hygiene; preview for Plan 3).

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.**

**Verify with:**
```
pytest tests/unit/test_preprocess.py -v
mypy firm/rag/preprocess.py
```

**Risks / notes:** Table prose must be deterministic (no LLM call). Keep the row template strict — used to compare against a snapshot in CI.

---

## Task 6: CorpusSource Protocol and FilingDoc

**Files:**
- New: `firm/rag/source.py`
- Test: `tests/unit/test_corpus_source.py`

**Tests:**
- `test_filing_doc_requires_published_at_tz_aware`: rejects naive or null datetimes.
- `test_corpus_source_protocol_is_iterable`: a fake source implementing `iter_docs()` is recognized via `isinstance(obj, CorpusSource)` (runtime-checkable protocol).
- `test_filing_doc_round_trips_pydantic`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/source.py`**:
  ```python
  class FilingDoc(BaseModel):
      doc_id: str
      ticker: str
      filing_type: str        # 10-K | 10-Q | 8-K
      published_at: datetime  # tz-aware, REQUIRED
      title: str
      html: str
      url: str | None = None
      metadata: dict[str, Any] = Field(default_factory=dict)

  @runtime_checkable
  class CorpusSource(Protocol):
      name: str
      def iter_docs(self) -> Iterator[FilingDoc]: ...
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_corpus_source.py -v
```

**Risks / notes:** `CorpusSource` is the Plan-3 plug-in point for news/transcripts; keep it minimal.

---

## Task 7: FinanceBench adapter

**Files:**
- New: `firm/rag/financebench.py`
- Test: `tests/unit/test_financebench_adapter.py`

**Tests:**
- `test_adapter_loads_from_local_fixture`: bypasses HF download by feeding the adapter a fixture-mode kwarg.
- `test_adapter_skips_eval_qa_set`: ensures the 150 verified Q&A pairs are **excluded** so Plan 4 eval can reuse them.
- `test_published_at_is_parsed_tz_aware`.

- [ ] **Step 1: Author fixture-mode kwarg.** The adapter accepts `dataset_loader: Callable[[], Iterable[dict]] = _load_from_hf` for tests to inject. Real loader calls `datasets.load_dataset("PatronusAI/financebench")`.

- [ ] **Step 2: Write tests using `tests/fixtures/financebench_two_docs.json`.**

- [ ] **Step 3: Run — verify fail.**

- [ ] **Step 4: Implement `firm/rag/financebench.py`** with class `FinanceBenchSource(CorpusSource)`. `iter_docs()` yields `FilingDoc`s. Skip rows that are part of the Q&A eval split (mark via a `_holdout` flag stored in a constant file referenced from `config/rag.yaml`). Parse `published_at` from `filing_date` field; fail-closed if missing.

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.**

**Verify with:**
```
pytest tests/unit/test_financebench_adapter.py -v
mypy firm/rag/financebench.py
```

**Risks / notes:** Do not hit HF during unit tests; gate any live HF test with `@pytest.mark.skipif(os.environ.get("FIRM_ALLOW_HF_DOWNLOAD") != "1", ...)`.

---

## Task 8: Embedder (dense nomic + sparse BM25)

**Files:**
- New: `firm/rag/embed.py`
- Test: `tests/unit/test_embed.py`

**Tests:**
- `test_dense_embedder_returns_768_dim_unit_vectors`.
- `test_dense_embedder_is_deterministic`: identical input → identical output to 1e-6.
- `test_sparse_embedder_preserves_ticker_tokens`: vocabulary contains `$AAPL`, `BRK.B`.
- `test_batch_embed_handles_empty_input`.

- [ ] **Step 1: Write tests. Mock the sentence-transformers model load** with a fixture that returns a tiny stub model (or use a pytest marker to skip in CI without local model cache).

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/embed.py`** with `class NomicEmbedder` (lazy-loads `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)`; exposes `embed(texts: list[str]) -> np.ndarray`) and `class BM25Sparse` (fits on a corpus iterator, exposes `transform(text: str) -> dict[int, float]` returning sparse token→weight for Qdrant's sparse vector format).

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_embed.py -v -m "not requires_models"
```

**Risks / notes:** Mark heavy tests with `@pytest.mark.requires_models` so they can be skipped in subagent windows; one integration test (T11/T29) exercises the real model.

---

## Task 9: Contextual augmentation (Haiku per-chunk summaries)

**Files:**
- New: `firm/rag/contextual.py`
- Test: `tests/unit/test_contextual.py`

**Tests:**
- `test_summary_generated_once_per_doc_and_reused`: 8 chunks from the same doc → 1 Haiku call (verified via mock call count).
- `test_summary_attached_to_each_chunk`: `chunk.doc_summary` non-empty after augmentation.
- `test_summary_uses_llm_cache`: second invocation across runs hits the cache, not the API.

- [ ] **Step 1: Write tests with a `FakeAnthropicClient` injected.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/contextual.py`**:
  ```python
  class ContextualAugmenter:
      def __init__(self, *, client: AnthropicClient, model: str, max_tokens: int = 512) -> None: ...
      def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]:
          """Generate one doc-level summary (using doc title + first ~2k chars),
          prepend to each chunk via .doc_summary. Single Haiku call per doc."""
  ```
  The prompt follows the Anthropic Contextual Retrieval pattern (spec §6.1, refs §15): "Here is the document this chunk belongs to: <doc>...</doc>. Provide a 1–2 sentence summary that situates the chunk for retrieval. Do not paraphrase content; describe context only."

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_contextual.py -v
mypy firm/rag/contextual.py
```

**Risks / notes:** Depends on T15 (Anthropic client). Implement an injectable `AnthropicClient` Protocol here; the real client lands in T15. Tests use a stub.

---

## Task 10: Qdrant store wrapper

**Files:**
- New: `firm/rag/qdrant_store.py`
- Modified: `docker-compose.yml`
- Test: `tests/unit/test_qdrant_store.py`

**Tests:**
- `test_create_collection_with_named_vectors`: dense (768, cosine) + sparse vectors live on the same collection.
- `test_upsert_then_search_returns_chunk_id`.
- `test_payload_filter_published_at_excludes_future`.

- [ ] **Step 1: Edit `docker-compose.yml`** adding:
  ```yaml
    qdrant:
      image: qdrant/qdrant:v1.11.0
      ports: ["6333:6333"]
      volumes:
        - qdrant_data:/qdrant/storage
      healthcheck:
        test: ["CMD", "wget", "-qO-", "http://localhost:6333/readyz"]
        interval: 5s
        retries: 10
  volumes:
    qdrant_data:
  ```
  And `firm.depends_on: {qdrant: {condition: service_healthy}}`.

- [ ] **Step 2: Write tests using `qdrant_client.QdrantClient(":memory:")`** (Qdrant Python client supports in-memory mode for unit tests).

- [ ] **Step 3: Run — verify fail.**

- [ ] **Step 4: Implement `firm/rag/qdrant_store.py`** with `class VectorStore`. Methods: `create_collection(name, dense_dim)`, `upsert(name, chunks, dense_vecs, sparse_vecs)`, `search_dense(name, dense_vec, k, *, published_before)`, `search_sparse(name, sparse_vec, k, *, published_before)`, `search_hybrid(name, dense_vec, sparse_vec, k, *, published_before)`. Use Qdrant's `models.Filter(must=[FieldCondition(key="published_at", range=Range(lte=...))])`.

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.**

**Verify with:**
```
pytest tests/unit/test_qdrant_store.py -v
docker compose config -q
```

**Risks / notes:** Qdrant in-memory mode shares its API surface with the network client; if a CI host lacks the in-memory binary, skip with `pytest.importorskip("qdrant_client.local.qdrant_local")`.

---

## Task 11: Ingest pipeline orchestration

**Files:**
- New: `firm/rag/ingest.py`
- Test: `tests/integration/test_ingest_pipeline.py`

**Tests:**
- `test_ingest_two_docs_end_to_end`: 2-doc fixture → preprocessor → chunker → augmenter (mocked) → embedder (mocked) → in-memory Qdrant. Asserts `ingest_runs.status='completed'` and a positive `chunks_written`.
- `test_ingest_rolls_back_on_failure`: simulating a per-doc raise leaves `ingest_runs.status='failed'` and writes nothing to Qdrant.
- `test_ingest_is_resumable`: a second `make ingest` skips docs already indexed (lookup by `doc_id` payload).

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/ingest.py`** with `class IngestPipeline` and `def run_ingest(*, source, store, embedder, augmenter, db_path, clock, rag_config) -> IngestRunResult`. Per-doc tx: write `ingest_runs` row → preprocess → chunk → augment → embed → upsert (batch of N) → bump `chunks_written`. On failure: mark row `failed` with error.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/integration/test_ingest_pipeline.py -v
mypy firm/rag/ingest.py
```

---

## Task 12: Retrieval interface (dense + sparse + PIT filter)

**Files:**
- New: `firm/rag/retrieve.py`
- Test: `tests/unit/test_retrieve.py`, `tests/integration/test_retrieval_pit.py`

**Tests:**
- `test_hybrid_retrieve_returns_top_50_max`.
- `test_dense_and_sparse_results_merged_by_rrf`: reciprocal rank fusion produces a stable order.
- `test_pit_filter_excludes_future_chunks`: a chunk with `published_at=2025-01-01` does not appear when `as_of=2024-12-31` (CI invariant per spec §6.4).
- `test_retrieval_returns_chunks_with_doc_summary_prefix_attached`.

- [ ] **Step 1: Write tests using an in-memory Qdrant seeded with 20 fake chunks across 3 dates.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/retrieve.py`**:
  ```python
  class HybridRetriever:
      def __init__(self, *, store, embedder, sparse, collection, k_retrieve=50): ...
      def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]: ...
  ```
  Use RRF (k=60) to merge dense/sparse rankings. `RetrievedChunk` carries `chunk: Chunk`, `score: float`, `rank_dense: int|None`, `rank_sparse: int|None`.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_retrieve.py tests/integration/test_retrieval_pit.py -v
```

**Risks / notes:** `test_retrieval_pit.py` is a permanent CI invariant — keep it independent of the LLM cache so it stays cheap.

---

## Task 13: Reranker (bge-reranker-v2-m3)

**Files:**
- New: `firm/rag/rerank.py`
- Test: `tests/unit/test_rerank.py`

**Tests:**
- `test_rerank_returns_top_k_in_descending_score`.
- `test_rerank_filters_below_score_floor` (0.3 per spec §7.4).
- `test_rerank_is_deterministic_for_same_input`.

- [ ] **Step 1: Write tests with a stub reranker that scores by lexical overlap.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/rag/rerank.py`**:
  ```python
  class BgeReranker:
      def __init__(self, *, model_id: str, score_floor: float = 0.3) -> None: ...
      def rerank(self, query: str, candidates: list[RetrievedChunk], *, k: int) -> list[RetrievedChunk]: ...
  ```
  Lazy-loads `CrossEncoder("BAAI/bge-reranker-v2-m3")` via sentence-transformers. Batches scoring. Stores `rerank_score` on each returned `RetrievedChunk`.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_rerank.py -v
```

**Risks / notes:** Real model load is multi-second; mark `@pytest.mark.requires_models` and assert correctness on the stub.

---

## Task 14: Compose retrieve → rerank into a single Retriever facade

**Files:**
- Modified: `firm/rag/retrieve.py` (add `GroundedRetriever`)
- Test: extend `tests/unit/test_retrieve.py`

**Tests:**
- `test_grounded_retriever_returns_8_chunks_with_doc_summary_and_score`.
- `test_grounded_retriever_propagates_pit_filter`.
- `test_grounded_retriever_empty_when_floor_filters_all`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Add to `firm/rag/retrieve.py`**:
  ```python
  class GroundedRetriever:
      def __init__(self, *, hybrid: HybridRetriever, reranker: BgeReranker, k_final: int = 8): ...
      def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]:
          cands = self.hybrid.retrieve(query, as_of=as_of)
          return self.reranker.rerank(query, cands, k=self.k_final)
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_retrieve.py -v
```

---

## Task 15: LLM cache table-backed cache

**Files:**
- New: `firm/llm/__init__.py`, `firm/llm/cache.py`
- Test: `tests/unit/test_llm_cache.py`

**Tests:**
- `test_cache_miss_then_hit`: `get` returns None, `put` then `get` returns the stored payload.
- `test_cache_key_includes_model_id`: same prompt across different models → separate entries.
- `test_cache_invalidates_on_prompt_change`: trivial — different prompt_hash, different row.
- `test_cache_stores_token_counts`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/llm/cache.py`**:
  ```python
  class LlmCache:
      def __init__(self, db_path: Path, clock: Clock) -> None: ...
      def get(self, *, prompt_hash: str, model: str) -> CachedResponse | None: ...
      def put(self, *, prompt_hash: str, model: str, response_json: str,
              input_tokens: int, output_tokens: int) -> None: ...

  def hash_prompt(*, system: str, messages: list[dict], tools: list[dict] | None) -> str:
      """sha256 over canonical JSON of all model-relevant inputs."""
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_llm_cache.py -v
mypy firm/llm
```

**Risks / notes:** Canonicalize JSON with `sort_keys=True, separators=(",",":")` so unrelated dict ordering does not invalidate the cache.

---

## Task 16: Anthropic client wrapper (cached, mode-aware)

**Files:**
- New: `firm/llm/anthropic_client.py`
- Test: `tests/unit/test_anthropic_client.py`

**Tests:**
- `test_client_in_cached_mode_returns_cache_or_raises`: with `FIRM_LLM_MODE=cached` and an empty cache, calling `messages_create` raises `LlmCacheMissError`.
- `test_client_in_record_mode_calls_api_and_writes_cache`: with a fake transport, the request goes out and the response lands in the cache.
- `test_client_in_live_mode_bypasses_cache_writes`.
- `test_client_handles_citations_response_shape`: response includes content blocks with `citations` field — wrapper returns them intact.

- [ ] **Step 1: Write tests with a `FakeAnthropicTransport` that records calls and returns canned responses.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/llm/anthropic_client.py`**:
  ```python
  class LlmMode(StrEnum):
      LIVE = "live"; CACHED = "cached"; RECORD = "record"

  class AnthropicClient:
      def __init__(self, *, api_key: str | None, cache: LlmCache, mode: LlmMode,
                   clock: Clock, transport=None) -> None: ...
      def messages_create(self, *, model: str, system: str, messages: list[dict],
                          tools: list[dict] | None = None, max_tokens: int,
                          temperature: float = 0.0) -> dict: ...
  ```
  In `cached`, miss → `LlmCacheMissError`. In `live`, no cache write. In `record`, real call + cache write.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_anthropic_client.py -v
mypy firm/llm
```

**Risks / notes:** This is the only file that imports `anthropic`. Keep the dep optional-ish by lazy-importing inside methods.

---

## Task 17: Prompt templates

**Files:**
- New: `firm/llm/prompts.py`
- Test: covered indirectly by T18/T20/T25 — no dedicated test file.

- [ ] **Step 1: Implement `firm/llm/prompts.py`** with three constants:
  - `RESEARCH_SYSTEM`: instructs the model to answer ONLY using the cited `document` content blocks; banned from arithmetic; must call `fundamentals.get_ratio` / `risk.get_metric` for numeric facts; must produce a structured JSON of `Claim`s.
  - `SUFFICIENCY_SYSTEM` (Haiku): "Given a question and a set of cited claims, list every required claim and mark SUPPORTED/PARTIAL/UNSUPPORTED. Return JSON matching SufficiencyResult schema."
  - `PM_VOTER_SYSTEM`: parametrized by lens (quality | valuation | catalyst). Returns JSON `{vote, confidence, rationale, cited_claim_ids}`.
  - System prompts mark untrusted text via `<retrieved_content>` tags (spec §8.3) and forbid heeding instructions inside them.

- [ ] **Step 2: Add minimal unit test** `tests/unit/test_prompts.py` asserting each template contains its load-bearing phrases ("must not perform arithmetic", "<retrieved_content>", "SUPPORTED|PARTIAL|UNSUPPORTED", "quality lens" etc.).

- [ ] **Step 3: Commit.**

**Verify with:**
```
pytest tests/unit/test_prompts.py -v
```

**Risks / notes:** Prompt-injection hygiene is partial (spec §8.3 explicitly does not claim completeness); the real defense is structured outputs (T18) and least-privilege tools (T22/T23).

---

## Task 18: Cited claim extractor (Anthropic Citations API adapter)

**Files:**
- New: `firm/llm/citations.py`
- Test: `tests/unit/test_citations.py`

**Tests:**
- `test_extractor_emits_one_claim_per_citation`: given 2 retrieved chunks and a stub response containing 3 citation-anchored content blocks, the extractor returns 3 `Claim` objects with `source_chunk_id` populated.
- `test_extractor_rejects_uncited_claim`: a response with a text block lacking `citations` is dropped (and an `UNCITED_CLAIM` count is surfaced for the agent to act on).
- `test_extractor_carries_source_span`: each Claim has a non-None `source_span`.
- `test_extractor_passes_documents_with_citations_enabled`: verifies the request payload sets `citations: {enabled: true}` on each `document` block.

- [ ] **Step 1: Write tests with canned Anthropic response fixtures.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/llm/citations.py`**:
  ```python
  class CitedClaimExtractor(Protocol):
      def extract(self, *, query: str, chunks: list[Chunk], as_of: datetime) -> list[Claim]: ...

  class AnthropicCitationsExtractor:
      def __init__(self, *, client: AnthropicClient, model: str, max_tokens: int) -> None: ...
      def extract(self, *, query, chunks, as_of) -> list[Claim]:
          # Build documents payload: [{type:"document", source:{type:"text", media_type:"text/plain", data:chunk.text},
          #                            title: chunk.doc_id, citations:{enabled: True}}]
          # System prompt = RESEARCH_SYSTEM
          # Parse response: iterate content blocks; for each text block, attach the cited_text/document_index/source_span
          ...
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_citations.py -v
mypy firm/llm/citations.py
```

**Risks / notes:** The Anthropic Citations API response shape uses content blocks with a `citations` field per claim. Pin the wrapper against the exact JSON we see in `record` mode; freeze a fixture for replay.

---

## Task 19: Research agent rewrite (LLM-backed, grounded)

**Files:**
- Modified: `firm/agents/research.py`
- Test: `tests/unit/test_research_agent.py`, `tests/integration/test_research_end_to_end.py`

**Tests:**
- `test_research_emits_decision_with_citations_and_claims`: state output has `research_decision.citations` non-empty and the upstream `retrieved_chunks`, `claims` set on `WorkingState`.
- `test_research_refuses_when_retriever_returns_empty`: emits `Decision(action=REFUSE, failure_mode=INSUFFICIENT_EVIDENCE)`.
- `test_research_uses_pit_filter_with_replay_clock`: passes `clock.now()` as `as_of` into the retriever.
- `test_research_falsification_condition_non_empty`.
- Integration: `test_research_end_to_end` wires real `GroundedRetriever` (with in-memory Qdrant seeded by T11's fixture) + cached Anthropic client → produces a Decision with ≥1 citation.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Rewrite `firm/agents/research.py`**. The factory now takes `retriever: GroundedRetriever`, `extractor: CitedClaimExtractor`, `clock`. Heartbeat behavior:
  1. Choose a research question for the heartbeat. For Plan 2, the question is fixed per universe ticker rotation (deterministic): `"Summarize {ticker}'s latest reported financial trajectory and any near-term catalysts."`
  2. `chunks = retriever.retrieve(question, as_of=clock.now())`. If empty → emit a `REFUSE` Decision with `failure_mode=INSUFFICIENT_EVIDENCE` and return.
  3. `claims = extractor.extract(query=question, chunks=chunks, as_of=clock.now())`.
  4. Build a `Decision(action=BUY|HOLD, ...)` whose `citations` are derived from `claims`, `rationale` is the concatenated claim text, and `falsification_condition` references the strongest claim. Plan 2 keeps action selection rule-based on claim sentiment (LLM does not output action; PM does).
  5. Write `retrieved_chunks`, `claims` to state alongside `research_decision`.
  - Use `ulid_new()` and HMAC-signed `nonce` (real, via `sign_nonce`).

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_research_agent.py tests/integration/test_research_end_to_end.py -v
ruff check firm/agents/research.py
mypy firm/agents/research.py
```

**Risks / notes:** The Plan 1 stub signature was `make_research(*, clock, broker, universe)` — keep it backwards-compat by adding new kwargs with defaults (retriever, extractor), then update `cli.py` in T29.

---

## Task 20: Sufficiency judge

**Files:**
- New: `firm/grounding/judge.py`
- Test: `tests/unit/test_judge.py`

**Tests:**
- `test_judge_returns_all_supported_for_strong_claims`.
- `test_judge_returns_partial_when_some_claims_partial`.
- `test_judge_returns_unsupported_when_evidence_missing`.
- `test_judge_response_schema_validation_failure_raises`: if Haiku returns malformed JSON, raise `JudgeResponseError` (caller maps to `FailureMode.LLM_UNAVAILABLE`).

- [ ] **Step 1: Write tests with canned Haiku responses.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/grounding/judge.py`**:
  ```python
  class SufficiencyJudge:
      def __init__(self, *, client: AnthropicClient, model: str, max_tokens: int = 2048) -> None: ...
      def assess(self, *, question: str, claims: list[Claim]) -> SufficiencyResult:
          # tools=[json_schema(SufficiencyResult)], temperature=0, max_tokens=2048
          # Return parsed SufficiencyResult.
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_judge.py -v
mypy firm/grounding/judge.py
```

---

## Task 21: Wire sufficiency gate into research flow

**Files:**
- Modified: `firm/agents/research.py`, `firm/orchestrator/state.py`
- Test: extend `tests/unit/test_research_agent.py`, `tests/integration/test_research_end_to_end.py`

**Tests:**
- `test_research_proceeds_when_all_supported`.
- `test_research_escalates_on_any_partial`: emits `Decision(action=ESCALATE)` with `escalation_reason="sufficiency:partial"`.
- `test_research_refuses_on_any_unsupported`: emits `Decision(action=REFUSE, failure_mode=INSUFFICIENT_EVIDENCE)`.
- `test_state_carries_sufficiency_result_for_downstream`.

- [ ] **Step 1: Add fields to `WorkingState`**: `retrieved_chunks: list[dict]`, `claims: list[dict]`, `sufficiency_result: dict`, `pm_votes: list[dict]`. Persisted as dict (LangGraph checkpoint serialization-friendly).

- [ ] **Step 2: Update tests.**

- [ ] **Step 3: Run — verify fail.**

- [ ] **Step 4: Inject `judge: SufficiencyJudge` into `make_research`.** After `extractor.extract(...)`, call `judge.assess(...)`. Branch:
  - `"ok"` → proceed (BUY/HOLD decision)
  - `"partial"` → ESCALATE
  - `"insufficient"` → REFUSE with `INSUFFICIENT_EVIDENCE`
  Write `sufficiency_result` into state.

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.**

**Verify with:**
```
pytest tests/unit/test_research_agent.py tests/integration/test_research_end_to_end.py -v
```

---

## Task 22: fundamentals.get_ratio tool + pre-computed data

**Files:**
- New: `firm/tools/__init__.py`, `firm/tools/fundamentals.py`, `scripts/precompute_fundamentals.py`, `data/precomputed/fundamentals.parquet` (generated artifact, gitignored or committed depending on size)
- Test: `tests/unit/test_fundamentals_tool.py`

**Tests:**
- `test_get_ratio_returns_decimal_for_known_pair`: `get_ratio("AAPL", "pe_ratio", date(2024,3,13))` returns a `Decimal`.
- `test_get_ratio_raises_for_unknown_ticker`.
- `test_get_ratio_uses_as_of_to_select_latest_filing`: with two filings (2024-02-01, 2024-05-01) and `as_of=2024-03-13`, returns the 2024-02-01 value (PIT consistency).
- `test_get_ratio_signature_matches_mcp_tool_schema`: callable through a `Tool` dataclass with JSON-schema input validation.

- [ ] **Step 1: Author `scripts/precompute_fundamentals.py`** that walks the FinanceBench corpus, extracts a handful of ratios per (ticker, filing) — `pe_ratio`, `gross_margin`, `revenue_yoy_growth`, `debt_to_equity`, `current_ratio` — computed from filing tables. Writes a parquet keyed `(ticker, ratio_name, as_of)`. Determinism: the script reads `FIRM_RAG_CONFIG` and `tests/fixtures/financebench_two_docs.json` (for two tickers) when `FIRM_FUNDAMENTALS_FIXTURE=1` so unit tests can use a tiny parquet checked into `tests/fixtures/`.

- [ ] **Step 2: Write tests using the fixture parquet.**

- [ ] **Step 3: Run — verify fail.**

- [ ] **Step 4: Implement `firm/tools/fundamentals.py`**:
  ```python
  @dataclass(frozen=True)
  class ToolDef:
      name: str
      description: str
      input_schema: dict

  class FundamentalsTool:
      def __init__(self, parquet_path: Path) -> None: ...
      tool_def: ClassVar[ToolDef] = ToolDef(
          name="fundamentals.get_ratio",
          description="...",
          input_schema={...},
      )
      def get_ratio(self, *, ticker: str, ratio_name: str, as_of: date) -> Decimal: ...
  ```

- [ ] **Step 5: Run — verify pass.**

- [ ] **Step 6: Commit.**

**Verify with:**
```
pytest tests/unit/test_fundamentals_tool.py -v
mypy firm/tools/fundamentals.py
```

**Risks / notes:** Spec §7.3 says the LLM cannot output numbers. Wire this tool into the research prompt via Anthropic `tools=[...]` in T18 — T19 verifies the Sonnet response uses tool calls rather than typed numeric output.

---

## Task 23: risk.get_metric tool

**Files:**
- New: `firm/tools/risk_metrics.py`
- Test: `tests/unit/test_risk_metrics_tool.py`

**Tests:**
- `test_get_metric_returns_volatility_decimal_for_30d`.
- `test_get_metric_raises_for_unknown_metric_name`.
- `test_get_metric_uses_as_of_window`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `firm/tools/risk_metrics.py`** with `RiskMetricsTool.get_metric(*, ticker, metric, window) -> Decimal`. Plan 2 supports `metric ∈ {"volatility_30d", "beta_180d", "max_drawdown_90d"}` computed offline by `scripts/precompute_fundamentals.py` (or a sibling `precompute_risk.py`) and stored beside fundamentals.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_risk_metrics_tool.py -v
```

**Risks / notes:** No real price feed in Plan 2 — windows are computed on whatever proxy price series the precompute script chooses (e.g., a deterministic series). Plan 3's market-data MCP swaps in real values.

---

## Task 24: Wire tools into research extractor

**Files:**
- Modified: `firm/llm/citations.py`, `firm/agents/research.py`
- Test: extend `tests/unit/test_citations.py`, `tests/unit/test_research_agent.py`

**Tests:**
- `test_extractor_passes_tools_when_provided`: tool definitions appear in the Anthropic request payload.
- `test_extractor_attaches_tool_call_id_to_claim`: when Sonnet returns a `tool_use` block, the resulting `Claim.value` and `Claim.tool_call_id` are populated.
- `test_research_records_tool_call_ids_in_state`.

- [ ] **Step 1: Extend tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Modify `AnthropicCitationsExtractor`** to accept `tools: list[ToolDef] | None`. When the response contains `tool_use` blocks, the extractor executes the matching `Tool.run(**input)` and feeds back `tool_result` content blocks in a second `messages_create` call (one bounce, deterministic).

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_citations.py tests/unit/test_research_agent.py -v
```

**Risks / notes:** The "no LLM arithmetic" guarantee is enforced by **prompt + tool routing**, not by post-validation. The extractor must hand-execute tool calls and re-prompt; if the model emits a numeric Claim with neither `source_chunk_id` nor `tool_call_id`, drop it and increment an `UNCITED_CLAIM` counter (caller may convert to a failure mode).

---

## Task 25: PM voter (single-lens)

**Files:**
- New: helper inside `firm/agents/pm.py` (`PmVoter` class)
- Test: `tests/unit/test_pm_agent.py`

**Tests:**
- `test_voter_quality_lens_produces_vote_with_rationale_and_cited_claim_ids`.
- `test_voter_returns_buy_hold_sell_only`: enum-validated.
- `test_voter_rejects_claim_ids_not_in_input_set`: cited_claim_ids must be a subset of provided claims (server-side filter).
- `test_voter_uses_correct_prompt_per_lens`.

- [ ] **Step 1: Write tests with canned Sonnet responses for each lens.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `class PmVoter`** inside `firm/agents/pm.py`:
  ```python
  class PmLens(StrEnum):
      QUALITY = "quality"; VALUATION = "valuation"; CATALYST = "catalyst"

  class PmVote(BaseModel):
      lens: PmLens
      vote: ActionEnum   # BUY|HOLD|SELL
      confidence: float = Field(ge=0.0, le=1.0)
      rationale: str = Field(min_length=1)
      cited_claim_ids: list[str]

  class PmVoter:
      def __init__(self, *, client: AnthropicClient, model: str, max_tokens: int = 1024) -> None: ...
      def vote(self, *, lens: PmLens, question: str, claims: list[Claim],
               research_rationale: str) -> PmVote: ...
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_pm_agent.py -v
```

**Risks / notes:** Constrain Sonnet to JSON-only with tool-use schema; reject malformed outputs with `PmVoteSchemaError` → caller maps to `FailureMode.SCHEMA_VALIDATION_FAILED`.

---

## Task 26: PM vote aggregation (pure Python)

**Files:**
- New: helper `aggregate_votes` inside `firm/agents/pm.py`
- Test: `tests/unit/test_pm_vote_aggregation.py`

**Tests:**
Each of these is a single-case test; together they cover the spec's aggregation rules:
- `test_three_buy_yields_buy_high_confidence`.
- `test_two_buy_one_hold_yields_buy_with_mild_reservation`.
- `test_two_buy_one_sell_yields_escalate` (informative split).
- `test_one_buy_two_sell_yields_sell`.
- `test_three_hold_yields_hold`.
- `test_buy_hold_sell_mix_yields_escalate`.
- `test_aggregation_carries_per_lens_rationales_into_metadata`.

- [ ] **Step 1: Write tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement `aggregate_votes(votes: list[PmVote]) -> tuple[ActionEnum, float, str, FailureMode | None]`** per the locked rules. Deterministic; no LLM call.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_pm_vote_aggregation.py -v
mypy firm/agents/pm.py
```

---

## Task 27: PM agent rewrite (vote-of-3 + aggregation)

**Files:**
- Modified: `firm/agents/pm.py`, `firm/orchestrator/state.py`
- Test: extend `tests/unit/test_pm_agent.py`

**Tests:**
- `test_pm_runs_three_voters_in_parallel_and_aggregates`: assert three distinct prompts hit the AnthropicClient.
- `test_pm_emits_decision_with_aggregated_action_and_combined_rationale`.
- `test_pm_state_carries_pm_votes_list`.
- `test_pm_falls_through_when_research_action_is_refuse`: skips voting, passes Research's REFUSE down.
- `test_pm_handles_escalate_research_input`.

- [ ] **Step 1: Update tests.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Rewrite `make_pm` factory** to accept `voter: PmVoter`. Plan 2 calls voters sequentially (one cached Anthropic client; concurrency is Plan 3). For each lens in `(QUALITY, VALUATION, CATALYST)` → vote → collect. Then `action, confidence, combined_rationale, fmode = aggregate_votes(votes)`. Build a `Decision` chaining `research.id`, copying `research.citations` and `research.falsification_condition`. Write `pm_votes` dicts to state.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/unit/test_pm_agent.py -v
ruff check firm/agents/pm.py
mypy firm/agents/pm.py
```

**Risks / notes:** PM does not call retrieval or tools — it reasons only over `claims` produced by Research. This is the Chinese-wall constraint from spec §3.2.

---

## Task 28: CLI ingest subcommand + Makefile target

**Files:**
- Modified: `firm/cli.py`, `Makefile`, `.env.example`
- Test: `tests/integration/test_cli_ingest.py`

**Tests:**
- `test_cli_ingest_runs_against_fixture_corpus_and_writes_collection`: subprocess invocation with `FIRM_RAG_CONFIG` pointing at a fixture rag.yaml whose corpus is the 2-doc fixture, with `FIRM_LLM_MODE=cached` and pre-seeded cache for the doc summaries. Asserts `ingest_runs` table has a `completed` row and Qdrant collection contains chunks.

- [ ] **Step 1: Write the test.**

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Add `firm ingest` to `firm/cli.py`**:
  ```python
  @cli.command()
  @click.option("--config", default="config/rag.yaml")
  @click.option("--max-docs", default=None, type=int)
  def ingest(config: str, max_docs: int | None) -> None:
      """Ingest corpus into Qdrant. Idempotent — already-indexed docs are skipped."""
      ...
  ```
  Plus a `Makefile` target:
  ```
  ingest:
      FIRM_LLM_MODE=$${FIRM_LLM_MODE:-cached} python -m firm.cli ingest
  ```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
make ingest    # against fixture corpus by env override
pytest tests/integration/test_cli_ingest.py -v
```

---

## Task 29: Update `firm run` to construct the LLM stack

**Files:**
- Modified: `firm/cli.py`
- Test: extend `tests/integration/test_end_to_end_smoke.py` (Plan 1) — keep green; new test in T30 covers grounded path.

- [ ] **Step 1: Read existing `cli.py` `run` command**; replace `make_research(...)` and `make_pm()` construction with the Plan 2 dependency graph:
  - `llm_cache = LlmCache(db, clock)`
  - `client = AnthropicClient(api_key=os.environ.get("ANTHROPIC_API_KEY"), cache=llm_cache, mode=LlmMode(os.environ.get("FIRM_LLM_MODE","cached")), clock=clock)`
  - `embedder = NomicEmbedder(...)`; `sparse = BM25Sparse.load(...)`; `store = VectorStore(url=os.environ["QDRANT_URL"])`
  - `hybrid = HybridRetriever(store=store, embedder=embedder, sparse=sparse, collection=rag.qdrant.collection)`
  - `reranker = BgeReranker(model_id=rag.rerank.model)`
  - `retriever = GroundedRetriever(hybrid=hybrid, reranker=reranker, k_final=rag.retrieval.top_k_rerank)`
  - `tools = [FundamentalsTool(...), RiskMetricsTool(...)]`
  - `extractor = AnthropicCitationsExtractor(client=client, model=llm.research.model, max_tokens=llm.research.max_tokens)`
  - `judge = SufficiencyJudge(client=client, model=llm.judge.model)`
  - `voter = PmVoter(client=client, model=llm.pm.model)`
  - `research = make_research(clock=clock, retriever=retriever, extractor=extractor, judge=judge, tools=tools, universe=universe)`
  - `pm = make_pm(voter=voter)`

- [ ] **Step 2: Keep Plan 1 smoke test green** by making the LLM stack lazy: if the configured `FIRM_LLM_MODE=cached` and the cache misses on the first heartbeat, surface a clear error pointing to `make ingest` and `make record`. The Plan-1 `test_end_to_end_smoke.py` continues to run with the FakeBroker but is updated to either (a) skip when Qdrant is unreachable, or (b) seed Qdrant + LLM cache from fixtures in a pytest conftest.

- [ ] **Step 3: Run full unit suite — verify green.**

- [ ] **Step 4: Commit.**

**Verify with:**
```
pytest tests/unit -v
ruff check firm/cli.py
mypy firm/cli.py
```

---

## Task 30: End-to-end grounded demo integration test

**Files:**
- New: `tests/integration/test_end_to_end_grounded.py`, `tests/fixtures/llm_cache_seed.sql` (or a Python conftest), `tests/fixtures/qdrant_seed.json`
- Test: one big integration

**Tests:**
- `test_grounded_demo_produces_confirmed_trade_with_citations`:
  1. Seed in-memory Qdrant with the 2-doc fixture chunks (T11 helper).
  2. Seed `llm_cache` with canned responses for: 2 doc summaries (Haiku), 1 research extraction (Sonnet+Citations), 1 sufficiency assessment (Haiku, all SUPPORTED), 3 PM votes (Sonnet) — all 3 BUY on AAPL.
  3. `subprocess.run(["python","-m","firm.cli","run","--once"])` with `FIRM_LLM_MODE=cached`, `FIRM_BROKER=FAKE`, `FIRM_REPLAY_AT=2024-03-13T14:30:00+00:00`.
  4. Assert: `outbox` has 1 `confirmed` row; `decisions` table has ≥5 rows (research, 1 pm, 1 risk, optional hitl, execution-result decision row if recorded); `reports/2024-03-13/decisions.jsonl` contains at least one citation with a `chunk_id` traceable to a seeded chunk.

- [ ] **Step 1: Write fixtures and test.**

- [ ] **Step 2: Run — verify pass.**

- [ ] **Step 3: Commit.**

**Verify with:**
```
pytest tests/integration/test_end_to_end_grounded.py -v
```

**Risks / notes:** This is the Plan 2 acceptance test. It must remain runnable in CI without any network — every external call is cached or in-memory.

---

## Task 31: Flip `escalate_new_ticker` back to `true`

**Files:**
- Modified: `config/policy.yaml`
- Test: extend `tests/integration/test_end_to_end_grounded.py` to cover the new path

**Tests:**
- `test_grounded_demo_new_ticker_routes_through_hitl`: when the research thesis is on a ticker the firm holds zero shares of (the default for Plan 2 demo), Risk emits `ESCALATE`, the HITL gate parks the graph at the checkpoint, the test thread invokes `mark_approved(...)`, and on resume the order is confirmed.

- [ ] **Step 1: Write the test.**

- [ ] **Step 2: Run — verify fail (still false in policy).**

- [ ] **Step 3: Edit `config/policy.yaml`**: `escalate_new_ticker: true`. Remove the Plan-1 comment explaining the Plan-2 deferral.

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit.**

**Verify with:**
```
pytest tests/integration/test_end_to_end_grounded.py -v
```

---

## Task 32: README, runbook touch-ups, final validation pass

**Files:**
- Modified: `README.md`, `docs/runbook.md` (create if absent)
- (No new code.)

- [ ] **Step 1: Update `README.md`** Quickstart so the Plan 2 demo path is:
  ```bash
  docker compose up -d qdrant
  pip install -e ".[dev]"
  make ingest    # one-time, populates Qdrant from FinanceBench
  make demo      # heartbeat with grounded research and confirmed paper trade
  ```
  Mark Plan 2 row in the Status table as `[x]`.

- [ ] **Step 2: Append to `docs/runbook.md`** sections on: (a) what `make ingest` does and when to re-run, (b) `FIRM_LLM_MODE` semantics (live/cached/record), (c) the `llm_cache` table and how to clear it, (d) Qdrant volume backup.

- [ ] **Step 3: Run the full validation gate** below and fix anything red.

- [ ] **Step 4: Commit.**

**Verify with (full validation gate):**
```
ruff check firm tests
mypy firm
pytest -v
docker compose config -q
make ingest
make demo
```

Expected:
- `ruff` and `mypy` clean.
- Every unit and integration test passes (Plan 1's plus Plan 2's).
- `make ingest` exits 0 and creates a `firm_chunks` Qdrant collection with the FinanceBench corpus indexed.
- `make demo` exits 0; produces a JSONL report under `data/reports/<date>/` containing at least one citation; `outbox` table has one `confirmed` row.

---

## Validation Gates (Plan 2)

The plan is considered complete when **all** of the following hold:

1. `ruff check firm tests` — zero findings.
2. `mypy firm` (strict mode, as configured in Plan 1's `pyproject.toml`) — zero errors.
3. `pytest -v` — every test green; no skipped tests outside `requires_models` / `requires_hf_download` markers.
4. `tests/integration/test_retrieval_pit.py` — passes (PIT invariant from spec §6.4 holds).
5. `tests/integration/test_research_end_to_end.py` — passes.
6. `tests/integration/test_end_to_end_grounded.py` — passes; produced report contains at least one Citation with non-empty `chunk_id` and `source_span`.
7. `make ingest && make demo` — both exit 0; `make demo` produces a confirmed paper trade whose research thesis cites real FinanceBench chunks.
8. `escalate_new_ticker: true` in `config/policy.yaml`, and the corresponding HITL-routing test passes.
9. All 5 mandatory `FailureMode` paths that Plan 2 newly exercises (`INSUFFICIENT_EVIDENCE`, `UNCITED_CLAIM`, `LLM_UNAVAILABLE`, `SCHEMA_VALIDATION_FAILED`, `STALE_DATA`) have at least one passing test exercising them (extending Plan 1's coverage toward the spec §9.5 full-enum invariant).

---

## Risks, known limitations, and deferrals

**Risks accepted in Plan 2:**
- **No live Anthropic calls in CI.** All grounded tests run in `cached` mode against seeded fixtures. The `record` mode path is exercised manually before merge.
- **Pre-computed fundamentals/risk metrics.** Plan 2's MCP tools serve values computed offline at ingest time, not from a live market-data feed. The interface is stable; Plan 3's market-data MCP swaps the data source behind the same `get_ratio` / `get_metric` signatures.
- **Vote-of-3 PM runs sequentially.** Three Sonnet calls per heartbeat under the cached client — acceptable cost in cached/replay mode. Parallel execution is a Plan 3 concern (cost router + observability spans).
- **Reranker model load is heavy.** `bge-reranker-v2-m3` is multi-hundred-MB. CI marks these tests `requires_models`; the integration test seeds a stubbed reranker via a fixture.
- **Doc-summary cache is forever.** Changing the contextual augmentation prompt invalidates summaries silently. Mitigation: include the prompt text in the `prompt_hash`. A `make reindex` target is left for Plan 3/4.

**Explicitly deferred to Plan 3:**
- Slack signed approvals replacing the CLI ack stand-in (spec §8.4).
- OpenTelemetry tracing over LLM/tool/retrieval calls (spec §10.1).
- Cost router and per-call cost ledger (spec §10.2). Plan 2's `llm_cache` stores token counts so Plan 3 can read costs without re-instrumentation.
- Earnings transcripts and news `CorpusSource` adapters (spec §6.3).
- Markdown/XLSX daily reports and EOD reconciliation block (spec §5.7).
- Parallel PM voting + LLM fallback ladder (spec §10.2).
- Litestream actually running (still config-only).

**Explicitly deferred to Plan 4:**
- Eval harness using FinanceBench's 150 verified Q&A pairs (spec §9). Plan 2 deliberately does **not** consume them in the demo.
- Red-team corpus and prompt-injection invariants (spec §8.5).
- GitHub Actions CI workflows (spec §11.3).
- Terraform/AgentCore deployment artefacts (spec §11.1, §11.2).
- The full FailureMode enum coverage CI invariant — Plan 2 brings the count from Plan 1's baseline up by 5 modes; Plan 4 closes the remaining gaps and asserts 13/13 in CI.

**Documented RAG limitation (carries forward from spec §6.4):** Forward references inside otherwise-valid chunks ("as we'll discuss in Q4...") cannot be filtered automatically by the PIT payload filter. Acknowledged in `docs/runbook.md`.

---

**End of Plan 2.**
