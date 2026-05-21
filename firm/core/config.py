"""Pydantic-validated YAML config loaders. See spec §3.7."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from firm.core.models import ProfileName


class PolicyLimits(BaseModel):
    max_position_pct: float = Field(gt=0, le=1.0)
    max_sector_pct: float = Field(gt=0, le=1.0)
    max_gross_exposure: float = Field(gt=0, le=2.0)
    max_trade_pct: float = Field(gt=0, le=1.0)
    max_trades_per_day: int = Field(gt=0)
    min_cash_pct: float = Field(ge=0, lt=1.0)
    max_daily_loss_pct: float = Field(gt=0, le=1.0)
    stale_quote_seconds: int = Field(gt=0)
    stale_filing_days: int = Field(gt=0)


class HitlConfig(BaseModel):
    trade_threshold_pct: float = Field(ge=0, le=1.0)
    escalate_new_ticker: bool
    slack_channel: str = Field(min_length=1)
    slack_approver_id: str = Field(min_length=1)


class PolicyConfig(BaseModel):
    limits: PolicyLimits
    hitl: HitlConfig


class UniverseConfig(BaseModel):
    as_of: date
    tickers: list[str] = Field(min_length=1)
    sector_map: dict[str, str]

    @model_validator(mode="after")
    def _check_tickers_consistency(self) -> "UniverseConfig":
        unmapped = [t for t in self.tickers if t not in self.sector_map]
        if unmapped:
            raise ValueError(f"unmapped tickers: {unmapped}")
        if len(self.tickers) != len(set(self.tickers)):
            counts: dict[str, int] = {}
            for t in self.tickers:
                counts[t] = counts.get(t, 0) + 1
            dupes = sorted(t for t, c in counts.items() if c > 1)
            raise ValueError(f"duplicate tickers: {dupes}")
        orphans = sorted(k for k in self.sector_map if k not in set(self.tickers))
        if orphans:
            raise ValueError(f"sector_map keys not in tickers: {orphans}")
        return self


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_universe(path: Path) -> UniverseConfig:
    return UniverseConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class FinanceBenchCorpusConfig(BaseModel):
    split: str
    max_docs: int | None = None
    eval_holdout_file: str | None = None


class CorpusConfig(BaseModel):
    financebench: FinanceBenchCorpusConfig


class ChunkConfig(BaseModel):
    target_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)


class EmbeddingConfig(BaseModel):
    dense_model: str = Field(min_length=1)
    dense_dim: int = Field(gt=0)
    sparse: str


class RetrievalConfig(BaseModel):
    top_k_retrieve: int = Field(gt=0)
    top_k_rerank: int = Field(gt=0)


class RerankConfig(BaseModel):
    model: str = Field(min_length=1)
    score_floor: float = Field(ge=0.0, le=1.0)


class ContextualConfig(BaseModel):
    summary_model: str = Field(min_length=1)


class QdrantConfig(BaseModel):
    collection: str = Field(min_length=1)
    url_env: str = Field(min_length=1)


class RagConfig(BaseModel):
    corpus: CorpusConfig
    chunk: ChunkConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    contextual: ContextualConfig
    qdrant: QdrantConfig


class LlmCallConfig(BaseModel):
    model: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)


class LlmConfig(BaseModel):
    research: LlmCallConfig
    judge: LlmCallConfig
    pm: LlmCallConfig


def load_rag_config(path: Path) -> RagConfig:
    return RagConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_llm_config(path: Path) -> LlmConfig:
    return LlmConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class RouterProfile(BaseModel):
    """One entry under ``profiles:`` in ``config/router.yaml`` (Plan 3 T06).

    Carries everything the CostRouter (T07) needs to actually fire a call:
    the Anthropic model id, the per-request ``max_tokens`` / ``temperature``
    knobs, and the per-Mtok USD rate card consumed by
    :func:`firm.llm.cost.compute_cost_usd`.
    """

    model_id: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    input_per_mtok_usd: float = Field(ge=0.0)
    output_per_mtok_usd: float = Field(ge=0.0)


class RouterConfig(BaseModel):
    """Typed ``config/router.yaml`` (Plan 3 T06).

    Three keyed profiles (``haiku`` / ``sonnet`` / ``opus``), the four
    feature weights consumed by :meth:`firm.core.models.RouterFeatures.score`,
    and an ordered ``fallback_chain`` of profile names used by the T07
    CostRouter when the primary call fails.
    """

    profiles: dict[ProfileName, RouterProfile]
    weights: dict[str, float]
    fallback_chain: list[ProfileName]

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> "RouterConfig":
        # 1. Exactly the three documented profile names must be present.
        expected = {"haiku", "sonnet", "opus"}
        present = set(self.profiles.keys())
        missing = expected - present
        if missing:
            raise ValueError(f"router profiles missing: {sorted(missing)}")
        extra = present - expected
        if extra:
            raise ValueError(f"router profiles has unknown keys: {sorted(extra)}")

        # 2. weights must have exactly the four RouterFeatures field names —
        #    a missing or stray key indicates a router.yaml typo and should
        #    fail at load time, not at first call.
        expected_w = {"risk_weight", "novelty", "complexity", "time_pressure"}
        present_w = set(self.weights.keys())
        if present_w != expected_w:
            missing_w = expected_w - present_w
            extra_w = present_w - expected_w
            raise ValueError(
                "router weights mismatch — "
                f"missing={sorted(missing_w)}, extra={sorted(extra_w)}"
            )

        # 3. fallback_chain references must resolve, and the chain itself
        #    must be non-empty (otherwise the router has nowhere to fall back).
        if not self.fallback_chain:
            raise ValueError("fallback_chain must be non-empty")
        unknown = [n for n in self.fallback_chain if n not in self.profiles]
        if unknown:
            raise ValueError(f"fallback_chain references unknown profiles: {unknown}")

        return self


def load_router_config(path: Path) -> RouterConfig:
    """Return the parsed, validated :class:`RouterConfig`."""
    return RouterConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
