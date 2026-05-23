# `tests/` — test layout

```
tests/
├── unit/          # fast, isolated; no I/O, no models
├── integration/   # exercise multiple modules; some require model files or Qdrant
├── eval/          # eval harness + cassettes for replay-mode regression tests
├── red_team/      # 51-case adversarial corpus + 10 invariant test modules
├── fixtures/      # FailureMode triggering fixtures + shared test data
└── conftest.py    # session-wide pytest hooks (VCR mode pin, clock injection, …)
```

## Running the suite

```powershell
pytest                                   # full suite (~4-6 min with cassettes)
pytest tests/unit                        # fast subset only
pytest -m "not requires_models"          # skip BGE reranker + Nomic embedder tests
pytest -m "not live"                     # skip live-API tests (default in CI)
```

## Live (paid) e2e

```powershell
$env:FIRM_E2E_LIVE = "1"
$env:ANTHROPIC_API_KEY = "sk-..."
pytest tests/integration/test_end_to_end_grounded_live.py -v
# ~90s, ~$0.10 per run, exercises real Anthropic round-trip + Citations API + tool_use
```

The cached counterpart (`test_end_to_end_grounded.py`) runs the same flow from
seeded LlmCache and is CI-runnable with no API key.

## Markers

| Marker | Effect |
|--------|--------|
| `requires_models` | Loads BGE reranker / Nomic embedder weights from `~/.cache/huggingface/`. CI runs these as a separate job so missing weights don't block the unit suite. |
| `live` | Hits the real Anthropic API. Gated by `FIRM_E2E_LIVE=1` env var (skip by default). |

See [`../docs/CONTRIBUTING.md`](../docs/CONTRIBUTING.md) for the re-recording
flow when prompts or tool specs change.
