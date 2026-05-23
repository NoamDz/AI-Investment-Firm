# Contributing

## Running tests

```powershell
python -m pytest
```

### Tests that require local model files

Tests marked `@pytest.mark.requires_models` exercise local model weights —
the BGE reranker and the Nomic embedder. They run by default (no marker
filter is applied in `pyproject.toml`). On a machine without the model
weights cached, they fail at import time inside `sentence_transformers`.

To skip the model-loading subset and run only the lighter tests:

```powershell
python -m pytest -m "not requires_models"
```

Three integration files and two unit files carry this marker today; grep
for it under `tests/` if you need the current list. Plan 4's CI matrix
will run the model subset as a separate job so model-availability flakes
don't block the unit job.

## Python version

Python 3.11.x is required (see `firm/__init__.py` for the guard). If `uv`
or pip caches the wrong interpreter, recreate the venv explicitly:

```powershell
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Re-recording LLM cassettes

The eval pipeline replays LLM responses from `.yaml` cassette files stored
under `tests/eval/cassettes/`. The cassette key is a SHA-256 hash of the
request's `(model, system, messages, tools)` fields. Any change to those
fields — a deliberate prompt edit, a model swap, or a tools update — causes a
hard **`CassetteMissError`** in CI eval logs. That is the intended behaviour
(prompt drift surfaces immediately rather than returning stale responses).

**When to re-record:** after any intentional change to a prompt, model name,
or tools spec that you want reflected in the eval results.

**How to re-record:**

```powershell
# 1. Set the required env vars
$env:FIRM_VCR_MODE   = "record"
$env:FIRM_LLM_MODE   = "live"
$env:ANTHROPIC_API_KEY = "<your key>"

# 2. Run the affected eval (makes live API calls and writes new .yaml files)
make eval

# 3. Commit the updated cassette files
git add tests/eval/cassettes/
git commit -m "chore(cassettes): re-record after prompt change"
```

Cassette files are plain YAML and safe to review in code review. Do not
commit cassettes that contain sensitive data (PII, real portfolio positions).
