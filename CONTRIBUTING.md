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
