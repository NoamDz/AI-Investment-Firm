"""Firm package — Python 3.11.x required.

T27c: explicit guard so the failure mode surfaces in seconds rather than
deep inside ``sentence_transformers`` import on a 3.13 wheel that's
missing ``torch.SymInt`` (or similar). See README "Recreate the venv"
section for the explicit incantation.
"""
import sys

if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
    raise RuntimeError(
        f"firm requires Python 3.11.x (found {sys.version_info.major}."
        f"{sys.version_info.minor}). Recreate the venv with: "
        f'uv venv --python 3.11 && uv pip install -e ".[dev]"'
    )
