"""Plan 4 §T41 — `[agentcore]` optional-extra contract tests.

Asserts the three properties promised by T41:

1. ``import firm.agentcore`` succeeds in ALL environments — the package's
   ``__init__.py`` is docstring-only and pulls no third-party deps, so it
   imports cleanly whether or not the optional ``[agentcore]`` extra has
   been installed.

2. ``import firm.agentcore.reporter_adapter`` raises ``ImportError`` with
   a friendly message pointing the operator at ``pip install -e .[agentcore]``
   when the optional ``bedrock_agentcore_sdk`` dependency is NOT available.
   This is the graceful-failure path — the WHOLE point of T41 is to make
   the missing extra discoverable, so the test must NOT skip when the SDK
   is absent. It must assert the ImportError fires.

3. The rest of ``firm/`` keeps importing cleanly with or without the
   extra. We sample a few representative modules (``firm.agents.reporter``,
   ``firm.cli``, ``firm.core.models``, ``firm.orchestrator.state``) to
   confirm the optional extra doesn't bleed into the core import graph.

Implementation note: when the SDK ``bedrock_agentcore_sdk`` IS installed
(e.g. on an operator's box that ran ``pip install -e .[agentcore]``),
the ImportError-fires test is inapplicable. We use ``importlib.util.find_spec``
to detect the SDK and branch: skip the negative-path assertion only in
that case, and add a positive-path assertion that the adapter module
imports cleanly. The negative path remains the default-install canary.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


# Detect whether the optional SDK is importable in this environment.
# This decision is made once at module import time; tests below branch on
# it. Default dev install (``pip install -e .``) → False → negative path
# runs. Operator install with extra → True → positive path runs.
_SDK_AVAILABLE = importlib.util.find_spec("bedrock_agentcore_sdk") is not None


def test_firm_agentcore_package_imports_without_sdk() -> None:
    """``import firm.agentcore`` must always succeed — package init is
    docstring-only and pulls no third-party deps. This is the canary that
    the package layout doesn't accidentally hoist SDK imports into the
    package ``__init__`` (a common refactor footgun).
    """
    # Drop any cached module so we exercise a fresh top-level import. A
    # prior test in this process may have populated ``sys.modules``.
    sys.modules.pop("firm.agentcore", None)

    import firm.agentcore  # noqa: F401 — import side-effect is the test

    # Sanity: the module loaded and exposes its docstring (no exec error).
    assert firm.agentcore.__doc__ is not None
    assert "agentcore" in firm.agentcore.__doc__.lower()


@pytest.mark.skipif(
    _SDK_AVAILABLE,
    reason=(
        "bedrock_agentcore_sdk IS installed in this env, so the graceful-"
        "ImportError path is unreachable. Run on a default `pip install -e .` "
        "environment (no [agentcore] extra) to exercise this assertion."
    ),
)
def test_reporter_adapter_raises_friendly_importerror_when_sdk_missing() -> None:
    """Without the ``[agentcore]`` extra, importing the adapter must raise
    ``ImportError`` whose message points the operator at the install
    command. This is the user-visible contract of the optional extra.

    The adapter's source (``firm/agentcore/reporter_adapter.py``) wraps
    its SDK import in a ``try/except ImportError`` and re-raises with the
    friendly message, so the exception type is ``ImportError`` (not
    ``ModuleNotFoundError``, which is a subclass — both match ``except
    ImportError``).
    """
    # Make sure we import fresh — a prior failed import leaves an entry
    # in ``sys.modules`` that pytest's import machinery may reuse.
    sys.modules.pop("firm.agentcore.reporter_adapter", None)

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("firm.agentcore.reporter_adapter")

    msg = str(excinfo.value)
    # The exact phrasing comes from T40 (`firm/agentcore/reporter_adapter.py`):
    #   "AgentCore SDK is not installed. Run `pip install -e .[agentcore]` ..."
    # We assert on the operator-facing install hint, which is the
    # load-bearing part of the message contract.
    assert "pip install -e .[agentcore]" in msg, (
        f"ImportError message must direct operators at the install command; got: {msg!r}"
    )
    assert "agentcore" in msg.lower(), (
        f"ImportError message must mention the agentcore extra; got: {msg!r}"
    )


@pytest.mark.skipif(
    not _SDK_AVAILABLE,
    reason=(
        "bedrock_agentcore_sdk is NOT installed; positive-path import is "
        "exercised by the negative-path test above. Install via "
        "`pip install -e .[agentcore]` to exercise this assertion."
    ),
)
def test_reporter_adapter_imports_cleanly_when_sdk_installed() -> None:
    """Mirror test for environments that DID install the extra — the
    adapter module must import without error and expose the ``reporter``
    AgentCore entrypoint. Skipped on the default dev install (the common
    case); active on operator boxes / CI jobs that opt into the extra.
    """
    sys.modules.pop("firm.agentcore.reporter_adapter", None)

    adapter = importlib.import_module("firm.agentcore.reporter_adapter")

    # The ``@agent`` decorator wraps a function named ``reporter`` —
    # asserting it's exposed pins the module's public surface.
    assert hasattr(adapter, "reporter"), (
        "adapter module must expose `reporter` (the @agent-decorated entrypoint)"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        # Representative slice of core ``firm/`` modules — picks the
        # Reporter (the agent that has an AgentCore adapter), the CLI
        # entrypoint, the core data models, and the orchestrator state.
        # If ANY of these started transitively importing the AgentCore
        # SDK, the optional-extra promise would be broken.
        "firm.agents.reporter",
        "firm.cli",
        "firm.core.models",
        "firm.orchestrator.state",
    ],
)
def test_core_firm_modules_import_without_agentcore_extra(module_name: str) -> None:
    """Each listed core module must import cleanly regardless of whether
    the ``[agentcore]`` extra is installed. We don't strip the SDK from
    ``sys.modules`` — even on operator boxes with the extra installed,
    these modules must not depend on it transitively. The strong form
    (which holds on default dev installs) is that they import even when
    the SDK is entirely absent from the environment.
    """
    # Force a fresh import so we don't fall back to a cached module that
    # was imported by a previous test that may have side-effects.
    sys.modules.pop(module_name, None)

    mod = importlib.import_module(module_name)
    assert mod is not None
