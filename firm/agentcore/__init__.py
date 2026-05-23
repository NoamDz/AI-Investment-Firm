"""firm.agentcore — Bedrock AgentCore Runtime adapters (Plan 4 §T40–T41).

This package wraps the LangGraph agents in `firm.agents.*` so they can be
served via AWS Bedrock AgentCore Runtime. The adapters live behind the
optional `[agentcore]` install extra (wired in T41) so the core LangGraph
path keeps working without the AgentCore SDK installed.

Submodules import the SDK lazily and raise a helpful ImportError when the
extra is missing — importing this package itself does NOT pull the SDK in.
"""
