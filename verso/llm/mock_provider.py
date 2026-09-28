"""Deterministic mock LLM provider for dry runs.

Responses are supplied by the caller, keyed by ``schema_name``, either as
a fixed dict, a list consumed in order (one call per response), or a
callable ``(system, user, schema) -> dict`` for dynamic behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verso.llm.base import AgentTurn, LLMProvider, ToolCall

Responder = dict[str, Any] | list[dict[str, Any]] | Callable[[str, str, dict[str, Any]], dict[str, Any]]


class MockProvider(LLMProvider):
    def __init__(self, responses: dict[str, Responder]) -> None:
        self._responses = dict(responses)
        self.calls: list[tuple[str, str, str]] = []  # (schema_name, system, user)

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        self.calls.append((schema_name, system, user))
        if schema_name not in self._responses:
            raise KeyError(f"MockProvider has no scripted response for schema_name={schema_name!r}")
        responder = self._responses[schema_name]
        if callable(responder):
            return responder(system, user, schema)
        if isinstance(responder, list):
            if not responder:
                raise IndexError(f"MockProvider ran out of scripted responses for {schema_name!r}")
            return responder.pop(0)
        return responder


def tool_call_turn(name: str, tool_input: dict[str, Any], call_id: str = "call") -> AgentTurn:
    """Build a scripted AgentTurn representing the model calling one tool
    (used to script MockAgentProvider)."""
    block = {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
    return AgentTurn(content=[block], tool_calls=[ToolCall(id=call_id, name=name, input=tool_input)])


class MockAgentProvider(LLMProvider):
    """Scripted provider for hermetic runs of verso.llm.agent_loop.run_agent.

    ``turns`` is a fixed sequence of AgentTurn objects consumed in order,
    one per run_agent_turn call — e.g. a tool call, then another tool
    call, then a submit-tool call built with `tool_call_turn(submit_name,
    payload)`. Raises IndexError if the agent asks for more turns than
    scripted, which fails loudly rather than hanging.
    """

    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []  # (system, messages) per call

    def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("MockAgentProvider only supports run_agent_turn().")

    def run_agent_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn:
        self.calls.append((system, [dict(m) for m in messages]))
        if not self._turns:
            raise IndexError("MockAgentProvider ran out of scripted turns")
        return self._turns.pop(0)
