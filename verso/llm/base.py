"""LLM provider interface.

Verso needs bounded,
multi-turn, tool-using subagents that explore before answering —
`run_agent_turn` plus verso.llm.agent_loop.run_agent provide that, layered
on the same provider interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMOutputError(RuntimeError):
    """Raised when a provider's raw response can't be turned into the
    requested structured output at all — e.g. no matching tool call in
    the response. Distinct from a schema-validation failure (the
    response has the *wrong shape entirely*, not an invalid one) but
    still meant to be caught and retried the same way: observed in
    practice on DeepSeek's endpoint, where the forced tool_choice isn't
    always honored (e.g. returning a tool call named after one of the
    schema's own field names instead of the requested tool name)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class AgentTurn:
    """One turn of a multi-turn tool-using conversation.

    ``content`` is the raw assistant-turn content blocks (provider wire
    format) — callers append it verbatim to the message history so the
    next turn has full context, without verso.llm.agent_loop needing to
    know provider-specific block shapes.
    """

    content: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    stop_reason: str = ""


class LLMProvider(ABC):
    @abstractmethod
    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Run one LLM call and return output matching the given JSON schema.

        ``schema`` is a JSON Schema object (e.g. from
        ``pydantic.BaseModel.model_json_schema()``). Implementations must
        return a plain dict that validates against it.
        """
        raise NotImplementedError

    def run_agent_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """One turn of a bounded, multi-turn, tool-using agent.

        ``messages`` is the full conversation so far in provider wire
        format (earlier turns' ``AgentTurn.content`` plus the tool-result
        messages the caller appended); ``tools`` is a list
        of ``{name, description, input_schema}``. The model chooses
        whether to call a tool or not (unlike ``complete_structured``,
        tool use here is never forced to one specific tool) — see
        verso.llm.agent_loop.run_agent for the calling convention.

        Optional: only providers that support multi-turn tool use need
        to implement this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support run_agent_turn().")
