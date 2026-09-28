"""A call-counting LLMProvider wrapper, for measuring the real number of
LLM calls an audit makes. The paper reports measured call counts rather
than counts estimated from the pipeline's structure."""

from __future__ import annotations

from typing import Any

from verso.llm.base import AgentTurn, LLMProvider


class CountingProvider(LLMProvider):
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.complete_structured_calls = 0
        self.run_agent_turn_calls = 0

    @property
    def total_calls(self) -> int:
        return self.complete_structured_calls + self.run_agent_turn_calls

    def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        self.complete_structured_calls += 1
        return self._inner.complete_structured(**kwargs)

    def run_agent_turn(self, **kwargs: Any) -> AgentTurn:
        self.run_agent_turn_calls += 1
        return self._inner.run_agent_turn(**kwargs)
