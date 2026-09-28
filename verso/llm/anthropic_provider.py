"""LLM provider backed by the Anthropic Messages API — or any endpoint
that speaks the same wire protocol (e.g. DeepSeek's Anthropic-compatible
endpoint at https://api.deepseek.com/anthropic).

Structured output is obtained by forcing a single tool call whose input
schema is the caller's JSON schema — this avoids parsing free-text JSON
out of a message body.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from verso.llm.base import AgentTurn, LLMOutputError, LLMProvider, ToolCall

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 8000,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, put it in .env "
                "(see .env.example), or pass api_key= explicitly."
            )
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            # A forced tool_choice is incompatible with extended thinking on at
            # least one Anthropic-compatible endpoint (DeepSeek) — Verso wants a
            # single structured result, not a thinking trace, so disable it.
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": schema_name,
                    "description": f"Return the {schema_name} result.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": schema_name},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == schema_name:
                return block.input
        raise LLMOutputError(
            f"Response did not include a '{schema_name}' tool call (tool_choice was forced to "
            f"it, but this endpoint doesn't always honor that). Raw response: {response}"
        )

    def run_agent_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            thinking={"type": "disabled"},
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
        )
        content = [block.model_dump() for block in response.content]
        tool_calls = [
            ToolCall(id=block["id"], name=block["name"], input=block["input"])
            for block in content
            if block.get("type") == "tool_use"
        ]
        return AgentTurn(content=content, tool_calls=tool_calls, stop_reason=response.stop_reason or "")
