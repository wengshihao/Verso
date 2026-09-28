"""Generic, bounded, multi-turn tool-using agent runner (Sec. III-B and
Sec. III-D of the paper). Every agent (S_vis/S_impl Extractor, Oracle
Search Agent, Witness Synthesis Agent) is `run_agent()` with a different
system prompt, tool set, and submission schema; the loop itself never
changes.

Design choices:

- The final answer is submitted via a dedicated tool (schema-validated,
  same as verso.llm.structured's single-shot calls), never parsed out of
  free text.
- A validation failure on submission is fed back as a tool_result error
  and the agent gets another turn to fix it — the same corrective-retry
  discipline verso.llm.structured already uses for single-shot calls,
  extended to a multi-turn setting.
- `max_turns` is a hard bound (Sec. III-A: "every subagent's exploration
  must be bounded") — there is no unbounded reflection loop.
- Every executed tool call is recorded and handed back to the caller
  alongside the final answer, and an optional `validate_submission`
  callback can mechanically check the submission against that record
  before it is accepted — rejecting it (with the same corrective-retry
  treatment as a schema failure) if the model's claim isn't actually
  backed by a tool call it made. Without this, a submission is just an
  unverified self-report — exactly the "one call proposes and judges"
  pattern that Sec. III-A's propose/judge separation rules out; callers
  whose submission asserts something checkable (e.g. "I
  ran this and it passed") should always supply a validator rather than
  trust the field at face value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from verso.llm.base import LLMProvider
from verso.llm.coerce import coerce_json_strings

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ExecutedToolCall:
    """A record of one tool call actually made during an agent run —
    the mechanical ground truth a `validate_submission` callback checks
    a submission against, independent of what the model claims."""

    name: str
    input: dict[str, Any]
    output: str


class AgentBudgetExceededError(RuntimeError):
    """The agent used its entire turn budget without ever calling the
    submit tool with a valid payload. Sec. III-A: every subagent's
    exploration must be bounded, so hitting this bound is an anticipated
    outcome, not a bug — callers should handle it, not let it crash the
    task. Carries `executed_calls` (whatever tool calls happened before
    the budget ran out) so a caller can still make a mechanically-grounded
    decision (e.g. Sec. III-D's "did any candidate ever compile") rather
    than discarding that work along with the exception."""

    def __init__(self, message: str, executed_calls: list[ExecutedToolCall] | None = None) -> None:
        super().__init__(message)
        self.executed_calls = executed_calls or []


def run_agent(
    provider: LLMProvider,
    *,
    system: str,
    user: str,
    tools: list[ToolSpec],
    submit_name: str,
    submit_model_cls: type[ModelT],
    max_turns: int = 8,
    validate_submission: Callable[[ModelT, list[ExecutedToolCall]], str | None] | None = None,
) -> tuple[ModelT, list[ExecutedToolCall]]:
    """Returns (submission, executed_tool_calls) — the latter is the
    mechanical record `validate_submission` (and, if the caller wants to
    look closer, code downstream of this call) checks claims against;
    never trust a submission field that a validator could have checked."""
    submit_tool = {
        "name": submit_name,
        "description": (
            f"Submit your final {submit_name} result. Call this only when you are done "
            "investigating and ready to give your answer — not before."
        ),
        "input_schema": submit_model_cls.model_json_schema(),
    }
    tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
    tool_defs.append(submit_tool)
    handlers = {t.name: t.handler for t in tools}

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    executed_calls: list[ExecutedToolCall] = []
    last_error: ValidationError | None = None

    for _ in range(max_turns):
        turn = provider.run_agent_turn(system=system, messages=messages, tools=tool_defs)
        messages.append({"role": "assistant", "content": turn.content})

        submit_call = next((c for c in turn.tool_calls if c.name == submit_name), None)
        if submit_call is not None:
            try:
                result = submit_model_cls.model_validate(coerce_json_strings(submit_call.input))
            except ValidationError as exc:
                last_error = exc
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": submit_call.id,
                                "content": f"Your submission was invalid: {exc}\nFix it and call "
                                f"`{submit_name}` again with a complete, valid input.",
                                "is_error": True,
                            }
                        ],
                    }
                )
                continue

            rejection = validate_submission(result, executed_calls) if validate_submission is not None else None
            if rejection is not None:
                last_error = None
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": submit_call.id,
                                "content": rejection,
                                "is_error": True,
                            }
                        ],
                    }
                )
                continue
            return result, executed_calls

        if not turn.tool_calls:
            # The model responded with plain text and no tool call — nudge
            # it back toward the loop rather than silently stalling.
            messages.append(
                {
                    "role": "user",
                    "content": f"Call one of your tools to keep investigating, or call "
                    f"`{submit_name}` with your final answer.",
                }
            )
            continue

        tool_results = []
        for call in turn.tool_calls:
            handler = handlers.get(call.name)
            output = handler(call.input) if handler is not None else f"Unknown tool: {call.name}"
            executed_calls.append(ExecutedToolCall(name=call.name, input=call.input, output=output))
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        messages.append({"role": "user", "content": tool_results})

    if last_error is not None:
        raise AgentBudgetExceededError(
            f"Agent exceeded max_turns={max_turns}; its last `{submit_name}` submission was still "
            f"invalid: {last_error}",
            executed_calls=executed_calls,
        )
    raise AgentBudgetExceededError(
        f"Agent exceeded max_turns={max_turns} without calling `{submit_name}`.", executed_calls=executed_calls
    )
