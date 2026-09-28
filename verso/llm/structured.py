"""Get a validated pydantic model out of an LLMProvider, retrying with
corrective feedback on validation failure.

Needed because at least one Anthropic-compatible endpoint (DeepSeek) has
been observed to occasionally return an incomplete tool call — a
required field silently missing, or a nested object returned as a
JSON-encoded string — despite `stop_reason=tool_use` with plenty of
`max_tokens` headroom unused. That rules out truncation: it's a
schema-compliance reliability issue with the endpoint.

A blind retry (same prompt, hope for a different roll) measurably help
but a batch run still saw ~16% of tasks exhaust 3 blind retries. Instead,
each retry appends the previous invalid response and the exact pydantic
validation error to the prompt and asks the model to fix it — a
"self-correction on validation error" turn, not just another roll of the
dice.
"""

from __future__ import annotations

import json as json_module
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from verso.llm.base import LLMOutputError, LLMProvider
from verso.llm.coerce import coerce_json_strings

ModelT = TypeVar("ModelT", bound=BaseModel)


def complete_structured_model(
    llm: LLMProvider,
    *,
    system: str,
    user: str,
    schema_name: str,
    model_cls: type[ModelT],
    max_attempts: int = 4,
) -> ModelT:
    schema = model_cls.model_json_schema()
    last_error: ValidationError | LLMOutputError | None = None
    current_user = user

    for attempt in range(max_attempts):
        try:
            raw = llm.complete_structured(
                system=system, user=current_user, schema=schema, schema_name=schema_name
            )
        except LLMOutputError as exc:
            # The raw response wasn't even shaped like a call to the right
            # tool at all (e.g. this endpoint's forced tool_choice wasn't
            # honored) — same corrective-retry treatment as a validation
            # failure, just with no `raw` dict to show the model back.
            last_error = exc
            if attempt == max_attempts - 1:
                break
            current_user = (
                f"{user}\n\n"
                "--- CORRECTION NEEDED ---\n"
                f"Your previous response did not call the `{schema_name}` tool correctly: "
                f"{exc}\n\n"
                f"Call the `{schema_name}` tool — and only that tool, with exactly that "
                "name — with a complete, valid input."
            )
            continue
        try:
            return model_cls.model_validate(coerce_json_strings(raw))
        except ValidationError as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            try:
                raw_preview = json_module.dumps(raw, indent=2)[:2000]
            except TypeError:
                raw_preview = str(raw)[:2000]
            current_user = (
                f"{user}\n\n"
                "--- CORRECTION NEEDED ---\n"
                f"Your previous tool call for `{schema_name}` was invalid and failed "
                f"schema validation:\n{exc}\n\n"
                f"Your previous (invalid) input was:\n{raw_preview}\n\n"
                "Call the tool again with a COMPLETE, valid input. Double-check every "
                "required field listed in the errors above is present, non-null, and "
                "correctly typed — do not omit any of them."
            )

    assert last_error is not None
    raise last_error
