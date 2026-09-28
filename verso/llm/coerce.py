"""Defensive coercion for structured LLM output.

Some Anthropic-compatible endpoints (observed with DeepSeek's) occasionally
return a nested object/array field as a JSON-encoded string instead of an
actual nested value inside an otherwise-valid tool call. This recursively
un-stringifies anything that parses as JSON, so pydantic validation sees
real nested structures either way.
"""

from __future__ import annotations

import json
from typing import Any


def coerce_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                return coerce_json_strings(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                return value
        return value
    if isinstance(value, dict):
        return {k: coerce_json_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_json_strings(v) for v in value]
    return value
