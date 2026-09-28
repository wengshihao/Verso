"""Stage 2 — Entailment Judge (Sec. III-C). The single point in the
whole pipeline where "does visible information cover this?" is decided.

A bounded, narrow agent (Sec. III-A) — not a broad exploratory one:
S_vis/S_impl are already Stage 1's fully-curated, atomic requirement
lists, so there is no codebase left to explore. But a S_vis entry's
`cited_from` (the quoted evidence a `statement` was distilled from) can
carry more specific detail than the summary alone: a `statement` may
paraphrase as "must not overflow" while its own `cited_from` quotes an
issue report spelling out the exact fix formula, which a judge never
shown `cited_from` would miss. Showing it keeps this a
single mostly-self-contained call rather than turning it into a general
exploration agent: every `statement` is given directly (Stage 1 already
distilled these to atomic, concise claims — there is nothing to save by
hiding them), and only the potentially-long `cited_from` field is
fetched on demand, via one narrow tool, when a statement alone isn't
enough to decide.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from verso.llm.agent_loop import ToolSpec, run_agent
from verso.llm.base import LLMProvider
from verso.models import EntailmentResult, Requirement

SCHEMA_NAME = "entailment_judgment"
_MAX_TURNS = 10

ENTAILMENT_SYSTEM = """\
You are the Entailment Judge in Verso, an auditor \
for program-repair benchmarks.

You are given two independently-extracted lists of atomic requirements:
- S_vis: what a repair system could infer from visible information alone.
- S_impl: what the developer's actual patch commits to.

For EACH requirement r in S_impl, decide whether the full S_vis list \
above entails r — read all of it for every r, not just entries whose \
`subject` string happens to match. `subject` is a hint, not a filter: \
S_vis and S_impl were extracted independently by two agents that never \
saw each other's output, so they may describe the same method with \
differently-worded subjects — never dismiss an S_vis entry just because \
its `subject` text doesn't match r's verbatim. Judge strictly: S_vis \
entails r only if it actually pins down the SAME specific behavior r \
describes, not merely something related to the same method or a generic \
smoke-test expectation. Be conservative about `true` — when in doubt \
between `true` and `uncertain`, prefer `uncertain`.

Each S_vis entry's `statement` above is the extractor's own SUMMARY of \
what it found — not the full visible evidence itself. Its `cited_from` \
(fetch it with the `get_citation` tool) names or quotes the actual \
visible artifact the statement was distilled from, and can carry more \
specific detail than the summary does (e.g. an issue report might spell \
out an exact formula/algorithm that the summary only paraphrases as \
"avoids overflow"). Call `get_citation` for any S_vis entry whose \
statement looks plausibly relevant to an S_impl requirement you aren't \
yet confident about — don't decide `false`/`uncertain` off the summary \
alone when the underlying evidence might actually pin the behavior down; \
you don't need to call it for entries that are obviously irrelevant or \
already obviously decisive from the statement alone.

For each r, output:
- `entailed`: "true" (S_vis pins down this exact behavior), "false" \
(S_vis says nothing that would lead to this specific behavior), or \
"uncertain" (genuinely ambiguous).
- `checked_ids`: the ids of every S_vis requirement you actually compared \
r against (even if none support it — that still tells the next stage \
what was ruled out).
- `rationale`: one or two sentences.

Your submission MUST include a `results` entry for every S_impl \
requirement id given, in the same order, none omitted.
"""


class _EntailmentSubmission(BaseModel):
    results: list[EntailmentResult] = Field(default_factory=list)


def _render_s_vis_index(s_vis: list[Requirement]) -> str:
    if not s_vis:
        return "  (S_vis is empty)"
    return "\n".join(f"  - [{v.id}] ({v.subject}) {v.statement}" for v in s_vis)


def _render_s_impl(s_impl: list[Requirement]) -> str:
    return "\n".join(f"  - [{r.id}] ({r.subject}) {r.statement}" for r in s_impl)


def _get_citation_tool(s_vis: list[Requirement]) -> ToolSpec:
    by_id = {v.id: v for v in s_vis}

    def handler(args: dict) -> str:
        vid = args.get("id", "")
        v = by_id.get(vid)
        if v is None:
            return f"No such S_vis id: {vid!r}"
        if not v.cited_from:
            return f"[{vid}] has no cited_from recorded."
        return f"[{vid}] cited_from: {v.cited_from}"

    return ToolSpec(
        name="get_citation",
        description="Fetch the cited_from field (the actual visible evidence quote/pointer a S_vis "
        "statement was distilled from) for one S_vis requirement id — use this when a statement's "
        "summary alone isn't enough to tell whether it pins down the exact behavior you're checking.",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        handler=handler,
    )


def run_entailment_judgment(
    s_vis: list[Requirement], s_impl: list[Requirement], llm: LLMProvider
) -> list[EntailmentResult]:
    if not s_impl:
        return []
    user = f"""\
## S_vis (every entry — subject is a hint only, not a filter)
{_render_s_vis_index(s_vis)}

## S_impl (judge entailment for every one of these)
{_render_s_impl(s_impl)}

Judge entailment for every S_impl requirement listed above, as instructed.
"""
    tools = [_get_citation_tool(s_vis)]
    result, _calls = run_agent(
        llm,
        system=ENTAILMENT_SYSTEM,
        user=user,
        tools=tools,
        submit_name=SCHEMA_NAME,
        submit_model_cls=_EntailmentSubmission,
        max_turns=_MAX_TURNS,
    )
    return result.results
