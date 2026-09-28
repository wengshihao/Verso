"""Stage 1, Specification Extraction (Sec. III-B of the paper): two
independent subagents, each a bounded, tool-using agent over a disjoint
source. The S_vis Extractor never sees the developer patch and the S_impl
Extractor never sees S_vis, so neither output can be anchored or biased
by the other.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from verso.adapters.base import BenchmarkAdapter, Task
from verso.diffutils import parse_unified_diff
from verso.llm.agent_loop import ToolSpec, run_agent
from verso.llm.base import LLMProvider
from verso.models import NotebookEntry, Requirement

SVIS_SCHEMA_NAME = "s_vis_extraction"
SIMPL_SCHEMA_NAME = "s_impl_extraction"

# Generous on purpose (Sec. III-A doesn't optimize for call-count economy):
# a budget has to cover reading every visible file at least once, which a
# task with 9 of them already exhausts under a tight bound (list_files +
# 9 reads, before a single turn is left to submit). Turn pressure that
# tight measurably degrades extraction quality (thinner, sometimes
# literally empty S_vis) compared to giving the same agent
# room to actually finish exploring.
_MAX_TURNS = 20


class _RequirementListSubmission(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list,
        description="Short, neutral, factual/structural notes about the codebase worth passing "
        "to later stages (e.g. \"class Foo's constructor is package-private\") — never a "
        "judgment about what's required or not.",
    )

    @model_validator(mode="after")
    def _empty_requirements_must_be_explained(self) -> "_RequirementListSubmission":
        # An empty result is a legitimate outcome (e.g. an S_vis task with
        # a genuinely uninformative issue report and no disclosed tests),
        # but it must never be indistinguishable from a rushed or aborted
        # extraction — require the model to say why, in its own words,
        # so a reviewer (or a later pipeline stage) has something
        # concrete to check rather than a silent, unexplained blank.
        if not self.requirements and not self.notes:
            raise ValueError(
                "requirements is empty but notes is also empty — if you genuinely found nothing "
                "extractable, you must still explain why in notes (what you looked at and why none "
                "of it yielded a checkable requirement), not submit silently empty."
            )
        return self


SVIS_SYSTEM = """\
You are the S_vis Extractor in Verso, an auditor \
for program-repair benchmarks.

Your job: read ONLY what a repair system relying on visible information \
would see — an issue report, the buggy source code, and whichever tests \
(if any) this benchmark discloses — and extract the Visible Specification \
(S_vis): a list of atomic, checkable behavioral requirements a repair \
system could reasonably infer from this material alone.

You have tools to list and read files. List first, then read every \
visible file at least once before concluding — this pipeline does not \
optimize for a low call count, and a requirement grounded in a file you \
skipped is one the next stage can never recover.

You are NOT shown the developer's actual fix. Do not guess at it or \
speculate about what changed — describe only what the visible material \
itself actually establishes.

Each requirement is ONE atomic, checkable behavioral claim (e.g. \
"createNumber(String) throws NumberFormatException for a string that \
does not parse as any supported numeric literal") — general enough to \
cover every case a single fix would settle together, specific enough \
that one witness could confirm or refute it. The test for whether two \
observations belong in the SAME requirement or SEPARATE ones: would any \
plausible fix satisfy or break them together? If so (e.g. the same \
method's behavior at two values of the same flag, or two assertions in \
one test method) — state ONE requirement citing both, never near-duplicates \
that only vary a parameter. Split into separate requirements only when \
they are independently checkable — a fix could plausibly satisfy one \
without the other (distinct methods are usually independent this way, \
even ones that currently share the same bug). Never a vague paraphrase, \
and never a claim about what SHOULD happen if you're not sure the \
visible material actually says so. Tag each with:
- `subject`: the class/method/API surface it constrains.
- `cited_from`: QUOTE the actual decisive text VERBATIM, not just a \
pointer to where it lives. If the issue report or a test contains a \
concrete formula, code snippet, example input/output pair, or expected \
value, copy it into `cited_from` character-for-character — "see the \
overflow trace in the issue" is useless to a later stage that never \
reads the issue itself; the exact quoted formula/example is not. A bare \
file/method name with nothing quoted is only acceptable when the \
statement itself is already fully self-contained and no more specific \
text exists to quote.

Also record, in `notes`, any short, neutral, factual observations about \
the codebase that might help a later stage understand it faster (e.g. \
"tests live under src/test/java/...", "class X wraps class Y") — never \
a judgment about what's required. If you conclude there are genuinely \
NO extractable requirements, `notes` must explain why: what you read and \
why none of it yielded a checkable behavioral claim — never submit an \
empty result without that explanation.

When done, call the submission tool with your full list of requirements \
(it is fine and expected for this list to be short — most patches only \
have a few genuinely load-bearing visible requirements — but short \
because you looked and found few, never because you looked at less than \
everything visible).
"""

SIMPL_SYSTEM = """\
You are the S_impl Extractor in Verso, an auditor \
for program-repair benchmarks.

Your job: read ONLY the developer's actual patch (a diff — for each \
changed span, both the before-state (buggy) and after-state (gold), with \
its location) and extract the Implemented Specification (S_impl): the \
full list of atomic, checkable behavioral requirements this patch \
actually commits to — regardless of whether that behavior is documented \
or inferable from anything else. You are NOT shown the Visible \
Specification and must not try to guess what's "already visible" — your \
only job is to characterize what the patch itself does.

You have tools to read additional buggy/gold source files for context \
(e.g. what a called helper method does) if a hunk alone isn't enough to \
understand the change.

Each requirement describes OBSERVABLE BEHAVIOR — something a test could \
witness from outside the method (a return value, an exception, a side \
effect) — phrased exactly like a specification clause (e.g. \
"ClassUtils.toClass maps a null array element to a null Class entry \
rather than throwing"). Never state an internal implementation \
mechanism (a specific named constant, a specific loop bound, a specific \
intermediate formula, which internal branch runs) as its own \
requirement unless that mechanism is itself independently observable by \
some possible test — if two different internal mechanisms would produce \
identical observable behavior for every input, describe the resulting \
behavior once, not the mechanism. Tag each with:
- `subject`: the class/method/API surface it constrains.
- `hunk_ids`: which hunk(s) (from the catalogue below) implement it.

A single hunk can house more than one distinct requirement, and a single \
requirement can span more than one hunk — decompose by BEHAVIOR, not by \
hunk boundary. Two observations belong in the SAME requirement, not \
separate ones, if any plausible fix would satisfy or break them \
together; split into separate requirements only when they are \
independently checkable. Never restate the same information twice at \
different levels of aggregation (e.g. every individual guard clause of a \
method AND a separate "summary" requirement that just conjuncts them \
back together) — the second statement adds nothing a reviewer or later \
stage doesn't already have from the first.

Also record, in `notes`, any short, neutral, factual observations about \
the codebase (never a judgment about what's required). A patch that \
compiles and does something always commits to at least one behavioral \
requirement — if you conclude there is genuinely nothing extractable, \
`notes` must explain why rather than submitting an empty result silently.

When done, call the submission tool with your full list of requirements.
"""


def _render_notebook(notebook: list[NotebookEntry]) -> str:
    """Sec. III-D: every agent reads everything written by agents that
    ran before it. Rendered as a block in the initial user prompt — this
    is the read side of the Task Notebook; each agent's own findings are
    appended by the pipeline (verso/pipeline.py) after it returns."""
    if not notebook:
        return "(no notes from earlier stages yet)"
    return "\n".join(f"- [{e.author}] {e.content}" for e in notebook)


def _list_files_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        files = adapter.list_visible_files(task, args.get("subdir", ""))
        return "\n".join(files) if files else "(no files found under this path)"

    return ToolSpec(
        name="list_files",
        description="List repo-relative file paths visible to a repair system, optionally under a subdirectory.",
        input_schema={
            "type": "object",
            "properties": {"subdir": {"type": "string", "description": "Subdirectory prefix, or empty for all."}},
        },
        handler=handler,
    )


def _read_visible_file_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        try:
            return adapter.read_visible_file(task, args["path"])
        except KeyError:
            return f"No such visible file: {args.get('path')}"

    return ToolSpec(
        name="read_file",
        description="Read the full content of a visible file by its repo-relative path.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        handler=handler,
    )


def run_s_vis_extraction(
    task: Task, adapter: BenchmarkAdapter, llm: LLMProvider, notebook: list[NotebookEntry] | None = None
) -> tuple[list[Requirement], list[NotebookEntry]]:
    # S_vis Extractor runs first in the pipeline, so `notebook` is always
    # empty in practice — the parameter exists for interface symmetry
    # with the other three agents and so a future reordering doesn't
    # silently lose Sec. III-D's "reads everything written before it".
    issue = adapter.issue_text(task)
    user = f"""\
Task: {task.task_id} ({task.project} bug {task.bug_id})

## Issue / visible description
{issue}

## Notes from earlier stages (Task Notebook)
{_render_notebook(notebook or [])}

Explore the visible material with your tools and extract S_vis as instructed.
"""
    tools = [_list_files_tool(adapter, task), _read_visible_file_tool(adapter, task)]
    result, _calls = run_agent(
        llm,
        system=SVIS_SYSTEM,
        user=user,
        tools=tools,
        submit_name=SVIS_SCHEMA_NAME,
        submit_model_cls=_RequirementListSubmission,
        max_turns=_MAX_TURNS,
    )
    notes = [NotebookEntry(author="S_vis Extractor", content=n) for n in result.notes]
    return result.requirements, notes


def _read_buggy_file_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        try:
            return adapter.read_buggy_file(task, args["path"])
        except KeyError:
            return f"No such buggy file: {args.get('path')}"

    return ToolSpec(
        name="read_buggy_file",
        description="Read the full content of a file at the buggy (pre-fix) revision.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )


def _read_gold_file_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        try:
            return adapter.read_gold_file(task, args["path"])
        except KeyError:
            return f"No such gold file: {args.get('path')}"

    return ToolSpec(
        name="read_gold_file",
        description="Read the full content of a file at the gold (post-fix) revision.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )


def run_s_impl_extraction(
    task: Task, adapter: BenchmarkAdapter, llm: LLMProvider, notebook: list[NotebookEntry] | None = None
) -> tuple[list[Requirement], list[NotebookEntry]]:
    from verso.diffutils import format_hunk_catalogue_with_diff

    patch = adapter.developer_patch(task)
    hunks = parse_unified_diff(patch)
    catalogue = format_hunk_catalogue_with_diff(hunks)
    user = f"""\
Task: {task.task_id} ({task.project} bug {task.bug_id})

## Developer patch — hunk catalogue (before/after for each hunk)
{catalogue}

## Notes from earlier stages (Task Notebook)
{_render_notebook(notebook or [])}

Extract S_impl as instructed, citing hunk_ids from the catalogue above.
"""
    tools = [_read_buggy_file_tool(adapter, task), _read_gold_file_tool(adapter, task)]
    result, _calls = run_agent(
        llm,
        system=SIMPL_SYSTEM,
        user=user,
        tools=tools,
        submit_name=SIMPL_SCHEMA_NAME,
        submit_model_cls=_RequirementListSubmission,
        max_turns=_MAX_TURNS,
    )
    notes = [NotebookEntry(author="S_impl Extractor", content=n) for n in result.notes]
    return result.requirements, notes
