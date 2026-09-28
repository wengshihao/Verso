"""Stage 3 — Witness Finding (Sec. III-D): Oracle Search Agent first,
Witness Synthesis Agent as a fallback. Both submit the same shape of
answer (a candidate Witness), and both must back it with real execution
against buggy and gold — a submission claiming found=true is mechanically
rejected (not just discouraged by the prompt) unless the executed tool
call record actually shows a run_test(against="gold") that PASSED and a
run_test(against="buggy") that FAILED for that exact test_source. Trusting
the model's self-reported gold_result/buggy_result fields on their own
would be exactly the "one call proposes and judges" pattern that
Sec. III-A's propose/judge separation rules out.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from verso.adapters.base import BenchmarkAdapter, Task
from verso.agents.extraction import (
    _list_files_tool,
    _read_buggy_file_tool,
    _read_gold_file_tool,
    _read_visible_file_tool,
    _render_notebook,
)
from verso.diffutils import DiffHunk, find_hunk
from verso.llm.agent_loop import AgentBudgetExceededError, ExecutedToolCall, ToolSpec, run_agent
from verso.llm.base import LLMProvider
from verso.models import NotebookEntry, Requirement, Witness, WitnessKind

ORACLE_SEARCH_SCHEMA_NAME = "oracle_witness"
WITNESS_SYNTHESIS_SCHEMA_NAME = "synthesized_witness"

# Generous on purpose, same reasoning as extraction's budget (Sec. III-A
# doesn't optimize for call-count economy): `list_oracle_test_files`
# returns Defects4J's full `tests.relevant` set for the modified classes,
# which for a widely-used utility class can run past a dozen files —
# list + read each once + submit already exceeds a tight budget before
# the agent has even finished looking, which risks the same
# turn-pressure-driven quality loss already found and fixed for
# extraction (a rushed or incomplete search, not a genuine "nothing
# found").
_MAX_TURNS = 20
# Was 12 — sized back when this agent's only tools were run_test/
# read_buggy_file/read_gold_file. It later gained list_files/
# read_visible_file (to check candidates against visible material
# on demand instead of a prompt dump), which now competes with the
# core propose -> run_test(gold) -> run_test(buggy) -> revise loop
# (Sec. III-D wants that loop genuinely iterative, not cut short) for
# the same fixed budget — raised for the same reason as _MAX_TURNS above.
_MAX_SYNTHESIS_TURNS = 20


def _assertion_lines(test_source: str) -> list[str]:
    return [line.strip() for line in test_source.splitlines() if "assert" in line.lower() and line.strip()]


def _is_witness_genuinely_hidden(adapter: BenchmarkAdapter, task: Task, test_source: str) -> bool:
    """False if every assertion line in `test_source` already appears
    (content-based, not path-based) somewhere in visible material,
    meaning this "witness" doesn't demonstrate the oracle enforcing
    anything beyond visible information (Sec. II-B requires a Witness
    to confirm the oracle enforces something visible information does
    NOT establish; a witness built from visible content can't demonstrate
    that, no matter what the model's own explanation claims — note that a benchmark's "hidden" test suite can
    contain a file whose content is identical to one already shown as
    visible, and a search agent has no built-in reason to notice that on
    its own).

    Delegates the actual per-line search to
    `adapter.visible_content_contains` rather than concatenating every
    visible file into one string here — fine for a benchmark whose
    visible surface is a handful of curated files (Defects4J), but a
    benchmark whose visible surface is an entire real checkout
    (SWE-bench: the whole repo at base_commit) needs this to be a search
    the adapter can do efficiently (e.g. `git grep`), not a multi-megabyte
    string built and scanned in Python/an LLM prompt on every check."""
    lines = _assertion_lines(test_source)
    if not lines:
        return True  # nothing assertion-shaped to check; don't block on this alone
    return not all(adapter.visible_content_contains(task, line) for line in lines)


class WitnessSubmission(BaseModel):
    found: bool = Field(..., description="True only if you found/built a genuine, execution-confirmed witness.")
    test_name: str = ""
    test_source: str = ""
    gold_result: str = ""
    buggy_result: str = ""
    explanation: str = Field(
        ..., description="Why this test genuinely corresponds to the target requirement (not just "
        "co-located code touching the same method), or why nothing qualified."
    )


ORACLE_SEARCH_SYSTEM = """\
You are the Oracle Search Agent in Verso, an auditor for \
program-repair benchmarks.

You are given ONE target requirement `r` — a specific behavioral claim \
the developer's patch commits to, already established as NOT entailed by \
visible information. Your job: determine whether the benchmark's real, \
pre-existing oracle (its full test suite) already contains a test that \
serves as a Witness for r — i.e. a test that passes against the gold \
(fixed) program and fails against the buggy (pre-fix) program, and whose \
assertion genuinely tests the SAME behavior r describes (not some other \
requirement that happens to live in the same method or file).

You are told which tests currently fail against the buggy program (a \
test failing there is a NECESSARY condition for it to be a witness for \
ANY requirement in this patch — but a hunk or method can house more than \
one distinct behavior, so a failing test is not automatically about YOUR \
specific `r`; you must actually read it and judge that for yourself).

You have tools to read the source of oracle test files, list all of \
them (in case a relevant test isn't among the immediately-failing ones, \
e.g. because it currently errors out before reaching the relevant \
assertion), and to mechanically run a candidate test's source against \
the buggy or gold program to confirm pass/fail for yourself — do not \
trust your own reading of a test's likely outcome; always confirm by \
running it before submitting `found=true`. This pipeline does not \
optimize for a low call count: check every file the oracle actually \
contains before concluding `found=false`, not just the obvious or \
failing ones — a real match you didn't read is indistinguishable from \
one that doesn't exist.

**Critical**: `read_test_file` labels every file's content [VISIBLE] or \
[HIDDEN] (content-based — a file the oracle happens to also contain can \
be byte-for-byte the same material a repair system already saw). A \
Witness must demonstrate the oracle enforcing something visible \
information does NOT establish — an assertion drawn from [VISIBLE] \
content can NEVER serve as one, no matter how well it seems to match `r`, \
because it was never hidden in the first place. Only ever build your \
`test_source` from [HIDDEN] assertions. If everything relevant you find \
is [VISIBLE], that is not a match — submit found=false.

If you find a genuine match: submit found=true with `test_source` set to \
a COMPLETE, standalone JUnit 4 test method — including the `@Test` \
annotation and full method signature, e.g. `@Test public void t() { ... }` \
— derived from the real test's assertion (add an explicit `@Test` \
annotation even if the original file uses a different convention; it \
will be compiled and run standalone, in isolation from the original \
file). Also give both confirmed run results. **Your submission's \
`test_source` must be byte-for-byte the exact string you last passed to \
`run_test` — you must have actually called `run_test` with this exact \
source against BOTH "gold" (and gotten PASSED) and "buggy" (and gotten \
FAILED) before submitting found=true; a submission that isn't backed by \
matching run_test calls will be rejected and you'll be asked to actually \
confirm it.** If nothing in the oracle genuinely tests r (even if you \
looked at every failing test): submit found=false and explain what you \
checked and why none qualified — this is a normal, expected outcome, not \
a failure on your part.
"""

WITNESS_SYNTHESIS_SYSTEM = """\
You are the Witness Synthesis Agent in Verso, an auditor for \
program-repair benchmarks, invoked only because the Oracle \
Search Agent could not find a suitable existing test.

You are given ONE target requirement `r`, the gold (fixed) program's \
actual code implementing it, and the buggy (pre-fix) program's code for \
the same location. Your job: write ONE complete, standalone JUnit 4 test method — including \
the `@Test` annotation and full method signature, e.g. `@Test public \
void t() throws Exception { ... }` — that captures r as a concrete, \
executable assertion, then use your `run_test` tool to run it against \
both programs and see the real result — iterate based on that real \
feedback (a compile error, both passing, both failing, or the wrong \
direction) rather than guessing blindly. A genuine witness PASSES on \
gold and FAILS on buggy. It will be compiled standalone in its own \
class, so it must not depend on any fields/setup from an existing test \
class — construct everything it needs within the method body itself.

**Before finalizing any candidate**: use `list_visible_files`/`read_visible_file` \
to check whatever part of the visible material is actually relevant to \
your candidate input — not the whole task's visible surface, just enough \
to confirm this specific input/behavior isn't already pinned down there. \
The Oracle Search Agent already ruled out reusing an existing test \
verbatim, but nothing stops you from independently reinventing an \
already-visible assertion in different words (different variable names, \
`assertTrue(x instanceof Long)` instead of `assertEquals(Long.valueOf(...), x)`, \
decimal instead of hex literals, etc.) — that would still be worthless as \
a Witness, since the oracle enforcing it was never a hidden fact to begin \
with. If what's visible already tells you what your chosen input \
produces, pick a genuinely different input/scenario that it does NOT \
already pin down instead.

**Your submission's `test_source` must be byte-for-byte the exact string \
you last passed to `run_test` — you must have actually called `run_test` \
with this exact source against BOTH "gold" (and gotten PASSED) and \
"buggy" (and gotten FAILED) before submitting found=true; a submission \
that isn't backed by matching run_test calls will be rejected and you'll \
be asked to actually confirm it.** If you exhaust your reasonable \
attempts without a discriminating test, submit found=false and explain \
what you tried.
"""


def _list_test_files_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        files = adapter.list_oracle_test_files(task)
        return "\n".join(files) if files else "(no oracle test files found)"

    return ToolSpec(
        name="list_test_files",
        description="List every test file that is part of the real oracle (visible and hidden — "
        "read_test_file labels each one's content as [VISIBLE] or [HIDDEN]).",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


def _read_test_file_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        try:
            content = adapter.read_oracle_test_source(task, args["path"])
        except KeyError:
            return f"No such oracle test file: {args.get('path')}"
        label = "VISIBLE" if not _is_witness_genuinely_hidden(adapter, task, content) else "HIDDEN"
        return f"[{label}] (a witness may only rest on [HIDDEN] content — see system prompt)\n{content}"

    return ToolSpec(
        name="read_test_file",
        description="Read the full source of an oracle test file by its repo-relative path — labeled "
        "[VISIBLE]/[HIDDEN] so you know whether it can serve as a Witness.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )


def _run_test_tool(adapter: BenchmarkAdapter, task: Task) -> ToolSpec:
    def handler(args: dict) -> str:
        against = args.get("against", "")
        if against not in ("buggy", "gold"):
            return "`against` must be exactly 'buggy' or 'gold'."
        result = adapter.run_candidate_test(task, args.get("test_source", ""), against)
        status = "PASSED" if result.passed else "FAILED"
        return f"Running against {against}: {status}" + (f"\nDetail: {result.detail}" if result.detail else "")

    return ToolSpec(
        name="run_test",
        description="Mechanically compile and run a test method's source against the buggy or gold "
        "program. Always use this to confirm a candidate before submitting — never guess.",
        input_schema={
            "type": "object",
            "properties": {
                "test_source": {"type": "string", "description": "The complete test method source."},
                "against": {"type": "string", "enum": ["buggy", "gold"]},
            },
            "required": ["test_source", "against"],
        },
        handler=handler,
    )


def _validate_witness_submission(
    sub: WitnessSubmission, calls: list[ExecutedToolCall], *, adapter: BenchmarkAdapter, task: Task
) -> str | None:
    """Sec. III-A: never accept a self-reported claim about hidden/
    execution information without an independent, mechanical check. A
    found=true submission is rejected (triggering another turn, same
    treatment as a schema failure) unless the exact test_source was
    actually confirmed via run_test — the model's own gold_result/
    buggy_result text fields are never taken on their own say-so.

    Also rejects a submission whose assertions are already fully present
    in visible material (content-based check, not path-based — see
    _is_witness_genuinely_hidden): a witness built from what was visible
    all along cannot demonstrate that the oracle enforces something
    visible information does NOT establish, regardless of what the
    model's own explanation claims."""
    if not sub.found:
        return None
    if not _is_witness_genuinely_hidden(adapter, task, sub.test_source):
        return (
            "Rejected: found=true, but every assertion in this test_source already appears in material that "
            "was VISIBLE to a repair system all along (issue text / visible source / visible tests) — this "
            "cannot serve as a Witness, since a Witness must demonstrate the oracle enforcing something "
            "visible information does NOT establish. If a genuinely hidden test exists, find and cite that "
            "one instead; otherwise submit found=false."
        )
    run_test_calls = [c for c in calls if c.name == "run_test"]

    def _confirmed(against: str, expect: str) -> bool:
        return any(
            c.input.get("against") == against
            and c.input.get("test_source", "").strip() == sub.test_source.strip()
            and c.output.startswith(f"Running against {against}: {expect}")
            for c in run_test_calls
        )

    if not _confirmed("gold", "PASSED"):
        return (
            "Rejected: found=true, but you never called run_test with against='gold' using this EXACT "
            "test_source (byte-for-byte the same string) and got PASSED. Call run_test with your exact "
            "test_source against 'gold' and confirm PASSED before submitting found=true."
        )
    if not _confirmed("buggy", "FAILED"):
        return (
            "Rejected: found=true, but you never called run_test with against='buggy' using this EXACT "
            "test_source (byte-for-byte the same string) and got FAILED. Call run_test with your exact "
            "test_source against 'buggy' and confirm FAILED before submitting found=true."
        )
    return None


def _any_candidate_compiled(calls: list[ExecutedToolCall]) -> bool:
    """Sec. III-D: distinguishes "a candidate was tried and never
    discriminated" (NO EVIDENCE OF GAP) from "no candidate ever even
    compiled" (INCONCLUSIVE) — derived mechanically from the executed
    run_test calls, not self-reported."""
    for c in calls:
        if c.name != "run_test":
            continue
        if "PASSED" in c.output:
            return True
        if "FAILED" in c.output and "compile failed" not in c.output.lower():
            return True
    return False


def _submission_to_witness(sub: WitnessSubmission, kind: WitnessKind) -> Witness:
    if not sub.found:
        return Witness(kind=WitnessKind.NONE, explanation=sub.explanation)
    return Witness(
        kind=kind,
        test_name=sub.test_name,
        test_source=sub.test_source,
        gold_result=sub.gold_result,
        buggy_result=sub.buggy_result,
        explanation=sub.explanation,
    )


def run_oracle_search(
    task: Task,
    target: Requirement,
    adapter: BenchmarkAdapter,
    llm: LLMProvider,
    notebook: list[NotebookEntry] | None = None,
) -> tuple[Witness, list[NotebookEntry]]:
    oracle_result = adapter.run_oracle_on_buggy(task)
    failing = "\n".join(oracle_result.failing_tests) or "(none recorded)"
    user = f"""\
Target requirement [{target.id}] ({target.subject}): {target.statement}

Tests currently failing against the buggy program:
{failing}

## Notes from earlier stages (Task Notebook)
{_render_notebook(notebook or [])}

Investigate and submit a witness (or found=false) as instructed.
"""
    tools = [
        _list_test_files_tool(adapter, task),
        _read_test_file_tool(adapter, task),
        _run_test_tool(adapter, task),
    ]
    try:
        result, _calls = run_agent(
            llm,
            system=ORACLE_SEARCH_SYSTEM,
            user=user,
            tools=tools,
            submit_name=ORACLE_SEARCH_SCHEMA_NAME,
            submit_model_cls=WitnessSubmission,
            max_turns=_MAX_TURNS,
            validate_submission=lambda sub, calls: _validate_witness_submission(
                sub, calls, adapter=adapter, task=task
            ),
        )
    except AgentBudgetExceededError as exc:
        # Sec. III-D: "nothing in the oracle genuinely tests r" is a
        # normal, expected outcome — running out of exploration budget
        # without ever concluding that is functionally the same outcome
        # (the agent gave up without finding one), not a pipeline crash.
        witness = Witness(kind=WitnessKind.NONE, explanation=f"Exploration budget exhausted: {exc}")
        note = NotebookEntry(
            author="Oracle Search Agent",
            content=f"Searched oracle for [{target.id}]: gave up after exhausting its exploration budget "
            f"without concluding ({len(exc.executed_calls)} tool calls made).",
        )
        return witness, [note]
    witness = _submission_to_witness(result, WitnessKind.ORACLE)
    note = NotebookEntry(
        author="Oracle Search Agent",
        content=f"Searched oracle for [{target.id}]: {'found a match' if result.found else result.explanation}",
    )
    return witness, [note]


def run_witness_synthesis(
    task: Task,
    target: Requirement,
    hunks: list[DiffHunk],
    checked_ids: list[str],
    adapter: BenchmarkAdapter,
    llm: LLMProvider,
    notebook: list[NotebookEntry] | None = None,
) -> tuple[Witness, list[NotebookEntry], bool]:
    """`hunks` is the already-parsed developer-patch hunk catalogue.
    Resolving target.hunk_ids against it gives the agent gold's and
    buggy's real code directly, so the common case needs no
    re-exploration and the alternative behavior stays grounded in real
    code. `checked_ids` are the S_vis requirement ids that Stage 2
    (Sec. III-C) already found insufficient for `target`, so the
    synthesized test does not accidentally contradict an
    already-established requirement.

    Returns (witness, new_notebook_entries, any_candidate_compiled) —
    the third value lets the pipeline distinguish NO EVIDENCE OF GAP
    from INCONCLUSIVE per Sec. III-D, derived mechanically rather than
    self-reported (see _any_candidate_compiled)."""
    resolved = [h for hid in target.hunk_ids if (h := find_hunk(hunks, hid)) is not None]
    code_blocks = "\n\n".join(
        f"[{h.hunk_id}] {h.file_path}\n"
        f"--- gold (current) ---\n{h.new_block}\n"
        f"--- buggy (pre-fix, for reference) ---\n{h.old_block}"
        for h in resolved
    ) or "(no hunk_ids resolved for this requirement)"
    checked_block = ", ".join(checked_ids) if checked_ids else "(none recorded)"
    tools = [
        _run_test_tool(adapter, task),
        _read_buggy_file_tool(adapter, task),
        _read_gold_file_tool(adapter, task),
        _list_files_tool(adapter, task),
        _read_visible_file_tool(adapter, task),
    ]
    user = f"""\
Target requirement [{target.id}] ({target.subject}): {target.statement}

S_vis requirements already checked and found insufficient for this target: {checked_block}

## Gold implementation at this requirement's location(s)
{code_blocks}

## Notes from earlier stages (Task Notebook)
{_render_notebook(notebook or [])}

Write and iteratively test a candidate witness as instructed. Additional \
buggy/gold files can be read via your tools if the code above references \
something you need more context on.
"""
    try:
        result, calls = run_agent(
            llm,
            system=WITNESS_SYNTHESIS_SYSTEM,
            user=user,
            tools=tools,
            submit_name=WITNESS_SYNTHESIS_SCHEMA_NAME,
            submit_model_cls=WitnessSubmission,
            max_turns=_MAX_SYNTHESIS_TURNS,
            validate_submission=lambda sub, calls: _validate_witness_submission(
                sub, calls, adapter=adapter, task=task
            ),
        )
    except AgentBudgetExceededError as exc:
        # Sec. III-D already distinguishes "tried and never
        # discriminated" from "no candidate ever compiled" via
        # _any_candidate_compiled — exhausting the exploration budget
        # without ever submitting deserves the exact same treatment,
        # computed from whatever run_test calls it did manage before
        # running out, not a pipeline crash that discards that work.
        witness = Witness(kind=WitnessKind.NONE, explanation=f"Exploration budget exhausted: {exc}")
        note = NotebookEntry(
            author="Witness Synthesis Agent",
            content=f"Synthesized for [{target.id}]: gave up after exhausting its exploration budget "
            f"without concluding ({len(exc.executed_calls)} tool calls made).",
        )
        return witness, [note], _any_candidate_compiled(exc.executed_calls)
    witness = _submission_to_witness(result, WitnessKind.SYNTHESIZED)
    note = NotebookEntry(
        author="Witness Synthesis Agent",
        content=f"Synthesized for [{target.id}]: {'succeeded' if result.found else result.explanation}",
    )
    return witness, [note], _any_candidate_compiled(calls)
