"""Verso pipeline orchestration (Sec. III of the paper).

Two independent extraction subagents -> one narrow Entailment Judge call
-> Oracle Search -> Witness Synthesis fallback, with a shared, append-only,
task-scoped task notebook threaded through as input to every stage that
runs after the one that wrote it (Sec. III-D), not merely collected for
the final report.
"""

from __future__ import annotations

from verso.adapters.base import BenchmarkAdapter, ExecutionUnavailableError, Task
from verso.agents.entailment import run_entailment_judgment
from verso.agents.extraction import run_s_impl_extraction, run_s_vis_extraction
from verso.agents.witness_finding import run_oracle_search, run_witness_synthesis
from verso.diffutils import parse_unified_diff
from verso.llm.agent_loop import AgentBudgetExceededError
from verso.llm.base import LLMProvider
from verso.models import (
    AuditRecord,
    Confidence,
    EntailmentResult,
    EntailmentStatus,
    NotebookEntry,
    Requirement,
    Verdict,
    Witness,
    WitnessKind,
)

# UNENTAILED (false) is a stronger signal than UNCERTAIN, so it is
# targeted first when several requirements are flagged — see Sec. III-C.
_PRIORITY = {EntailmentStatus.FALSE: 0, EntailmentStatus.UNCERTAIN: 1}


def _select_target(
    s_impl: list[Requirement], entailments: list[EntailmentResult]
) -> tuple[Requirement, EntailmentResult] | None:
    by_id = {r.id: r for r in s_impl}
    flagged = [e for e in entailments if e.entailed != EntailmentStatus.TRUE and e.requirement_id in by_id]
    if not flagged:
        return None
    flagged.sort(key=lambda e: _PRIORITY.get(e.entailed, 1))
    chosen = flagged[0]
    return by_id[chosen.requirement_id], chosen


def audit_task(task_id: str, *, llm: LLMProvider, adapter: BenchmarkAdapter) -> AuditRecord:
    task = adapter.load_task(task_id)
    try:
        return _audit_task(task_id, task, llm=llm, adapter=adapter)
    except ExecutionUnavailableError as exc:
        # Sec. III-E/base.py: an expected, handled condition — the
        # environment can't actually execute/inspect something right
        # now (e.g. no `defects4j` on PATH) — not a bug, and not
        # something any verdict above INCONCLUSIVE could honestly claim.
        return AuditRecord(
            task_id=task_id,
            verdict=Verdict.INCONCLUSIVE,
            rationale=f"Execution unavailable in this environment: {exc}",
            confidence=Confidence.LOW,
        )
    except AgentBudgetExceededError as exc:
        # Sec. III-A: every subagent is bounded, so exhausting that
        # bound is an anticipated outcome. Oracle Search / Witness
        # Synthesis already degrade gracefully on their own (they have a
        # normal "give up" outcome to fall back to); only S_vis/S_impl
        # Extraction have no such fallback — without a Requirement list
        # there is nothing to honestly claim beyond INCONCLUSIVE.
        return AuditRecord(
            task_id=task_id,
            verdict=Verdict.INCONCLUSIVE,
            rationale=f"A subagent exhausted its exploration budget without concluding: {exc}",
            confidence=Confidence.LOW,
        )


def _audit_task(task_id: str, task: Task, *, llm: LLMProvider, adapter: BenchmarkAdapter) -> AuditRecord:
    notebook: list[NotebookEntry] = []

    s_vis, notes = run_s_vis_extraction(task, adapter, llm, notebook=notebook)
    notebook.extend(notes)

    s_impl, notes = run_s_impl_extraction(task, adapter, llm, notebook=notebook)
    notebook.extend(notes)

    entailments = run_entailment_judgment(s_vis, s_impl, llm)

    selection = _select_target(s_impl, entailments)
    if selection is None:
        return AuditRecord(
            task_id=task.task_id,
            s_vis=s_vis,
            s_impl=s_impl,
            entailments=entailments,
            witness=Witness(),
            notebook=notebook,
            verdict=Verdict.NO_EVIDENCE_OF_GAP,
            rationale="Every S_impl requirement is entailed by S_vis; no requirement to investigate.",
            confidence=Confidence.HIGH,
        )
    target, target_entailment = selection

    witness, notes = run_oracle_search(task, target, adapter, llm, notebook=notebook)
    notebook.extend(notes)

    any_candidate_compiled = True  # not applicable unless Witness Synthesis actually runs
    if witness.kind == WitnessKind.NONE:
        hunks = parse_unified_diff(adapter.developer_patch(task))
        witness, notes, any_candidate_compiled = run_witness_synthesis(
            task, target, hunks, target_entailment.checked_ids, adapter, llm, notebook=notebook
        )
        notebook.extend(notes)

    if witness.kind != WitnessKind.NONE:
        verdict = Verdict.SPECIFICATION_GAP
        confidence = Confidence.HIGH if witness.kind == WitnessKind.ORACLE else Confidence.MEDIUM
        rationale = (
            f"[{target.id}] ({target.subject}): {target.statement} — not entailed by S_vis "
            f"({target_entailment.rationale}); confirmed by a {witness.kind.value.lower()} witness: "
            f"{witness.explanation}"
        )
    elif not any_candidate_compiled:
        # Sec. III-D: "no candidate ever compiled at all" is distinct
        # from "a candidate was tried and never discriminated" — the
        # former means Witness Synthesis was never actually able to test
        # anything, so there is no basis for a NO EVIDENCE OF GAP claim.
        verdict = Verdict.INCONCLUSIVE
        confidence = Confidence.LOW
        rationale = (
            f"[{target.id}] ({target.subject}) was flagged as {target_entailment.entailed.value} by S_vis, "
            f"Oracle Search found no match, and no synthesized candidate ever compiled successfully: "
            f"{witness.explanation}"
        )
    else:
        verdict = Verdict.NO_EVIDENCE_OF_GAP
        confidence = Confidence.MEDIUM
        rationale = (
            f"[{target.id}] ({target.subject}) was flagged as {target_entailment.entailed.value} by S_vis, "
            f"but neither Oracle Search nor Witness Synthesis found a confirming witness: {witness.explanation}"
        )

    return AuditRecord(
        task_id=task.task_id,
        s_vis=s_vis,
        s_impl=s_impl,
        entailments=entailments,
        target_requirement_id=target.id,
        witness=witness,
        notebook=notebook,
        verdict=verdict,
        rationale=rationale,
        confidence=confidence,
    )
