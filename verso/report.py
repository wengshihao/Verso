"""Render an AuditRecord"""

from __future__ import annotations

import json

from verso.models import AuditRecord, WitnessKind


def render_text(record: AuditRecord) -> str:
    lines: list[str] = [record.task_id, ""]

    lines.append("Visible Specification (S_vis)")
    for r in record.s_vis:
        lines.append(f"- [{r.id}] {r.statement} ({r.subject}; {r.cited_from})")
    if not record.s_vis:
        lines.append("- (none extracted)")
    lines.append("")

    lines.append("Implemented Specification (S_impl)")
    for r in record.s_impl:
        lines.append(f"- [{r.id}] {r.statement} ({r.subject}; hunks={','.join(r.hunk_ids)})")
    if not record.s_impl:
        lines.append("- (none extracted)")
    lines.append("")

    lines.append("Target Requirement")
    if record.target_requirement_id:
        target = next((r for r in record.s_impl if r.id == record.target_requirement_id), None)
        entailment = next((e for e in record.entailments if e.requirement_id == record.target_requirement_id), None)
        if target is not None:
            lines.append(f"- [{target.id}] {target.statement}")
        if entailment is not None:
            lines.append(f"- entailed = {entailment.entailed.value}, checked_ids = {entailment.checked_ids}")
    else:
        lines.append("- (none — all requirements entailed)")
    lines.append("")

    lines.append("Witness")
    lines.append(f"- kind: {record.witness.kind.value}")
    if record.witness.kind != WitnessKind.NONE:
        lines.append(f"- test: {record.witness.test_name}")
        lines.append(f"- source:\n{record.witness.test_source}")
        lines.append(f"- gold result: {record.witness.gold_result}")
        lines.append(f"- buggy result: {record.witness.buggy_result}")
    lines.append("")

    lines.append("Verdict")
    lines.append(f"- {record.verdict.value}")
    lines.append("")

    lines.append("Rationale")
    lines.append(f"- {record.rationale}")
    lines.append("")

    lines.append("Confidence")
    lines.append(f"- {record.confidence.value}")
    lines.append("")

    if record.notebook:
        lines.append("Task Notebook")
        for entry in record.notebook:
            lines.append(f"- [{entry.author}] {entry.content}")

    return "\n".join(lines)


def render_json(record: AuditRecord) -> str:
    return json.dumps(record.model_dump(mode="json"), indent=2)
