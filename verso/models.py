"""Core data model for Verso: two structurally symmetric specifications,
S_vis and S_impl, compared for entailment, with the result confirmed by an
executable witness (Sec. II-B of the paper).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """Atomic unit of both S_vis and S_impl (Sec. II-B) — same shape
    regardless of which specification it belongs to, only the source
    differs (visible artifacts vs. the developer patch)."""

    id: str
    statement: str = Field(
        ..., description="One atomic, checkable behavioral claim — never a vague paraphrase."
    )
    subject: str = Field(
        ..., description="The class/method/API surface this claim constrains — used for mechanical "
        "subject-matching, not free-text search, everywhere this Requirement needs to be related "
        "to another artifact."
    )
    hunk_ids: list[str] = Field(
        default_factory=list, description="S_impl only: which developer-patch hunk(s) implement this."
    )
    cited_from: str = Field(
        default="", description="S_vis only: the specific visible artifact this claim was derived from."
    )


class EntailmentStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNCERTAIN = "uncertain"


class EntailmentResult(BaseModel):
    """Stage 2 output (Sec. III-C) — the only point in the pipeline where
    "does visible information cover this?" is decided."""

    requirement_id: str
    entailed: EntailmentStatus
    checked_ids: list[str] = Field(
        default_factory=list, description="Which S_vis requirement ids were compared and found insufficient."
    )
    rationale: str = ""


class WitnessKind(str, Enum):
    ORACLE = "ORACLE"
    SYNTHESIZED = "SYNTHESIZED"
    NONE = "NONE"


class Witness(BaseModel):
    """Sec. II-B — a concrete, executable test confirming gold
    satisfies a target Requirement and buggy does not. Always backed by
    real execution against both, regardless of kind."""

    kind: WitnessKind = WitnessKind.NONE
    test_name: str = ""
    test_source: str = ""
    gold_result: str = ""
    buggy_result: str = ""
    explanation: str = ""


class Verdict(str, Enum):
    NO_EVIDENCE_OF_GAP = "NO EVIDENCE OF GAP"
    SPECIFICATION_GAP = "SPECIFICATION GAP"
    INCONCLUSIVE = "INCONCLUSIVE"


class Confidence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class NotebookEntry(BaseModel):
    """Task notebook entry (Sec. III-D): append-only, task-scoped, and
    strictly factual/structural, never a judgment, so that sharing it
    across agents cannot reintroduce the cross-contamination that the
    S_vis/S_impl isolation (Sec. III-B) exists to prevent."""

    author: str
    content: str


class AuditRecord(BaseModel):
    task_id: str
    s_vis: list[Requirement] = Field(default_factory=list)
    s_impl: list[Requirement] = Field(default_factory=list)
    entailments: list[EntailmentResult] = Field(default_factory=list)
    target_requirement_id: str = ""
    witness: Witness = Field(default_factory=Witness)
    notebook: list[NotebookEntry] = Field(default_factory=list)
    verdict: Verdict
    rationale: str
    confidence: Confidence
