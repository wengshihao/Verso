"""Parse a unified diff into indexed hunks, and mechanically reconstruct
an "anchored" counterfactual diff from one.

The whole point: Verso computes hunk headers, line counts, and diff syntax
itself. The LLM only ever supplies plain replacement source text for an
already-located span — never a diff, never a hunk header, never a line
count. This eliminates a failure mode in which (~86% of
LLM-authored diffs failed to apply, almost always over hunk-header/context
mismatches, never over the underlying code idea being bad).

Reconstruction strategy: treat a hunk atomically. `old_block` is the
exact original text for the hunk's span (every context/removed line,
prefix stripped, in order) — literally a contiguous slice of the real
buggy file. A rebuilt hunk always emits "remove the entire old_block,
add the entire new_block" — every old line prefixed '-', every new line
prefixed '+'. This is always syntactically valid regardless of how much
of old_block a human would call "unchanged," and sidesteps having to
classify which original lines survive into the replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class DiffHunk:
    hunk_id: str  # e.g. "H0" — unique within one developer_patch, in file order
    file_path: str  # repo-relative path (no a/ b/ prefix) — matches Task.buggy_files keys
    old_start: int  # 1-based line number in the buggy file where this hunk begins
    old_block: str  # exact original (buggy-side) content for this hunk's span, "\n"-joined
    new_start: int  # 1-based line number in the fixed file where this hunk begins
    new_block: str = ""  # exact fixed/gold-side content for this hunk's span, "\n"-joined


def parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    """Parses a `diff --git`/unified-diff text into a flat, globally-indexed
    list of hunks. Multiple files and multiple hunks per file are all
    flattened into one list with sequential H0, H1, ... ids."""
    hunks: list[DiffHunk] = []
    current_file: str | None = None
    lines = diff_text.splitlines()
    i = 0
    hunk_counter = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            raw = line[len("+++ ") :].strip()
            if raw.startswith("b/"):
                raw = raw[2:]
            current_file = raw.split("\t")[0]
            i += 1
            continue

        match = _HUNK_HEADER_RE.match(line)
        if match and current_file is not None:
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            i += 1

            old_lines: list[str] = []
            new_lines: list[str] = []
            old_consumed = 0
            new_consumed = 0
            while i < len(lines) and (old_consumed < old_count or new_consumed < new_count):
                body_line = lines[i]
                if body_line.startswith(" "):
                    old_lines.append(body_line[1:])
                    new_lines.append(body_line[1:])
                    old_consumed += 1
                    new_consumed += 1
                    i += 1
                elif body_line.startswith("-"):
                    old_lines.append(body_line[1:])
                    old_consumed += 1
                    i += 1
                elif body_line.startswith("+"):
                    new_lines.append(body_line[1:])
                    new_consumed += 1
                    i += 1
                else:
                    break

            hunks.append(
                DiffHunk(
                    hunk_id=f"H{hunk_counter}",
                    file_path=current_file,
                    old_start=old_start,
                    old_block="\n".join(old_lines),
                    new_start=new_start,
                    new_block="\n".join(new_lines),
                )
            )
            hunk_counter += 1
            continue

        i += 1
    return hunks


def format_hunk_catalogue(hunks: list[DiffHunk]) -> str:
    """Human-readable listing for Stage 1's prompt — the LLM references
    hunks by id (hunk_ids on each SemanticDecision) rather than describing
    locations itself."""
    if not hunks:
        return "(no hunks parsed from developer_patch)"
    blocks = []
    for h in hunks:
        blocks.append(f"[{h.hunk_id}] {h.file_path} (buggy lines starting at {h.old_start})\n{h.old_block}")
    return "\n\n".join(blocks)


def format_hunk_catalogue_with_diff(hunks: list[DiffHunk]) -> str:
    """The S_impl Extractor needs both sides of each hunk (Sec. III-B),
    unlike a catalogue that only needs the
    buggy-side old_block as an anchor location."""
    if not hunks:
        return "(no hunks parsed from developer_patch)"
    blocks = []
    for h in hunks:
        blocks.append(
            f"[{h.hunk_id}] {h.file_path}\n"
            f"--- before (buggy, starting at line {h.old_start}) ---\n{h.old_block}\n"
            f"--- after (gold, starting at line {h.new_start}) ---\n{h.new_block}"
        )
    return "\n\n".join(blocks)


def find_hunk(hunks: list[DiffHunk], hunk_id: str) -> DiffHunk | None:
    for h in hunks:
        if h.hunk_id == hunk_id:
            return h
    return None


def build_anchored_patch(hunk: DiffHunk, new_block: str) -> str:
    """Mechanically reconstructs a syntactically valid unified diff hunk:
    remove hunk.old_block in full, add new_block in full. Headers and line
    counts are computed here, never by the model."""
    old_lines = hunk.old_block.split("\n") if hunk.old_block else []
    new_lines = new_block.split("\n") if new_block else []
    old_count = len(old_lines)
    new_count = len(new_lines)
    header = f"@@ -{hunk.old_start},{old_count} +{hunk.new_start},{new_count} @@"
    body = "\n".join([f"-{l}" for l in old_lines] + [f"+{l}" for l in new_lines])
    return (
        f"diff --git a/{hunk.file_path} b/{hunk.file_path}\n"
        f"--- a/{hunk.file_path}\n"
        f"+++ b/{hunk.file_path}\n"
        f"{header}\n"
        f"{body}\n"
    )
