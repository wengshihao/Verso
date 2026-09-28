"""Shared loaders for the released audit records and expert labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"
LABELS = ROOT / "expert_labels"
DATA = ROOT / "data"

# Paper order: Table III, Table V, Table VII.
BENCHMARKS = {
    "Defects4J": "defects4j",
    "BugsInPy": "bugsinpy",
    "SWE-bench Unverified": "swebench_unverified",
    "SWE-bench Verified": "swebench_verified",
}
CRITERIA = ["C1", "C2", "C3", "C4", "C5", "C6"]
GAP_TYPES = ["return_value", "exception", "side_effect", "api_surface", "output_format",
             "membership_order", "type_structure", "message_text", "other"]


def audits(benchmark: str | None = None) -> dict[str, dict]:
    """case_id -> audit record, for one benchmark or for all of them."""
    out = {}
    for bench, folder in BENCHMARKS.items():
        if benchmark and bench != benchmark:
            continue
        for path in sorted((AUDITS / folder).glob("*.json")):
            out[f"{bench}::{path.stem}"] = json.loads(path.read_text())["record"]
    return out


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV file as dicts."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def expert_labels() -> dict[str, dict]:
    """case_id -> {benchmark, witness_kind, C1..C6, confirmed} for the 1,278
    cases Verso reported as specification gaps. C1..C6 are the answers the
    experts gave; a case is confirmed when all six are Yes."""
    return {
        r["case_id"]: {
            "benchmark": r["benchmark"],
            "witness_kind": r["witness_kind"],
            **{c: r[c] for c in CRITERIA},
            "confirmed": r["confirmed"] == "Yes",
        }
        for r in read_csv(LABELS / "final_labels.csv")
    }


def gap_types() -> dict[str, dict]:
    """case_id -> {benchmark, gap_type, edge_case} for the 720 confirmed gaps,
    as the experts recorded them."""
    return {
        r["case_id"]: {
            "benchmark": r["benchmark"],
            "gap_type": r["gap_type"],
            "edge_case": r["edge_case"] == "Yes",
        }
        for r in read_csv(LABELS / "gap_types.csv")
    }


def leaderboard() -> dict[str, dict]:
    """Per-task outcomes of the 15 SWE-bench Verified leaderboard systems."""
    return json.loads((DATA / "swebench/leaderboard_results_15systems.json").read_text())


def verified_task_groups() -> tuple[set[str], set[str]]:
    """(gap tasks, clean tasks) of SWE-bench Verified, as defined in Sec. IV-E.
    Inconclusive audits belong to neither group."""
    labels = expert_labels()
    gap = {k.split("::", 1)[1] for k, v in labels.items()
           if v["benchmark"] == "SWE-bench Verified" and v["confirmed"]}
    clean = set()
    for key, record in audits("SWE-bench Verified").items():
        task = key.split("::", 1)[1]
        if task not in gap and record["verdict"] != "INCONCLUSIVE":
            clean.add(task)
    return gap, clean


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
