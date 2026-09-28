"""Reproduce the result tables of the paper from the released data.

  tables/table5_prevalence.csv     Table V, prevalence of specification gaps
  tables/table7_validation.csv     Table VII, expert validation results
  tables/fig4_gap_types.csv        the counts behind Fig. 4

Usage: python scripts/make_tables.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import (  # noqa: E402
    BENCHMARKS, CRITERIA, GAP_TYPES, ROOT, audits, expert_labels, gap_types, write_csv,
)

OUT = ROOT / "tables"


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}" if n else ""


def table5(records: dict[str, dict], labels: dict[str, dict]) -> list[list]:
    rows = []
    totals = Counter()
    for bench in BENCHMARKS:
        keys = [k for k in records if k.startswith(f"{bench}::")]
        recs = [records[k] for k in keys]
        n = len(recs)
        inconclusive = sum(r["verdict"] == "INCONCLUSIVE" for r in recs)
        s_vis = sum(len(r["s_vis"] or []) for r in recs)
        s_impl = sum(len(r["s_impl"] or []) for r in recs)
        unentailed = sum(e["entailed"] != "true" for r in recs for e in (r["entailments"] or []))
        candidate = sum(any(e["entailed"] != "true" for e in (r["entailments"] or [])) for r in recs)
        witnessed = [k for k in keys if records[k]["verdict"] == "SPECIFICATION GAP"]
        kinds = Counter(records[k]["witness"]["kind"] for k in witnessed)
        confirmed = sum(labels[k]["confirmed"] for k in witnessed)
        rows.append([bench, n, inconclusive, s_vis, s_impl, unentailed, pct(unentailed, s_impl),
                     kinds["ORACLE"], kinds["SYNTHESIZED"], candidate, pct(candidate, n),
                     len(witnessed), pct(len(witnessed), n), confirmed, pct(confirmed, n)])
        for name, value in zip(
            ["tasks", "inconclusive", "s_vis", "s_impl", "unentailed", "oracle", "synthesized",
             "candidate", "witnessed", "confirmed"],
            [n, inconclusive, s_vis, s_impl, unentailed, kinds["ORACLE"], kinds["SYNTHESIZED"],
             candidate, len(witnessed), confirmed]):
            totals[name] += value
    rows.append(["Total", totals["tasks"], totals["inconclusive"], totals["s_vis"], totals["s_impl"],
                 totals["unentailed"], pct(totals["unentailed"], totals["s_impl"]),
                 totals["oracle"], totals["synthesized"],
                 totals["candidate"], pct(totals["candidate"], totals["tasks"]),
                 totals["witnessed"], pct(totals["witnessed"], totals["tasks"]),
                 totals["confirmed"], pct(totals["confirmed"], totals["tasks"])])
    return rows


def table6(labels: dict[str, dict]) -> list[list]:
    groups = [(b, [k for k, v in labels.items() if v["benchmark"] == b]) for b in BENCHMARKS]
    groups += [("Oracle witness", [k for k, v in labels.items() if v["witness_kind"] == "oracle"]),
               ("Synthesized witness", [k for k, v in labels.items() if v["witness_kind"] == "synthesized"]),
               ("Total", list(labels))]
    rows = []
    for name, keys in groups:
        n = len(keys)
        yes = {c: sum(labels[k][c] == "Yes" for k in keys) for c in CRITERIA}
        gap_valid = sum(all(labels[k][c] == "Yes" for c in CRITERIA[:3]) for k in keys)
        witness_valid = sum(all(labels[k][c] == "Yes" for c in CRITERIA[3:]) for k in keys)
        confirmed = sum(labels[k]["confirmed"] for k in keys)
        rows.append([name, n, *[pct(yes[c], n) for c in CRITERIA[:3]], pct(gap_valid, n),
                     *[pct(yes[c], n) for c in CRITERIA[3:]], pct(witness_valid, n),
                     confirmed, pct(confirmed, n)])
    return rows


def gap_type_table(types: dict[str, dict]) -> list[list]:
    rows = []
    scopes = [("All benchmarks", list(types))] + [
        (b, [k for k, v in types.items() if v["benchmark"] == b]) for b in BENCHMARKS]
    for scope, keys in scopes:
        counts = Counter(types[k]["gap_type"] for k in keys)
        edge = Counter(types[k]["gap_type"] for k in keys if types[k]["edge_case"])
        for t in GAP_TYPES:
            rows.append([scope, t, counts[t], pct(counts[t], len(keys)), edge[t], pct(edge[t], counts[t])])
        n_edge = sum(types[k]["edge_case"] for k in keys)
        rows.append([scope, "all types", len(keys), "100.0", n_edge, pct(n_edge, len(keys))])
    return rows


def main() -> None:
    records, labels, types = audits(), expert_labels(), gap_types()

    rows5 = table5(records, labels)
    write_csv(OUT / "table5_prevalence.csv",
              ["benchmark", "tasks", "inconclusive", "s_vis", "s_impl", "unentailed", "unentailed_%",
               "oracle_witnesses", "synthesized_witnesses", "candidate", "candidate_%",
               "witnessed", "witnessed_%", "confirmed", "confirmed_%"], rows5)

    rows6 = table6(labels)
    write_csv(OUT / "table7_validation.csv",
              ["subset", "witnessed", "C1_%", "C2_%", "C3_%", "gap_validity_all_%",
               "C4_%", "C5_%", "C6_%", "witness_quality_all_%", "confirmed", "confirmed_%"], rows6)

    write_csv(OUT / "fig4_gap_types.csv",
              ["scope", "gap_type", "gaps", "share_%", "edge_case_gaps", "edge_case_share_%"],
              gap_type_table(types))

    print("Table V (prevalence)")
    for r in rows5:
        print(f"  {r[0]:22s} tasks={r[1]:5d} candidate={r[9]:5d} ({r[10]:>5s}%) "
              f"witnessed={r[11]:5d} ({r[12]:>5s}%) confirmed={r[13]:4d} ({r[14]:>5s}%)")
    print("\nTable VII (expert validation)")
    for r in rows6:
        print(f"  {r[0]:22s} n={r[1]:5d} gap validity={r[5]:>5s}%  witness quality={r[9]:>5s}%  "
              f"confirmed={r[10]:4d} ({r[11]:>5s}%)")
    print(f"\nwrote {OUT}/table5_prevalence.csv, table7_validation.csv, fig4_gap_types.csv")


if __name__ == "__main__":
    main()
