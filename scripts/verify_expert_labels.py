"""Check the released expert validation records against the final labels.

The script is read-only. It

  1. reads the six per-expert sheets and reports Cohen's kappa per criterion
     over the two independent labels of each case (paper Table IV);
  2. rebuilds every case from the two sheets plus the adjudicator's decision
     and compares the result with final_labels.csv.

Usage: python scripts/verify_expert_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import CRITERIA, LABELS, read_csv  # noqa: E402

EXPERTS = [f"E{i}" for i in range(1, 7)]
YES_NO = {"yes": "Yes", "no": "No"}


def norm(value: str) -> str | None:
    return YES_NO.get((value or "").strip().lower())


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items (Yes/No)."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(x == "Yes" for x in a) / n, sum(x == "Yes" for x in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    assignment = {r["case_id"]: r for r in read_csv(LABELS / "assignment.csv")}
    final = {r["case_id"]: r for r in read_csv(LABELS / "final_labels.csv")}

    labels: dict[tuple[str, str], dict[str, str]] = {}
    bad = []
    for e in EXPERTS:
        for row in read_csv(LABELS / f"expert_{e}.csv"):
            values = {}
            for c in CRITERIA:
                v = norm(row.get(c, ""))
                if v is None:
                    bad.append(f"expert_{e}.csv {row['case_id']} {c}: '{row.get(c, '')}' is not Yes/No")
                values[c] = v
            labels[(row["case_id"], e)] = values
    for line in bad[:20]:
        print("!", line)
    if bad:
        sys.exit(f"{len(bad)} cells are not Yes/No.")

    cases = list(assignment)
    print(f"{len(cases)} cases, two independent labels each, {len(final)} final labels.")

    print("\nCohen's kappa before adjudication (paper Table IV):")
    for c in CRITERIA:
        a = [labels[(cid, assignment[cid]["expert_a"])][c] for cid in cases]
        b = [labels[(cid, assignment[cid]["expert_b"])][c] for cid in cases]
        agree = 100 * sum(x == y for x, y in zip(a, b)) / len(a)
        print(f"  {c}: kappa = {kappa(a, b):.2f}   raw agreement = {agree:.1f}%")

    decision = {(r["case_id"], r["criterion"]): norm(r["final"])
                for r in read_csv(LABELS / "adjudication.csv")}
    print(f"\n{len(decision)} disagreements went to a third expert.")

    diffs = []
    for cid in cases:
        a = assignment[cid]
        for c in CRITERIA:
            la, lb = labels[(cid, a["expert_a"])][c], labels[(cid, a["expert_b"])][c]
            value = la if la == lb else decision.get((cid, c))
            if value != final[cid][c]:
                diffs.append((cid, c, value, final[cid][c]))

    if diffs:
        print(f"{len(diffs)} labels differ from final_labels.csv, e.g.:")
        for cid, c, got, want in diffs[:20]:
            print(f"  {cid} {c}: sheets say {got}, final_labels.csv says {want}")
        sys.exit(1)
    print("The expert sheets and the adjudication reproduce final_labels.csv exactly.")


if __name__ == "__main__":
    main()
