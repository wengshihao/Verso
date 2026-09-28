"""Fig. 7: where specification gaps come from. The share of tasks with a
confirmed gap rises with the number of behaviors the developer patch commits
to (|S_impl|), but not with the length of the issue.

Inputs: audits/*/*.json, expert_labels/final_labels.csv,
data/swebench/issue_lengths.csv.
Requires matplotlib, seaborn, scipy. Output: figures/fig7_gap_origins.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import ROOT, DATA, audits, expert_labels  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

OUT = ROOT / "figures" / "fig7_gap_origins.pdf"

RAMP = ["#fad7c8", "#f6bea5", "#f19a74", "#eb6834", "#cf5526", "#b8491c"]  # 1 -> 6+ behaviors
NEUTRAL = "#bdbcb6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def load() -> tuple[list[tuple[int, bool]], list[tuple[int, bool]]]:
    """(|S_impl| capped at 6, gap) for every audited task, and
    (issue length in words, gap) for the SWE-bench tasks."""
    confirmed = {k for k, v in expert_labels().items() if v["confirmed"]}
    words = {}
    with (DATA / "swebench/issue_lengths.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            words[row["instance_id"]] = int(row["issue_words"])
    by_size, by_issue = [], []
    for key, record in audits().items():
        if record["verdict"] == "INCONCLUSIVE":
            continue
        bench, task = key.split("::", 1)
        gap = key in confirmed
        if record["s_impl"]:
            by_size.append((min(len(record["s_impl"]), 6), gap))
        if bench.startswith("SWE") and task in words:
            by_issue.append((words[task], gap))
    return by_size, by_issue


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def trend_p(x, y) -> float:
    """Two-sided Cochran-Armitage trend test."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    p, n = y.mean(), len(y)
    z = np.sum(x * (y - p)) / np.sqrt(p * (1 - p) * (np.sum(x ** 2) - np.sum(x) ** 2 / n))
    return 2 * norm.sf(abs(z))


def bars(ax, groups: list[list[bool]], colors: list[str], labels: list[str], top: float) -> None:
    for t in (25, 50):
        ax.axhline(t, color=GRID, lw=0.5, zorder=0)
    for i, (g, color) in enumerate(zip(groups, colors)):
        n, k = len(g), sum(g)
        rate, (lo, hi) = 100 * k / n, wilson(k, n)
        ax.bar(i, rate, width=0.68, color=color, edgecolor="none", zorder=2)
        ax.plot([i, i], [lo, min(hi, top - 0.5)], color=INK, lw=0.55, zorder=3)
        ax.text(i, min(hi, top - 6) + 1.5, f"{rate:.0f}%", ha="center", va="bottom", fontsize=6.1, color=INK)
        dark = color in RAMP[3:]
        ax.text(i, 1.6, f"{n}", ha="center", va="bottom", fontsize=5.1, color="white" if dark else MUTED, zorder=4)
    ax.set_xticks(range(len(groups)), labels)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.52, len(groups) - 0.48)
    ax.set_ylim(0, top)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    by_size, by_issue = load()
    sns.set_theme(context="paper", style="white", rc={
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.5, "axes.linewidth": 0.5,
        "ytick.major.width": 0.5, "ytick.major.size": 2.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.5, 1.78), sharey=True,
                                 gridspec_kw={"width_ratios": [6, 4], "wspace": 0.1})
    top = 70

    # (a) Number of behaviors the developer patch commits to.
    sizes = list(range(1, 7))
    bars(ax, [[g for k, g in by_size if k == s] for s in sizes], RAMP,
         [str(s) if s < 6 else "6+" for s in sizes], top)
    ax.set_xlabel("Behaviors the fix commits to ($|S_{\\mathrm{impl}}|$)", labelpad=3)
    p_a = trend_p([k for k, _ in by_size], [g for _, g in by_size])
    ax.text(-0.45, top - 2, f"trend $p$ < $10^{{{int(np.floor(np.log10(p_a))) + 1}}}$", ha="left", va="top",
            fontsize=5.9, color=MUTED)

    # (b) Issue length in words, in quartiles (SWE-bench only).
    by_issue.sort()
    parts = np.array_split(np.arange(len(by_issue)), 4)
    groups = [[by_issue[i][1] for i in ix] for ix in parts]
    labels = [f"Q{j + 1}" for j in range(4)]
    print("issue-length quartiles (words):", [(by_issue[ix[0]][0], by_issue[ix[-1]][0]) for ix in parts])
    bars(bx, groups, [NEUTRAL] * 4, labels, top)
    bx.set_xlabel("Issue length (word quartiles)", labelpad=3)
    quart = np.concatenate([[j] * len(ix) for j, ix in enumerate(parts)])
    p_b = trend_p(quart, [g for _, g in by_issue])
    bx.text(3.45, top - 2, f"no trend ($p$ = {p_b:.2f})", ha="right", va="top", fontsize=5.9, color=MUTED)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)

    ax.set_yticks((0, 25, 50), ["0%", "25%", "50%"])
    ax.set_ylabel("Tasks with a confirmed gap", labelpad=2)
    for a, letter, title in ((ax, "a", "What the fix does"), (bx, "b", "What the issue says")):
        a.text(-0.52, top + 3, f"$\\bf{{{letter}}}$  {title}", ha="left", va="bottom", fontsize=6.6)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  (a: n = {len(by_size)}, p = {p_a:.1e}; b: n = {len(by_issue)}, p = {p_b:.2f})")


if __name__ == "__main__":
    main()
