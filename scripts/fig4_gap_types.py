"""Fig. 4: what the 720 confirmed specification gaps leave unstated.
(a) gap types overall, split by whether an edge-case input triggers them;
(b) share of each type within each benchmark.

Input: expert_labels/gap_types.csv.
Requires matplotlib, seaborn. Output: figures/fig4_gap_types.pdf
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import ROOT, gap_types  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Patch

OUT = ROOT / "figures" / "fig4_gap_types.pdf"

GAP = "#eb6834"          # the "gap" hue used in Fig. 5
GAP_LIGHT = "#f6c2ab"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

BENCHMARKS = ["Defects4J", "BugsInPy", "SWE-bench Unverified", "SWE-bench Verified"]
NAMES = {
    "return_value": "Return value",
    "exception": "Exception behavior",
    "side_effect": "Side effect & state",
    "api_surface": "API surface",
    "output_format": "Output format",
    "membership_order": "Membership & order",
    "type_structure": "Type & structure",
    "message_text": "Message text",
    "other": "Other",
}


def main() -> None:
    labels = [{"benchmark": v["benchmark"], "category": v["gap_type"], "edge_case": v["edge_case"]}
              for v in gap_types().values()]
    total = len(labels)
    count = collections.Counter(v["category"] for v in labels)
    edge = collections.Counter(v["category"] for v in labels if v["edge_case"])
    order = [c for c, _ in count.most_common() if c != "other"] + ["other"]

    sns.set_theme(context="paper", style="white", rc={
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "xtick.major.size": 2.5, "xtick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    })
    fig = plt.figure(figsize=(7.16, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.06,
                          left=0.13, right=0.995, top=0.76, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    sub = gs[1].subgridspec(1, len(BENCHMARKS), wspace=0.16)
    bxs = [fig.add_subplot(sub[i]) for i in range(len(BENCHMARKS))]
    # "Other" sits slightly apart from the named types.
    ys = np.array([len(order) - 1 - i + (0.25 if c != "other" else 0) for i, c in enumerate(order)])
    ylim = (-0.6, len(order) - 0.15)

    def rbar(y: float, x0: float, w: float, color: str, axis=None, r: float = 1.6, aspect: float = 0.12,
             h: float = 0.58, z: int = 2) -> None:
        if w <= 0:
            return
        (axis or ax).add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                              mutation_aspect=aspect, facecolor=color, edgecolor="none", zorder=z))

    # (a) Overall counts, split into edge-case and ordinary inputs.
    for x in range(0, 251, 50):
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    for c, y in zip(order, ys):
        e, n = edge[c], count[c]
        dim = c == "other"
        if dim:
            rbar(y, 0, n, "#b5b4af")
        else:
            rbar(y, 0, e, GAP)
            rbar(y, e + 1.2, n - e - 1.2, GAP_LIGHT)
        if e >= 22:
            ax.text(e / 2, y, f"{100 * e / n:.0f}%", ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold", zorder=3)
        ax.text(n + 4, y, f"{n}", ha="left", va="center", fontsize=6.5, color=INK)
        ax.text(n + 4 + 5.4 * len(str(n)), y, f"({100 * n / total:.1f}%)", ha="left", va="center",
                fontsize=6.2, color=MUTED)
    ax.set_yticks(ys, [NAMES[c] for c in order])
    for t, c in zip(ax.get_yticklabels(), order):
        t.set_color(MUTED if c == "other" else INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 280)
    ax.set_xticks(range(0, 251, 50))
    ax.set_xlabel("Confirmed specification gaps")
    ax.set_ylim(*ylim)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    n_edge = sum(edge.values())
    ax.legend(handles=[Patch(color=GAP, label=f"Edge-case input ({100 * n_edge / total:.0f}% of gaps)"),
                       Patch(color=GAP_LIGHT, label=f"Ordinary input ({100 * (total - n_edge) / total:.0f}%)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handlelength=1.1,
              handleheight=0.8, columnspacing=1.4, borderaxespad=0.1)

    # (b) Small multiples: share of each type within each benchmark.
    per = {b: [v for v in labels if v["benchmark"] == b] for b in BENCHMARKS}
    share = np.array([[100 * sum(v["category"] == c for v in per[b]) / len(per[b]) for b in BENCHMARKS]
                      for c in order])
    xmax = 45
    for j, (bx, bench) in enumerate(zip(bxs, BENCHMARKS)):
        for c, y, val in zip(order, ys, share[:, j]):
            dim = c == "other"
            rbar(y, 0, xmax, "#f1f0ec", axis=bx, r=0.35, aspect=0.55, h=0.5, z=1)  # track
            rbar(y, 0, max(val, 0.6), "#b5b4af" if dim else GAP, axis=bx, r=0.35, aspect=0.55, h=0.5)
            bx.text(xmax + 20, y, f"{val:.0f}%", ha="right", va="center", fontsize=6.1,
                    color=MUTED if dim or val < 10 else INK, fontweight="bold" if val >= 20 else "normal", zorder=3)
        bx.set_xlim(0, xmax + 21)
        bx.set_ylim(*ylim)
        bx.set_xticks([])
        bx.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            bx.spines[side].set_visible(False)
        nb = per[bench]
        head = bench.replace("SWE-bench ", "SWE-bench\n") if "SWE" in bench else "\n" + bench
        bx.set_title(f"{head}\n$n$ = {len(nb)}\n{100 * sum(v['edge_case'] for v in nb) / len(nb):.0f}% edge",
                     fontsize=6.2, color=INK, loc="left", pad=3, linespacing=1.15)
    hx = bxs[0]

    top = 0.945
    fig.text(0.005, top, r"$\bf{a}$  Gap types across all benchmarks ($n$ = " + f"{total})", ha="left", va="bottom", fontsize=7)
    fig.text(hx.get_position().x0 + 0.01, top, r"$\bf{b}$  Share of each type within a benchmark",
             ha="left", va="bottom", fontsize=7)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
