"""Fig. 5: resolve rate on confirmed-gap vs. clean tasks for the 15
SWE-bench Verified systems, with 95% Wilson intervals, the official
leaderboard rate, the gap-clean difference, and official vs. clean rank.

Inputs: audits/swebench_verified/*.json and expert_labels/final_labels.csv
for the task split, data/swebench/leaderboard_results_15systems.json for
the per-system resolved sets.
Requires matplotlib, seaborn, scipy. Output: figures/fig5_systems.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import ROOT, audits, leaderboard, verified_task_groups  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact

OUT = ROOT / "figures" / "fig5_systems.pdf"

GAP_COLOR = "#eb6834"    # categorical slot 2 (orange)
CLEAN_COLOR = "#2a78d6"  # categorical slot 1 (blue)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Concise "scaffold + model" names, keyed by the submission id.
NAMES = {
    "20231010_rag_claude2": "RAG + Claude 2",
    "20240402_sweagent_gpt4": "SWE-agent + GPT-4",
    "20240620_sweagent_claude3.5sonnet": "SWE-agent + Claude 3.5 Sonnet",
    "20240728_sweagent_gpt4o": "SWE-agent + GPT-4o",
    "20241028_agentless-1.5_gpt4o": "Agentless 1.5 + GPT-4o",
    "20241029_OpenHands-CodeAct-2.1-sonnet-20241022": "OpenHands 2.1 + Claude 3.5 Sonnet",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless 1.5 + Claude 3.5 Sonnet",
    "20250110_blackboxai_agent_v1.1": "Blackbox AI Agent",
    "20250224_tools_claude-3-7-sonnet": "Tools + Claude 3.7 Sonnet",
    "20250415_openhands": "OpenHands",
    "20250522_sweagent_claude-4-sonnet-20250514": "SWE-agent + Claude 4 Sonnet",
    "20250726_mini-v1.0.0_claude-sonnet-4-20250514": "mini-SWE-agent + Claude 4 Sonnet",
    "20251015_Prometheus_v1.2.1_gpt5": "Prometheus 1.2.1 + GPT-5",
    "20251124_mini-v1.16.0_claude-opus-4-5-20251101": "mini-SWE-agent + Claude 4.5 Opus",
    "20260217_mini-v2.0.0_claude-4-6-opus": "mini-SWE-agent + Claude 4.6 Opus",
}


def task_split() -> tuple[set[str], set[str], set[str]]:
    """Gap and clean task ids of SWE-bench Verified (Sec. IV-E), plus all
    500 task ids, over which the official leaderboard rate is computed."""
    gap, clean = verified_task_groups()
    every = {k.split("::", 1)[1] for k in audits("SWE-bench Verified")}
    return gap, clean, every


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def competition_rank(values: list[float]) -> list[int]:
    """1-based rank, ties share the best rank (1, 1, 3, ...)."""
    return [1 + sum(v > x for v in values) for x in values]


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% hybrid score interval (Newcombe) for p1 - p2, in percentage points."""
    p1, p2 = 100 * k1 / n1, 100 * k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d - np.hypot(p1 - l1, u2 - p2), d + np.hypot(u1 - p1, p2 - l2)


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def load_rows() -> tuple[list[dict], int, int]:
    gap, clean, every = task_split()
    systems = leaderboard()
    rows = []
    for sid, s in systems.items():
        resolved = set(s["resolved"])
        kg, kc = len(resolved & gap), len(resolved & clean)
        _, p = fisher_exact([[kg, len(gap) - kg], [kc, len(clean) - kc]])
        rows.append({
            "name": NAMES[sid], "date": s["date"][:7],
            "gap": 100 * kg / len(gap), "gap_ci": wilson(kg, len(gap)),
            "clean": 100 * kc / len(clean), "clean_ci": wilson(kc, len(clean)),
            "official": 100 * len(resolved & every) / len(every), "p": p,
            "delta_ci": newcombe(kc, len(clean), kg, len(gap)),
        })
    rows.sort(key=lambda r: r["date"])
    for key, rank in (("official", "rank_official"), ("clean", "rank_clean")):
        for r, rk in zip(rows, competition_rank([round(r[key], 6) for r in rows])):
            r[rank] = rk
    return rows, len(gap), len(clean)


def main() -> None:
    rows, n_gap, n_clean = load_rows()
    sns.set_theme(context="paper", style="white", rc={
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "xtick.major.size": 2.5, "xtick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    })
    band = lambda c: sns.light_palette(c, 4)[1]  # light tint of a series color

    fig = plt.figure(figsize=(7.16, 3.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 1.3, 0.34], wspace=0.1,
                          left=0.215, right=0.995, top=0.86, bottom=0.12)
    ax, bx, rx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ys = np.arange(len(rows))[::-1]  # oldest system on top

    for a in (ax, bx, rx):
        a.set_ylim(-0.7, len(rows) - 0.3)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.tick_params(axis="y", length=0)
    for a, ticks in ((ax, range(0, 101, 20)), (bx, range(0, 51, 10))):
        for x in ticks:
            a.axvline(x, color=GRID, lw=0.5, zorder=0)

    # (a) Dumbbells: rate on gap vs. clean tasks, with 95% Wilson bands.
    for r, y in zip(rows, ys):
        for key, color in (("gap", GAP_COLOR), ("clean", CLEAN_COLOR)):
            lo, hi = r[f"{key}_ci"]
            ax.plot([lo, hi], [y, y], color=band(color), lw=4.2, solid_capstyle="round", zorder=1)
        ax.plot([r["gap"], r["clean"]], [y, y], color="#8d8c87", lw=0.8, zorder=2)
        ax.plot([r["official"]] * 2, [y - 0.27, y + 0.27], color=INK, lw=0.9, zorder=3)
        ax.plot(r["gap"], y, "D", ms=4.2, color=GAP_COLOR, mec="white", mew=0.5, zorder=4)
        ax.plot(r["clean"], y, "o", ms=4.8, color=CLEAN_COLOR, mec="white", mew=0.5, zorder=4)
    ax.set_yticks(ys, [r["name"] for r in rows])
    ax.tick_params(axis="y", pad=30)
    tf = ax.get_yaxis_transform()
    for r, y in zip(rows, ys):
        ax.text(-0.012, y - 0.04, r["date"], transform=tf, ha="right", va="center", fontsize=6, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_xlabel("Resolve rate")

    # (b) Difference (clean minus gap) with 95% Newcombe intervals.
    deltas = [r["clean"] - r["gap"] for r in rows]
    for r, y, d in zip(rows, ys, deltas):
        lo, hi = r["delta_ci"]
        bx.barh(y, d, height=0.56, color="#c9c8c2", edgecolor="none", zorder=1)
        bx.plot([lo, hi], [y, y], color=INK, lw=0.6, zorder=2)
        bx.text(61, y, f"{d:.1f}", ha="right", va="center", fontsize=6.3, color=INK)
        bx.text(61.4, y + 0.08, stars(r["p"]), ha="left", va="center", fontsize=6, color=MUTED)
    mean = float(np.mean(deltas))
    bx.axvline(mean, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    bx.text(mean + 1, len(rows) - 0.55, f"mean {mean:.1f}", ha="left", va="bottom", fontsize=6, color=INK)
    bx.set_xlim(0, 67)
    bx.set_xticks(range(0, 51, 10))
    bx.set_yticks([])
    bx.set_xlabel("Difference (pp)")

    # Rank on the official leaderboard vs. on clean tasks.
    rx.set_xlim(0, 1)
    rx.set_xticks([])
    rx.set_yticks([])
    rx.spines["bottom"].set_visible(False)
    for r, y in zip(rows, ys):
        o, c = r["rank_official"], r["rank_clean"]
        mark = "" if o == c else (" ↑" if c < o else " ↓")
        rx.text(0.5, y, f"{o}→{c}{mark}", ha="center", va="center", fontsize=6.3,
                color=INK if mark else MUTED, fontweight="bold" if mark else "normal")

    for a, label, title in ((ax, "a", "Gap vs. clean tasks"), (bx, "b", "Difference (95% CI)"), (rx, "", "Rank")):
        a.set_title(f"$\\bf{{{label}}}$  {title}" if label else title, loc="left" if label else "center",
                    fontsize=7, pad=4, color=INK)

    handles = [
        Line2D([], [], ls="", marker="D", ms=4.2, color=GAP_COLOR, mec="white", label=f"Confirmed-gap tasks ($n$ = {n_gap})"),
        Line2D([], [], ls="", marker="o", ms=4.8, color=CLEAN_COLOR, mec="white", label=f"Clean tasks ($n$ = {n_clean})"),
        Line2D([], [], ls="", marker="|", ms=7, mew=0.9, color=INK, label="Official rate (all 500 tasks)"),
        (Line2D([], [], color=band(GAP_COLOR), lw=4.2, solid_capstyle="round"),
         Line2D([], [], color=band(CLEAN_COLOR), lw=4.2, solid_capstyle="round")),
    ]
    labels = [h.get_label() for h in handles[:3]] + ["95% Wilson CI"]
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
               loc="upper left", bbox_to_anchor=(0.215, 1.0), ncol=4, frameon=False,
               handletextpad=0.6, columnspacing=1.6, handlelength=2.0)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    for r in rows:
        print(f"{r['date']} {r['name']:36s} gap {r['gap']:5.1f} clean {r['clean']:5.1f} "
              f"d {r['clean']-r['gap']:5.1f} [{r['delta_ci'][0]:.1f},{r['delta_ci'][1]:.1f}] "
              f"rank {r['rank_official']}->{r['rank_clean']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
