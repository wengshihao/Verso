"""Fig. 6: the fewer of the 15 repair systems resolve a SWE-bench Verified
task, the more likely it carries a confirmed specification gap.
(a) share of gap tasks in four groups;
(b) the same share at every level with a logistic fit.

Inputs: audits/swebench_verified/*.json, expert_labels/final_labels.csv,
data/swebench/leaderboard_results_15systems.json.
Requires matplotlib, seaborn, scipy. Output: figures/fig6_unresolved_tasks.pdf
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verso_results import ROOT, leaderboard, verified_task_groups  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

OUT = ROOT / "figures" / "fig6_unresolved_tasks.pdf"

GAP = "#eb6834"
CLEAN = "#2a78d6"
CLEAN_LIGHT = "#c9dcf3"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
N_SYSTEMS = 15
GROUPS = [("11-15", 11, 15), ("6-10", 6, 10), ("1-5", 1, 5), ("0", 0, 0)]  # most systems succeed -> none


def load() -> list[tuple[int, bool]]:
    """(number of systems that resolve the task, has a confirmed gap) per
    audited task of SWE-bench Verified."""
    gap, clean = verified_task_groups()
    solved = collections.Counter(t for s in leaderboard().values() for t in s["resolved"])
    return [(solved[t], t in gap) for t in gap | clean]


def trend_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Cochran-Armitage test for a linear trend in proportions (two-sided)."""
    p, n = y.mean(), len(y)
    z = np.sum(x * (y - p)) / np.sqrt(p * (1 - p) * (np.sum(x ** 2) - np.sum(x) ** 2 / n))
    return z, 2 * norm.sf(abs(z))


def logistic_fit(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logistic regression by IRLS; returns fitted probability and a 95% band on `grid`."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta))
        w = mu * (1 - mu)
        beta += np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    G = np.column_stack([np.ones_like(grid), grid])
    eta, se = G @ beta, np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))
    f = lambda e: 1 / (1 + np.exp(-e))
    return f(eta), f(eta - 1.96 * se), f(eta + 1.96 * se)


def main() -> None:
    rows = load()
    x = np.array([k for k, _ in rows], float)
    y = np.array([g for _, g in rows], float)
    z, p = trend_test(x, y)

    sns.set_theme(context="paper", style="white", rc={
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
    })
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 3.9), gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.78})

    # (a) Four difficulty groups, as 100% stacked bars (gap share at the bottom).
    shares, counts = [], []
    for i, (name, lo, hi) in enumerate(GROUPS):
        sel = [g for k, g in rows if lo <= k <= hi]
        n, g = len(sel), sum(sel)
        share = 100 * g / n
        shares.append(share)
        counts.append((g, n))
        if hi == 0:
            ax.axvspan(i - 0.46, i + 0.46, color="#f1f0ec", lw=0, zorder=0)
        ax.bar(i, share, width=0.58, color=GAP, edgecolor="none", zorder=2)
        ax.bar(i, 100 - share - 0.8, bottom=share + 0.8, width=0.58, color=CLEAN_LIGHT, edgecolor="none", zorder=2)
        inside = share >= 20
        label = f"{share:.1f}%" + (f"\n({g}/{n})" if hi == 0 else "")
        ax.text(i, share / 2 if inside else share + 4, label, ha="center", va="center" if inside else "bottom",
                fontsize=7.4 if hi == 0 else 6.6, fontweight="bold", color="white" if inside else INK,
                linespacing=1.3, zorder=4)
    # Rising trend across the groups, drawn through the tops of the gap segments.
    ax.plot(range(len(GROUPS)), shares, color=INK, lw=0.7, ls=(0, (2.5, 1.8)), marker="o", ms=2.2, zorder=3)
    ax.set_xticks(range(len(GROUPS)), [f"{g[0]}{' (none)' if g[2] == 0 else ''}\n$n$ = {c[1]}"
                                       for g, c in zip(GROUPS, counts)])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    ax.set_ylabel("Share of tasks")
    ax.set_xlabel("Repair systems (of 15) that resolve the task", labelpad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GAP, label="Confirmed gap"),
                       plt.Rectangle((0, 0), 1, 1, color=CLEAN_LIGHT, label="Clean")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, handlelength=1.0,
              handleheight=0.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

    # (b) Every difficulty level, with a fitted trend.
    by = collections.defaultdict(lambda: [0, 0])
    for k, g in rows:
        by[k][0] += g
        by[k][1] += 1
    ks = np.arange(N_SYSTEMS + 1)
    grid = np.linspace(0, N_SYSTEMS, 200)
    fit, lo, hi = logistic_fit(x, y, grid)
    for t in range(0, 76, 25):
        bx.axhline(t, color=GRID, lw=0.5, zorder=0)
    bx.fill_between(grid, 100 * lo, 100 * hi, color=GAP, alpha=0.14, lw=0, zorder=1)
    bx.plot(grid, 100 * fit, color=GAP, lw=1.2, zorder=2)
    sizes = np.array([by[k][1] for k in ks])
    bx.scatter(ks, [100 * by[k][0] / by[k][1] for k in ks], s=sizes * 0.55, color=GAP, edgecolor="white",
               lw=0.5, zorder=3)
    bx.set_xlim(N_SYSTEMS + 0.6, -0.6)  # most systems succeed on the left, none on the right, as in (a)
    bx.set_xticks(range(0, N_SYSTEMS + 1, 3))
    bx.set_ylim(-3, 75)
    bx.set_yticks(range(0, 76, 25), [f"{t}%" for t in range(0, 76, 25)])
    bx.set_ylabel("Tasks with a gap")
    bx.set_xlabel("Repair systems (of 15) that resolve the task")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    exp = int(np.floor(np.log10(p)))
    bx.text(N_SYSTEMS + 0.3, 72, f"Trend: $p$ < $10^{{{exp + 1}}}$", ha="left", va="top", fontsize=6.4, color=INK)
    bx.text(N_SYSTEMS + 0.3, 63, "Point area $\\propto$ tasks", ha="left", va="top", fontsize=6.2, color=MUTED)

    for a, label, title in ((ax, "a", "Where every system fails, most tasks have a gap"),
                            (bx, "b", "Gap share rises as fewer systems succeed")):
        a.text(-0.2, 1.14 if a is ax else 1.08, f"$\\bf{{{label}}}$  {title}", transform=a.transAxes,
               fontsize=7, ha="left", va="bottom")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  (z = {z:.2f}, p = {p:.1e})")


if __name__ == "__main__":
    main()
