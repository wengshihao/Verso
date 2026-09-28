"""Fig. 8: confirmed specification gaps versus the human ratings OpenAI's
annotators gave the same SWE-bench tasks when screening SWE-bench Verified.
The gap rate rises with the severity rating, yet tasks the annotators rated
as problem-free still carry gaps.

Inputs: audits/swebench_unverified/*.json, audits/swebench_verified/*.json,
expert_labels/final_labels.csv, data/swebench/openai_annotations.csv.
Requires matplotlib, seaborn, scipy. Output: figures/fig8_annotator_ratings.pdf
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

OUT = ROOT / "figures" / "fig8_annotator_ratings.pdf"

RAMP = ["#f7c9b4", "#f19a74", "#eb6834", "#b8491c"]  # severity 0 -> 3, gap hue
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
PANELS = [("false_negative", "a", "Rated unfair tests"), ("underspecified", "b", "Rated underspecified issue")]


def load() -> list[dict]:
    """One row per audited SWE-bench task that the annotators also rated."""
    confirmed = {k for k, v in expert_labels().items() if v["confirmed"]}
    with (DATA / "swebench/openai_annotations.csv").open(newline="", encoding="utf-8-sig") as fh:
        ann = {r["instance_id"]: r for r in csv.DictReader(fh)}
    rows = []
    for key, record in audits().items():
        bench, task = key.split("::", 1)
        if not bench.startswith("SWE") or task not in ann or record["verdict"] == "INCONCLUSIVE":
            continue
        rows.append({"gap": key in confirmed, **{k: int(float(ann[task][k])) for k, _, _ in PANELS}})
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def trend_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Cochran-Armitage trend test."""
    p, n = y.mean(), len(y)
    z = np.sum(x * (y - p)) / np.sqrt(p * (1 - p) * (np.sum(x ** 2) - np.sum(x) ** 2 / n))
    return 2 * norm.sf(abs(z))


def p_text(p: float) -> str:
    return f"$p$ < $10^{{{int(np.floor(np.log10(p))) + 1}}}$" if p < 1e-3 else f"$p$ = {p:.3f}"


def main() -> None:
    rows = load()
    sns.set_theme(context="paper", style="white", rc={
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.5,
        "ytick.major.width": 0.5, "ytick.major.size": 2.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.72), sharey=True, gridspec_kw={"wspace": 0.1})
    top = 70

    for ax, (key, letter, title) in zip(axes, PANELS):
        for t in (25, 50):
            ax.axhline(t, color=GRID, lw=0.5, zorder=0)
        for s in range(4):
            sel = [r["gap"] for r in rows if r[key] == s]
            n, k = len(sel), sum(sel)
            rate, (lo, hi) = 100 * k / n, wilson(k, n)
            ax.bar(s, rate, width=0.68, color=RAMP[s], edgecolor="none", zorder=2)
            ax.plot([s, s], [lo, min(hi, top - 0.5)], color=INK, lw=0.55, zorder=3)
            ax.text(s, min(hi, top - 6) + 1.5, f"{rate:.0f}%", ha="center", va="bottom", fontsize=6.2,
                    fontweight="bold" if s == 0 else "normal", color=INK, zorder=4)
            ax.text(s, 1.6, f"{n}", ha="center", va="bottom", fontsize=5.2,
                    color="white" if s >= 2 else MUTED, zorder=4)
        x = np.array([r[key] for r in rows], float)
        y = np.array([r["gap"] for r in rows], float)
        ax.set_xticks(range(4), ["0", "1", "2", "3"])
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.set_xlim(-0.52, 3.52)
        ax.set_ylim(0, top)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(-0.52, top + 3, f"$\\bf{{{letter}}}$  {title}", ha="left", va="bottom", fontsize=6.8)
        ax.text(-0.45, top - 2, f"trend {p_text(trend_p(x, y))}", ha="left", va="top", fontsize=5.9, color=MUTED)

    axes[0].set_yticks((0, 25, 50), ["0%", "25%", "50%"])
    axes[0].set_ylabel("Tasks with a confirmed gap", fontsize=6.5, labelpad=2)
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    fig.text(0.54, 0.0, "Severity rated by OpenAI's annotators (0 = none, 3 = severe)", ha="center",
             va="top", fontsize=6.3, color=INK)
    # The blind spot: tasks rated as problem-free still carry gaps.
    a0 = axes[0]
    a0.plot([0, 0], [35.5, 47.5], color=MUTED, lw=0.5)
    a0.text(-0.4, 48.5, "Rated fine, yet\n1 in 4 has a gap", ha="left", va="bottom", fontsize=5.8,
            color=INK, linespacing=1.12)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    both_zero = [r["gap"] for r in rows if r["false_negative"] == 0 and r["underspecified"] == 0]
    print(f"wrote {OUT}; tasks rated 0 on both: {sum(both_zero)}/{len(both_zero)}")


if __name__ == "__main__":
    main()
