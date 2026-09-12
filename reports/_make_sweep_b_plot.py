"""Sweep B visualization: weighted score vs tool budget for 3 Qwen sizes.

Generates two plots:
  reports/sweep_b_score_vs_budget.{png, pdf}    — main figure (3 lines + SEM)
  reports/sweep_b_per_dim.{png, pdf}            — 5-dim breakdown
"""
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]
W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
     "impact": 1.5, "specificity": 0.5}
WS = sum(W.values())

MODELS = [
    ("qwen/qwen3.5-9b", "Qwen3.5-9B", "#fca5a5"),
    ("qwen/qwen3.5-27b", "Qwen3.5-27B", "#dc2626"),
    ("qwen/qwen3.5-397b-a17b", "Qwen3.5-397B", "#7f1d1d"),
]
BUDGETS = [1, 5, 10, 15]


def weighted(scores: dict) -> float:
    return sum(W[d] * scores.get(d, 0) for d in DIMS) / WS


def trim_mean(vals: list) -> float:
    if len(vals) < 2:
        return statistics.mean(vals) if vals else 0
    s = sorted(vals, reverse=True)
    return statistics.mean(s[1:])


def aggregate():
    db = ROOT / "data" / "results.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # (model, budget, paper, idea_index) -> [critic weighted scores]
    by_idea = defaultdict(list)
    by_idea_dim = defaultdict(list)  # (model, budget, dim) -> list of raw scores

    for r in conn.execute("""
        SELECT idea_model, budget, paper_id, idea_index, critic_model, scores_json
        FROM budget_sweep_scores WHERE scores_json IS NOT NULL
    """):
        s = json.loads(r["scores_json"])
        key = (r["idea_model"], r["budget"], r["paper_id"], r["idea_index"])
        by_idea[key].append(weighted(s))
        for d in DIMS:
            by_idea_dim[(r["idea_model"], r["budget"], d)].append(s.get(d, 0))

    # Per (model, budget): list of trimmed-mean scores per idea
    per_cell = defaultdict(list)
    for (m, b, _, _), vals in by_idea.items():
        per_cell[(m, b)].append(trim_mean(vals))

    # Tool calls and budget_util
    tool_stats = {}
    for r in conn.execute("""
        SELECT idea_model, budget, AVG(n_tool_calls), COUNT(*)
        FROM budget_sweep_ideas GROUP BY idea_model, budget
    """):
        tool_stats[(r[0], r[1])] = {"mean_tools": r[2] or 0, "n": r[3]}

    return per_cell, by_idea_dim, tool_stats


def plot_main(per_cell, tool_stats):
    fig, ax = plt.subplots(figsize=(10, 6))
    for mid, label, color in MODELS:
        means, sems = [], []
        for b in BUDGETS:
            vals = per_cell[(mid, b)]
            if not vals:
                means.append(None); sems.append(0); continue
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            sem = sd / math.sqrt(len(vals))
            means.append(m); sems.append(sem)
        ax.errorbar(BUDGETS, means, yerr=sems, label=label, color=color,
                    marker="o", linewidth=2.0, markersize=8, capsize=5,
                    capthick=1.2, elinewidth=1.2)
        # Annotate budget_util at b=15
        if (mid, 15) in tool_stats:
            tu = tool_stats[(mid, 15)]["mean_tools"] / 15 * 100
            ax.annotate(f"{tu:.0f}% util", xy=(15, means[-1]),
                        xytext=(8, -4), textcoords="offset points",
                        fontsize=8, color=color, alpha=0.8)
    ax.set_xlabel("Active-agent tool budget (max_iters)", fontsize=12)
    ax.set_ylabel("Weighted score (O×2+F+C×0.5+I×1.5+S×0.5, trimmed-mean)", fontsize=12)
    ax.set_xticks(BUDGETS)
    ax.set_xlim(0, 17)
    ax.set_title(
        "Sweep B — Tool budget saturation\n"
        "10 paper × 3 Qwen size × 4 budget × 3 idea (n=30 ideas / cell), error bars = SEM",
        fontsize=12, pad=12,
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    out_png = ROOT / "reports" / "sweep_b_score_vs_budget.png"
    out_pdf = ROOT / "reports" / "sweep_b_score_vs_budget.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_per_dim(by_idea_dim):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for di, dim in enumerate(DIMS):
        ax = axs[di]
        for mid, label, color in MODELS:
            means, sems = [], []
            for b in BUDGETS:
                vals = by_idea_dim.get((mid, b, dim), [])
                if not vals:
                    means.append(None); sems.append(0); continue
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0
                sem = sd / math.sqrt(len(vals))
                means.append(m); sems.append(sem)
            ax.errorbar(BUDGETS, means, yerr=sems, label=label if di == 0 else None,
                        color=color, marker="o", linewidth=1.6, markersize=6,
                        capsize=3, capthick=1.0, elinewidth=1.0)
        ax.set_title(dim.capitalize(), fontsize=11)
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("Budget", fontsize=10)
        if di == 0:
            ax.set_ylabel("Raw score (1–10)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(3.5, 8.5)
    axs[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle(
        "Sweep B per-dimension — Raw 1-10 scores vs budget (error bars = SEM, n=90/cell from 30 idea × 3 critic)",
        fontsize=12, y=1.02,
    )

    out_png = ROOT / "reports" / "sweep_b_per_dim.png"
    out_pdf = ROOT / "reports" / "sweep_b_per_dim.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    per_cell, by_idea_dim, tool_stats = aggregate()
    plot_main(per_cell, tool_stats)
    plot_per_dim(by_idea_dim)


if __name__ == "__main__":
    main()
